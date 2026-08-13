"""Generate deterministic Parquet datasets without retaining a full dataset in RAM."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.parquet_decompression import EXPERIMENT_NAME, SCHEMA_VERSION
from experiments.parquet_decompression.common import dataset_id, load_manifest

CATEGORIES = np.asarray(
    ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"],
    dtype=object,
)


def build_table(start: int, stop: int, *, profile: str, seed: int) -> pa.Table:
    row_id = np.arange(start, stop, dtype=np.int64)
    group_id = (row_id % 1024).astype(np.int32)
    value_i64 = ((row_id * 17 + seed * 101) % 1_000_003).astype(np.int64)
    value_f64 = np.sin((row_id + seed) * 0.001) * 1000.0 + group_id * 0.01
    flag = row_id % 7 == 0

    arrays: dict[str, Any] = {
        "row_id": pa.array(row_id),
        "group_id": pa.array(group_id),
        "value_i64": pa.array(value_i64),
        "value_f64": pa.array(value_f64, mask=row_id % 97 == 0),
        "flag": pa.array(flag),
    }

    if profile in {"mixed", "wide"}:
        category = CATEGORIES[(row_id + seed) % len(CATEGORIES)]
        arrays["category"] = pa.array(category, mask=row_id % 89 == 0)

    if profile == "wide":
        for index in range(16):
            values = np.cos((row_id + seed) * (index + 1) * 0.0001) * (index + 1)
            arrays[f"metric_{index:02d}"] = pa.array(
                values,
                mask=(row_id + index) % (101 + index) == 0,
            )

    return pa.table(arrays)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("PARQUET_BENCH_DATA_ROOT", "benchmark-data"))
        / "generated/parquet-decompression",
    )
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--rows-per-file", type=int, default=250_000)
    parser.add_argument("--row-group-rows", type=int, default=65_536)
    parser.add_argument("--codec", choices=["none", "snappy", "zstd"], default="zstd")
    parser.add_argument("--profile", choices=["numeric", "mixed", "wide"], default="mixed")
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("rows", "rows_per_file", "row_group_rows"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = {
        "codec": args.codec,
        "profile": args.profile,
        "row_group_rows": args.row_group_rows,
        "rows": args.rows,
        "rows_per_file": args.rows_per_file,
        "seed": args.seed,
    }
    identifier = dataset_id(config)
    output_dir = args.output_root / identifier
    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if manifest["config"] != config:
            raise RuntimeError(f"Dataset ID collision at {output_dir}")
        print(json.dumps({"status": "reused", "manifest": str(manifest_path)}, indent=2))
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite incomplete dataset directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    file_records: list[dict[str, Any]] = []
    compression = None if args.codec == "none" else args.codec

    for file_index, start in enumerate(range(0, args.rows, args.rows_per_file)):
        stop = min(start + args.rows_per_file, args.rows)
        table = build_table(start, stop, profile=args.profile, seed=args.seed)
        filename = f"part-{file_index:05d}.parquet"
        final_path = output_dir / filename
        temporary_path = output_dir / f".{filename}.tmp"
        pq.write_table(
            table,
            temporary_path,
            compression=compression,
            row_group_size=args.row_group_rows,
            use_dictionary=["category"] if "category" in table.column_names else False,
            write_statistics=True,
        )
        os.replace(temporary_path, final_path)
        metadata = pq.ParquetFile(final_path).metadata
        file_records.append(
            {
                "bytes": final_path.stat().st_size,
                "path": filename,
                "row_groups": metadata.num_row_groups,
                "rows": stop - start,
            }
        )

    sample_schema = pq.read_schema(output_dir / file_records[0]["path"])
    manifest = {
        "columns": sample_schema.names,
        "config": config,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_id": identifier,
        "elapsed_seconds": time.perf_counter() - started,
        "experiment": EXPERIMENT_NAME,
        "files": file_records,
        "parquet_bytes": sum(record["bytes"] for record in file_records),
        "schema": str(sample_schema),
        "schema_version": SCHEMA_VERSION,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "created", "manifest": str(manifest_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()
