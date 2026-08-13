"""Generate deterministic fact and dimension Parquet datasets in bounded chunks."""

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

from experiments.analytical_pipeline import EXPERIMENT_NAME, SCHEMA_VERSION
from experiments.analytical_pipeline.common import dataset_id, read_json

STATUSES = np.asarray(["active", "pending", "closed", "review"], dtype=object)
CHANNELS = np.asarray(["web", "store", "partner", "mobile"], dtype=object)
DESCRIPTIONS = np.asarray(
    [
        "standard order",
        "promo spring",
        "priority shipment",
        "promo loyalty",
        "manual review",
        "subscription renewal",
        "promo clearance",
        "international order",
    ],
    dtype=object,
)
SEGMENTS = np.asarray(["consumer", "small-business", "enterprise", "public", "partner"], dtype=object)
REGIONS = np.asarray(["north", "south", "east", "west", "central", "atlantic", "pacific", "remote"], dtype=object)


def build_fact_table(start: int, stop: int, *, profile: str, seed: int) -> pa.Table:
    row_id = np.arange(start, stop, dtype=np.int64)
    customer_id = ((row_id * 17 + seed) % 100_000).astype(np.int32)
    amount = (((row_id * 37 + seed * 101) % 100_000) / 100.0).astype(np.float64)
    quantity = ((row_id % 12) + 1).astype(np.int16)
    discount = ((row_id % 20) / 100.0).astype(np.float64)
    arrays: dict[str, Any] = {
        "row_id": pa.array(row_id),
        "customer_id": pa.array(customer_id),
        "product_id": pa.array(((row_id * 29 + seed) % 10_000).astype(np.int32)),
        "event_day": pa.array((row_id % 365).astype(np.int16)),
        "amount": pa.array(amount),
        "quantity": pa.array(quantity),
        "discount": pa.array(discount),
        "status": pa.array(STATUSES[(row_id + seed) % len(STATUSES)]),
        "channel": pa.array(CHANNELS[(row_id // 3 + seed) % len(CHANNELS)]),
        "description": pa.array(DESCRIPTIONS[(row_id // 11 + seed) % len(DESCRIPTIONS)]),
    }
    if profile == "wide":
        for index in range(12):
            arrays[f"metric_{index:02d}"] = pa.array(
                np.sin((row_id + seed) * (index + 1) * 0.0001) * (index + 1)
            )
    return pa.table(arrays)


def build_dimension_table(*, seed: int, customers: int = 100_000) -> pa.Table:
    customer_id = np.arange(customers, dtype=np.int32)
    return pa.table(
        {
            "customer_id": pa.array(customer_id),
            "segment": pa.array(SEGMENTS[(customer_id + seed) % len(SEGMENTS)]),
            "region": pa.array(REGIONS[(customer_id * 3 + seed) % len(REGIONS)]),
            "risk_score": pa.array(((customer_id * 13 + seed) % 1000) / 1000.0),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/data/generated/analytical-pipeline"))
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--rows-per-file", type=int, default=1_000_000)
    parser.add_argument("--row-group-rows", type=int, default=262_144)
    parser.add_argument("--codec", choices=["snappy", "zstd"], default="zstd")
    parser.add_argument("--profile", choices=["base", "wide"], default="wide")
    parser.add_argument("--seed", type=int, default=73)
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
        manifest = read_json(manifest_path)
        if manifest["config"] != config:
            raise RuntimeError(f"Dataset ID collision at {output_dir}")
        print(json.dumps({"manifest": str(manifest_path), "status": "reused"}, indent=2))
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite incomplete dataset directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fact_dir = output_dir / "fact"
    fact_dir.mkdir()
    started = time.perf_counter()
    files: list[dict[str, Any]] = []
    dictionary_columns = ["status", "channel", "description"]
    for file_index, start in enumerate(range(0, args.rows, args.rows_per_file)):
        stop = min(start + args.rows_per_file, args.rows)
        table = build_fact_table(start, stop, profile=args.profile, seed=args.seed)
        filename = f"part-{file_index:05d}.parquet"
        final_path = fact_dir / filename
        temporary_path = fact_dir / f".{filename}.tmp"
        pq.write_table(
            table,
            temporary_path,
            compression=args.codec,
            row_group_size=args.row_group_rows,
            use_dictionary=dictionary_columns,
            write_statistics=True,
        )
        os.replace(temporary_path, final_path)
        metadata = pq.ParquetFile(final_path).metadata
        files.append(
            {
                "bytes": final_path.stat().st_size,
                "path": f"fact/{filename}",
                "row_groups": metadata.num_row_groups,
                "rows": stop - start,
            }
        )

    dimension_path = output_dir / "customers.parquet"
    pq.write_table(
        build_dimension_table(seed=args.seed),
        dimension_path,
        compression=args.codec,
        row_group_size=args.row_group_rows,
        use_dictionary=["segment", "region"],
        write_statistics=True,
    )
    sample = pq.read_table(output_dir / files[0]["path"])
    logical_bytes_per_row = sample.nbytes / sample.num_rows
    manifest = {
        "columns": sample.column_names,
        "config": config,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_id": identifier,
        "dimension": {
            "bytes": dimension_path.stat().st_size,
            "path": dimension_path.name,
            "rows": 100_000,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "estimated_logical_bytes": int(logical_bytes_per_row * args.rows),
        "experiment": EXPERIMENT_NAME,
        "fact_files": files,
        "parquet_bytes": sum(item["bytes"] for item in files) + dimension_path.stat().st_size,
        "schema": str(sample.schema),
        "schema_version": SCHEMA_VERSION,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"manifest": str(manifest_path), "status": "created", **manifest}, indent=2))


if __name__ == "__main__":
    main()

