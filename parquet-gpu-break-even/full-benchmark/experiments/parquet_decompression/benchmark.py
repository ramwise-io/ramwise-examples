"""Benchmark warm-cache Parquet reads across CPU engines and cuDF."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import random
import socket
import statistics
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import psutil

from experiments.parquet_decompression import EXPERIMENT_NAME, SCHEMA_VERSION
from experiments.parquet_decompression.common import (
    compare_signatures,
    load_manifest,
    scalar,
    write_new_json,
)
from experiments.parquet_decompression.telemetry import GpuTelemetrySampler, gpu_snapshot

SUPPORTED_ENGINES = ("pyarrow", "pandas", "polars", "duckdb", "cudf")
_DUCKDB_CONNECTION: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("PARQUET_BENCH_DATA_ROOT", "benchmark-data"))
        / "benchmarks/parquet-decompression/raw",
    )
    parser.add_argument("--engines", default=",".join(SUPPORTED_ENGINES))
    parser.add_argument("--projection", default="all", help="all, core, or a comma-separated column list")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--telemetry-interval-ms", type=int, default=20)
    parser.add_argument("--matrix-id")
    parser.add_argument("--case-id")
    parser.add_argument(
        "--host-label",
        default=os.environ.get("PARQUET_BENCH_HOSTNAME", socket.gethostname()),
        help="Stable host name recorded separately from the ephemeral container hostname",
    )
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def git_commit() -> str:
    public_source_id = os.environ.get("PARQUET_BENCH_SOURCE_ID")
    if public_source_id:
        return public_source_id
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def resolve_columns(manifest: dict[str, Any], projection: str) -> list[str]:
    available = manifest["columns"]
    if projection == "all":
        return list(available)
    if projection == "core":
        selected = ["row_id", "value_i64", "value_f64", "category"]
        return [name for name in selected if name in available]
    selected = [name.strip() for name in projection.split(",") if name.strip()]
    missing = sorted(set(selected) - set(available))
    if not selected or missing:
        raise ValueError(f"Invalid projection; missing columns: {missing}")
    return selected


def configure_threads(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["POLARS_MAX_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    import pyarrow

    pyarrow.set_cpu_count(threads)


def parquet_files(dataset_dir: Path, manifest: dict[str, Any]) -> list[str]:
    return [str(dataset_dir / record["path"]) for record in manifest["files"]]


def load_frame(engine: str, files: list[str], columns: list[str], threads: int) -> Any:
    if engine == "pyarrow":
        import pyarrow.parquet as parquet

        return parquet.read_table(files, columns=columns, use_threads=True)
    if engine == "pandas":
        import pandas

        return pandas.read_parquet(files, columns=columns, engine="pyarrow")
    if engine == "polars":
        import polars

        return polars.read_parquet(files, columns=columns, parallel="auto", rechunk=True)
    if engine == "duckdb":
        import duckdb

        global _DUCKDB_CONNECTION
        if _DUCKDB_CONNECTION is None:
            _DUCKDB_CONNECTION = duckdb.connect()
            _DUCKDB_CONNECTION.execute(f"SET threads = {threads}")
        identifiers = ", ".join(f'"{name}"' for name in columns)
        glob = str(Path(files[0]).parent / "part-*.parquet")
        return _DUCKDB_CONNECTION.execute(
            f"SELECT {identifiers} FROM read_parquet(?)",
            [glob],
        ).to_arrow_table()
    if engine == "cudf":
        import cudf

        return cudf.read_parquet(files, columns=columns)
    raise ValueError(f"Unsupported engine: {engine}")


def synchronize(engine: str) -> None:
    if engine == "cudf":
        import cupy

        cupy.cuda.runtime.deviceSynchronize()


def cleanup(engine: str) -> None:
    gc.collect()
    if engine == "cudf":
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.cuda.runtime.deviceSynchronize()


def arrow_signature(table: Any) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.compute as compute

    stats: dict[str, Any] = {}
    for field, column in zip(table.schema, table.columns, strict=True):
        summary = {"nulls": int(column.null_count)}
        if pa.types.is_boolean(field.type):
            summary["sum"] = scalar(compute.sum(compute.cast(column, pa.int64())).as_py())
        elif pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            extrema = compute.min_max(column).as_py()
            summary.update(
                {
                    "max": scalar(extrema["max"]),
                    "min": scalar(extrema["min"]),
                    "sum": scalar(compute.sum(column).as_py()),
                }
            )
        else:
            extrema = compute.min_max(column).as_py()
            summary.update(
                {
                    "distinct": int(compute.count_distinct(column).as_py()),
                    "max": scalar(extrema["max"]),
                    "min": scalar(extrema["min"]),
                }
            )
        stats[field.name] = summary
    return {"columns": table.column_names, "rows": table.num_rows, "stats": stats}


def dataframe_signature(frame: Any) -> dict[str, Any]:
    from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

    columns = list(frame.columns)
    stats: dict[str, Any] = {}
    for name in columns:
        series = frame[name]
        summary = {"nulls": int(series.isna().sum())}
        if is_bool_dtype(series.dtype):
            summary["sum"] = scalar(series.sum())
        elif is_integer_dtype(series.dtype):
            summary.update(
                {
                    "max": scalar(series.max()),
                    "min": scalar(series.min()),
                    "sum": scalar(series.astype("int64").sum()),
                }
            )
        elif is_numeric_dtype(series.dtype):
            summary.update(
                {
                    "max": scalar(series.max()),
                    "min": scalar(series.min()),
                    "sum": scalar(series.sum()),
                }
            )
        else:
            summary.update(
                {
                    "distinct": int(series.nunique(dropna=True)),
                    "max": scalar(series.max()),
                    "min": scalar(series.min()),
                }
            )
        stats[name] = summary
    return {"columns": columns, "rows": len(frame), "stats": stats}


def polars_signature(frame: Any) -> dict[str, Any]:
    import polars

    columns = list(frame.columns)
    stats: dict[str, Any] = {}
    for name in columns:
        series = frame[name]
        summary = {"nulls": int(series.null_count())}
        if series.dtype == polars.Boolean:
            summary["sum"] = scalar(series.sum())
        elif series.dtype.is_integer():
            summary.update(
                {
                    "max": scalar(series.max()),
                    "min": scalar(series.min()),
                    "sum": scalar(series.cast(polars.Int64).sum()),
                }
            )
        elif series.dtype.is_numeric():
            summary.update(
                {
                    "max": scalar(series.max()),
                    "min": scalar(series.min()),
                    "sum": scalar(series.sum()),
                }
            )
        else:
            summary.update(
                {
                    "distinct": int(series.drop_nulls().n_unique()),
                    "max": scalar(series.max()),
                    "min": scalar(series.min()),
                }
            )
        stats[name] = summary
    return {"columns": columns, "rows": frame.height, "stats": stats}


def frame_signature(engine: str, frame: Any) -> dict[str, Any]:
    if engine in {"pyarrow", "duckdb"}:
        return arrow_signature(frame)
    if engine == "polars":
        return polars_signature(frame)
    if engine in {"pandas", "cudf"}:
        result = dataframe_signature(frame)
        synchronize(engine)
        return result
    raise ValueError(engine)


def warm_os_cache(files: list[str]) -> int:
    total = 0
    for filename in files:
        with open(filename, "rb", buffering=0) as handle:
            while chunk := handle.read(16 * 1024 * 1024):
                total += len(chunk)
    return total


def measured_read(
    engine: str,
    files: list[str],
    columns: list[str],
    threads: int,
    telemetry_interval_ms: int,
) -> tuple[Any, dict[str, Any]]:
    process = psutil.Process()
    rss_before = process.memory_info().rss
    gpu_before = gpu_snapshot()
    telemetry = GpuTelemetrySampler(telemetry_interval_ms)
    telemetry.start()
    started_ns = time.perf_counter_ns()
    frame = load_frame(engine, files, columns, threads)
    synchronize(engine)
    elapsed_ns = time.perf_counter_ns() - started_ns
    gpu_telemetry = telemetry.stop()
    rss_after = process.memory_info().rss
    gpu_after = gpu_snapshot()
    return frame, {
        "elapsed_seconds": elapsed_ns / 1_000_000_000,
        "gpu_after": gpu_after,
        "gpu_before": gpu_before,
        "gpu_memory_after_bytes": gpu_after.get("memory_used_bytes"),
        "gpu_memory_before_bytes": gpu_before.get("memory_used_bytes"),
        "gpu_telemetry": gpu_telemetry,
        "rss_after_bytes": rss_after,
        "rss_before_bytes": rss_before,
    }


def main() -> None:
    args = parse_args()
    if args.trials < 1 or args.warmups < 0 or args.threads < 1:
        raise ValueError("trials/threads must be positive and warmups non-negative")
    engines = [item.strip() for item in args.engines.split(",") if item.strip()]
    unknown = sorted(set(engines) - set(SUPPORTED_ENGINES))
    if not engines or unknown:
        raise ValueError(f"Unsupported engines: {unknown}")

    configure_threads(args.threads)
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    dataset_dir = manifest_path.parent
    files = parquet_files(dataset_dir, manifest)
    columns = resolve_columns(manifest, args.projection)
    parquet_bytes = int(manifest["parquet_bytes"])

    warmed_bytes = warm_os_cache(files)
    if warmed_bytes != parquet_bytes:
        raise RuntimeError(f"Warm-cache byte count mismatch: {warmed_bytes} != {parquet_bytes}")

    reference_frame = load_frame("pyarrow", files, columns, args.threads)
    expected_signature = frame_signature("pyarrow", reference_frame)
    del reference_frame
    cleanup("pyarrow")

    trials: list[dict[str, Any]] = []
    order = 0

    def run(engine: str, phase: str, repetition: int) -> None:
        nonlocal order
        order += 1
        frame, metrics = measured_read(
            engine,
            files,
            columns,
            args.threads,
            args.telemetry_interval_ms,
        )
        actual_signature = frame_signature(engine, frame)
        try:
            compare_signatures(expected_signature, actual_signature)
        except AssertionError as error:
            raise AssertionError(f"{engine} correctness failure: {error}") from error
        metrics.update(
            {
                "correct": True,
                "engine": engine,
                "order": order,
                "phase": phase,
                "repetition": repetition,
                "rows": expected_signature["rows"],
                "throughput_mib_s": parquet_bytes / (1024**2) / metrics["elapsed_seconds"],
            }
        )
        trials.append(metrics)
        del frame
        cleanup(engine)

    for engine in engines:
        for repetition in range(args.warmups):
            run(engine, "warmup", repetition)

    schedule = [(engine, repetition) for repetition in range(args.trials) for engine in engines]
    random.Random(args.seed).shuffle(schedule)
    for engine, repetition in schedule:
        run(engine, "measured", repetition)

    now = time.gmtime()
    run_id = time.strftime("%Y%m%dT%H%M%SZ", now) + f"-{uuid.uuid4().hex[:8]}"
    result = {
        "cache_state": "warm-os-cache",
        "columns": columns,
        "dataset": {
            "config": manifest["config"],
            "dataset_id": manifest["dataset_id"],
            "files": len(files),
            "parquet_bytes": parquet_bytes,
            "rows": manifest["config"]["rows"],
        },
        "experiment": EXPERIMENT_NAME,
        "case_id": args.case_id,
        "git_commit": git_commit(),
        "host": {
            "container_hostname": socket.gethostname(),
            "cpu_affinity": sorted(psutil.Process().cpu_affinity()),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_model": cpu_model(),
            "hostname": args.host_label,
            "load_average": list(os.getloadavg()),
            "memory_total_bytes": psutil.virtual_memory().total,
            "platform": platform.platform(),
        },
        "gpu": gpu_snapshot(),
        "policy": {
            "cpu_affinity": sorted(psutil.Process().cpu_affinity()),
            "engine_order": "seeded random for measured trials",
            "engine_session_policy": "imports and reusable sessions initialized by per-engine warmups",
            "gpu_synchronization": "before timer stop",
            "threads": args.threads,
            "telemetry_interval_ms": args.telemetry_interval_ms,
            "throughput_numerator": "total compressed Parquet file bytes, including unprojected columns",
            "timed_region": "read/decode/materialize to returned frame; DuckDB returns an Arrow table",
            "validation": "full-column aggregate signature outside timed region for every trial",
            "warmups_per_engine": args.warmups,
        },
        "run_id": run_id,
        "matrix_id": args.matrix_id,
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "software": {name: package_version(name) for name in (*SUPPORTED_ENGINES, "cupy", "pyarrow")},
        "trials": trials,
    }
    result_path = args.output_root / f"{run_id}.json"
    write_new_json(result_path, result)
    if _DUCKDB_CONNECTION is not None:
        _DUCKDB_CONNECTION.close()
    medians: dict[str, float] = {}
    for engine in engines:
        values = sorted(
            trial["throughput_mib_s"]
            for trial in trials
            if trial["phase"] == "measured" and trial["engine"] == engine
        )
        medians[engine] = statistics.median(values)
    print(json.dumps({"medians_mib_s": medians, "result": str(result_path), "run_id": run_id}, indent=2))


if __name__ == "__main__":
    main()
