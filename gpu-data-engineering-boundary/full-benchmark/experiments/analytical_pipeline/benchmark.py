"""Benchmark equivalent end-to-end analytical pipelines across four engines."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import psutil

from experiments.analytical_pipeline import EXPERIMENT_NAME, SCHEMA_VERSION
from experiments.analytical_pipeline.common import compare_values, read_json, write_new_json
from experiments.analytical_pipeline.queries import (
    WORKLOADS,
    dataset_paths,
    result_values,
    run_engine,
)
from experiments.parquet_decompression.telemetry import GpuTelemetrySampler, gpu_snapshot

ENGINES = ("duckdb", "polars-cpu", "polars-gpu", "cudf", "cudf-streaming")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/data/benchmarks/analytical-pipeline/raw"))
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=991)
    parser.add_argument("--telemetry-interval-ms", type=int, default=20)
    parser.add_argument("--host-label", default=os.environ.get("GPU_LAB_HOSTNAME", "unknown"))
    return parser.parse_args()


def source_git_commit() -> str:
    configured = os.environ.get("GPU_LAB_SOURCE_ID")
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def software_versions() -> dict[str, str]:
    import cudf
    import duckdb
    import polars
    import pyarrow

    return {
        "cudf": cudf.__version__,
        "duckdb": duckdb.__version__,
        "polars": polars.__version__,
        "pyarrow": pyarrow.__version__,
        "python": platform.python_version(),
    }


def synchronize(engine: str) -> None:
    if engine in {"cudf", "cudf-streaming", "polars-gpu"}:
        from numba import cuda

        cuda.synchronize()


def release_engine_memory(engine: str) -> None:
    gc.collect()
    if engine in {"cudf", "cudf-streaming", "polars-gpu"}:
        try:
            import rmm

            rmm.mr.get_current_device_resource().release()
        except (AttributeError, RuntimeError):
            pass


def run_once(
    engine: str,
    workload: str,
    fact_paths: list[str],
    dimension_path: str,
    *,
    threads: int,
    telemetry_interval_ms: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    release_engine_memory(engine)
    telemetry = GpuTelemetrySampler(telemetry_interval_ms)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    telemetry.start()
    started_ns = time.perf_counter_ns()
    table = run_engine(
        engine, fact_paths, dimension_path, workload, threads=threads
    )
    synchronize(engine)
    elapsed_seconds = (time.perf_counter_ns() - started_ns) / 1_000_000_000
    telemetry_result = telemetry.stop()
    rss_after = process.memory_info().rss
    values = result_values(table)
    return values, {
        "elapsed_seconds": elapsed_seconds,
        "gpu_telemetry": telemetry_result,
        "output_rows": table.num_rows,
        "process_rss_after_bytes": rss_after,
        "process_rss_before_bytes": rss_before,
        "process_rss_delta_bytes": rss_after - rss_before,
    }


def validate_selection(selection: list[str], allowed: tuple[str, ...], name: str) -> None:
    unknown = sorted(set(selection) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {name}: {', '.join(unknown)}")


def main() -> None:
    args = parse_args()
    engines = [item.strip() for item in args.engines.split(",") if item.strip()]
    workloads = [item.strip() for item in args.workloads.split(",") if item.strip()]
    validate_selection(engines, ENGINES, "engines")
    validate_selection(workloads, WORKLOADS, "workloads")
    if args.warmups < 0 or args.trials < 1 or args.threads < 1:
        raise ValueError("warmups must be non-negative; trials and threads must be positive")

    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    if manifest.get("experiment") != EXPERIMENT_NAME:
        raise ValueError("Unexpected dataset manifest")
    fact_paths, dimension_path = dataset_paths(manifest_path, manifest)

    reference: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    order = [(engine, workload) for workload in workloads for engine in engines]
    random.Random(args.seed).shuffle(order)

    for phase, repetitions in (("warmup", args.warmups), ("measured", args.trials)):
        for repetition in range(1, repetitions + 1):
            trial_order = list(order)
            random.Random(args.seed + repetition + (10_000 if phase == "measured" else 0)).shuffle(trial_order)
            for sequence, (engine, workload) in enumerate(trial_order):
                values, metrics = run_once(
                    engine,
                    workload,
                    fact_paths,
                    dimension_path,
                    threads=args.threads,
                    telemetry_interval_ms=args.telemetry_interval_ms,
                )
                if workload not in reference:
                    reference[workload] = values
                compare_values(reference[workload], values)
                records.append(
                    {
                        "correct": True,
                        "engine": engine,
                        "order": sequence,
                        "phase": phase,
                        "repetition": repetition,
                        "workload": workload,
                        **metrics,
                    }
                )
                print(
                    f"{phase} {repetition} {engine} {workload}: "
                    f"{metrics['elapsed_seconds']:.4f}s",
                    flush=True,
                )

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{uuid.uuid4().hex[:8]}"
    output = {
        "cache_state": "warm-os-cache",
        "dataset": manifest,
        "experiment": EXPERIMENT_NAME,
        "git_commit": source_git_commit(),
        "host": {
            "cpu_affinity": psutil.Process().cpu_affinity(),
            "cpu_count_logical": psutil.cpu_count(),
            "gpu": gpu_snapshot(),
            "label": args.host_label,
            "memory_total_bytes": psutil.virtual_memory().total,
            "platform": platform.platform(),
        },
        "policy": {
            "engines": engines,
            "cudf_spill": os.environ.get("CUDF_SPILL", "off"),
            "cudf_streaming": "one-file partitions with exact partial aggregation",
            "cudf_spill_device_limit": os.environ.get("CUDF_SPILL_DEVICE_LIMIT"),
            "polars_gpu_fallback": "disabled",
            "polars_gpu_mode": "streaming",
            "result_destination": "cpu-materialized-arrow",
            "seed": args.seed,
            "telemetry_interval_ms": args.telemetry_interval_ms,
            "threads": args.threads,
            "timed_region": "parquet-read-through-final-cpu-result",
            "trials": args.trials,
            "warmups": args.warmups,
            "workloads": workloads,
        },
        "reference_results": reference,
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "software": software_versions(),
        "trials": records,
    }
    result_path = args.output_root / f"{run_id}.json"
    write_new_json(result_path, output)
    print(json.dumps({"result": str(result_path), "run_id": run_id}, indent=2))


if __name__ == "__main__":
    main()
