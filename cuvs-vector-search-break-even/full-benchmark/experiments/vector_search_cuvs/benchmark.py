"""Benchmark one cuVS index build and a grid of resident searches."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np
from cuvs.common import Resources

from experiments.vector_search_cuvs import EXPERIMENT_NAME, SCHEMA_VERSION
from experiments.vector_search_cuvs.algorithms import ALGORITHMS, build, save, search, search_grid
from experiments.vector_search_cuvs.common import atomic_json, read_json, recall_at_k, stable_id
from experiments.vector_search_cuvs.telemetry import GpuSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/data/benchmarks/vector-search-cuvs/raw"))
    parser.add_argument("--batch-sizes", default="1,32,256,1024")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--replication", type=int, default=0)
    parser.add_argument("--search-params", help="JSON object; omit to run the tuning grid")
    parser.add_argument("--cases-json", type=Path, help="JSON list of batch/parameter cases")
    parser.add_argument("--truth-manifest", type=Path, required=True)
    return parser.parse_args()


def source_id() -> str:
    configured = os.environ.get("GPU_LAB_SOURCE_ID")
    if configured:
        return configured
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def timed(call, resources: Resources) -> tuple[float, Any]:
    started = time.perf_counter()
    value = call()
    resources.sync()
    elapsed = time.perf_counter() - started
    return elapsed, value


def timed_with_telemetry(call, resources: Resources) -> tuple[float, Any, dict[str, Any]]:
    sampler = GpuSampler()
    sampler.start()
    elapsed, value = timed(call, resources)
    return elapsed, value, sampler.stop()


def sustained_telemetry(call, resources: Resources, minimum_seconds: float = 0.25) -> dict[str, Any]:
    sampler = GpuSampler()
    sampler.start()
    started = time.perf_counter()
    iterations = 0
    while time.perf_counter() - started < minimum_seconds:
        call()
        resources.sync()
        iterations += 1
    elapsed = time.perf_counter() - started
    return {"elapsed_seconds": elapsed, "iterations": iterations, "gpu": sampler.stop()}


def environment() -> dict[str, Any]:
    import cuvs

    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True
    ).strip()
    return {
        "cupy": cp.__version__,
        "cuvs": getattr(cuvs, "__version__", "26.08.01"),
        "gpu": gpu,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def memory_snapshot() -> dict[str, int]:
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    return {
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
        "device_used_bytes": int(total_bytes - free_bytes),
        "cupy_pool_total_bytes": int(pool.total_bytes()),
        "cupy_pool_used_bytes": int(pool.used_bytes()),
    }


def main() -> None:
    args = parse_args()
    batches = [int(value) for value in args.batch_sizes.split(",")]
    if min(batches) < 1 or args.k < 1 or args.trials < 1 or args.warmups < 0:
        raise ValueError("batch sizes, k, and trials must be positive; warmups must be non-negative")
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    rows = int(manifest["config"]["rows"])
    dimensions = int(manifest["config"]["dimensions"])
    if args.search_params and args.cases_json:
        raise ValueError("Use either --search-params or --cases-json, not both")
    if args.cases_json:
        requested_cases = read_json(args.cases_json.resolve())["cases"]
    elif args.search_params:
        requested_cases = [
            {"batch_size": batch, "params": json.loads(args.search_params)} for batch in batches
        ]
    else:
        requested_cases = [
            {"batch_size": batch, "params": params}
            for batch in batches
            for params in search_grid(args.algorithm, rows)
        ]
    requested_batches = sorted({int(case["batch_size"]) for case in requested_cases})
    if any(batch not in batches for batch in requested_batches):
        raise ValueError("cases-json includes a batch not listed in --batch-sizes")
    identity = {
        "algorithm": args.algorithm,
        "batches": batches,
        "dataset_id": manifest["dataset_id"],
        "git_commit": source_id(),
        "replication": args.replication,
        "search_cases": requested_cases,
        "trials": args.trials,
        "warmups": args.warmups,
    }
    run_id = stable_id("cuvs", identity)
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    truth_manifest = read_json(args.truth_manifest.resolve())
    if truth_manifest["dataset_id"] != manifest["dataset_id"] or int(truth_manifest["k"]) != args.k:
        raise ValueError("ground-truth manifest does not match dataset and k")
    truth = np.load(args.truth_manifest.resolve().parent / truth_manifest["neighbors"]["path"], mmap_mode="r")
    corpus_host = np.load(manifest_path.parent / manifest["corpus"]["path"], mmap_mode="r")
    queries_host = np.load(manifest_path.parent / manifest["queries"]["path"], mmap_mode="r")
    if max(batches) > len(queries_host) or max(batches) > len(truth):
        raise ValueError("dataset does not contain enough queries for requested batches")
    memory_before_load = memory_snapshot()
    cp.cuda.runtime.deviceSynchronize()
    load_started = time.perf_counter()
    dataset = cp.asarray(corpus_host)
    queries = cp.asarray(queries_host[: max(batches)])
    cp.cuda.runtime.deviceSynchronize()
    load_seconds = time.perf_counter() - load_started
    memory_after_load = memory_snapshot()
    resources = Resources()
    build_seconds, built, build_telemetry = timed_with_telemetry(
        lambda: build(args.algorithm, dataset, rows, dimensions, resources), resources
    )
    index, build_params = built
    memory_after_build = memory_snapshot()
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        index_path = Path(temporary) / "index.bin"
        save_started = time.perf_counter()
        save(args.algorithm, str(index_path), index)
        serialized_bytes = index_path.stat().st_size
        serialize_seconds = time.perf_counter() - save_started
    searches: list[dict[str, Any]] = []
    for requested in requested_cases:
        batch = int(requested["batch_size"])
        params = requested["params"]
        batch_queries = queries[:batch]
        expected = truth[:batch]
        for _ in range(args.warmups):
            search(args.algorithm, index, batch_queries, args.k, params, resources)
            resources.sync()
        trials = []
        last_neighbors = None
        for trial in range(args.trials):
            elapsed, outputs = timed(
                lambda: search(args.algorithm, index, batch_queries, args.k, params, resources), resources
            )
            _, last_neighbors = outputs
            trials.append({"elapsed_seconds": elapsed, "trial": trial})
        actual = cp.asarray(last_neighbors).get()
        recall = recall_at_k(expected, actual, args.k)
        med = statistics.median(x["elapsed_seconds"] for x in trials)
        searches.append({
            "batch_size": batch,
            "latency_ms_per_batch": med * 1000,
            "latency_ms_per_query": med * 1000 / batch,
            "params": params,
            "queries_per_second": batch / med,
            "recall_at_k": recall,
            "sustained_telemetry": sustained_telemetry(
                lambda: search(args.algorithm, index, batch_queries, args.k, params, resources), resources
            ),
            "trials": trials,
        })
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "run_id": run_id,
        "git_commit": source_id(),
        "dataset": manifest,
        "benchmark": {
            "algorithm": args.algorithm,
            "batch_sizes": batches,
            "build_params": build_params,
            "k": args.k,
            "replication": args.replication,
            "trials": args.trials,
            "warmups": args.warmups,
        },
        "environment": environment(),
        "load_seconds": load_seconds,
        "memory": {
            "before_load": memory_before_load,
            "after_load": memory_after_load,
            "after_build": memory_after_build,
        },
        "build_seconds": build_seconds,
        "build_telemetry": build_telemetry,
        "serialized_index_bytes": serialized_bytes,
        "serialize_seconds": serialize_seconds,
        "searches": searches,
        "truth_manifest": truth_manifest,
    }
    result_path = output_dir / "result.json"
    atomic_json(result_path, result)
    print(json.dumps({"result": str(result_path), "run_id": run_id}, indent=2))


if __name__ == "__main__":
    main()
