"""Benchmark one CPU or RAPIDS Spark ETL topology in local mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from experiments.spark_rapids_fallback import EXPERIMENT_NAME, SCHEMA_VERSION
from experiments.spark_rapids_fallback.common import (
    atomic_new_json,
    canonical_json,
    compare_values,
    read_json,
    stable_id,
)
from experiments.spark_rapids_fallback.telemetry import TelemetrySampler
from experiments.spark_rapids_fallback.workload import (
    TOPOLOGIES,
    build_query,
    canonical_rows,
    plan_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--topology", choices=TOPOLOGIES, required=True)
    parser.add_argument(
        "--python-bridge", choices=["accelerated", "forced-cpu"], default="forced-cpu"
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("/data/benchmarks/spark-rapids-fallback/raw"))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--replication", type=int, default=0)
    parser.add_argument("--retry-attempt", type=int, default=0)
    parser.add_argument("--telemetry-interval-ms", type=int, default=100)
    return parser.parse_args()


def source_id() -> str:
    configured = os.environ.get("GPU_LAB_SOURCE_ID")
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def spark_session(args: argparse.Namespace, event_log_dir: Path, local_dir: Path):
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.master(f"local[{args.threads}]")
        .appName(f"{EXPERIMENT_NAME}-{args.mode}-{args.topology}")
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
        .config("spark.rapids.sql.enabled", str(args.mode == "gpu").lower())
        .config("spark.rapids.sql.explain", "ALL")
        .config("spark.rapids.sql.metrics.level", "MODERATE")
        .config("spark.rapids.sql.concurrentGpuTasks", "2")
        .config(
            "spark.rapids.sql.exec.ArrowEvalPythonExec",
            str(args.python_bridge == "accelerated").lower(),
        )
        .config("spark.rapids.memory.pinnedPool.size", "8G")
        .config("spark.rapids.sql.batchSizeBytes", "536870912")
        .config("spark.driver.memory", "80g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .config("spark.sql.files.maxPartitionBytes", "268435456")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.python.worker.reuse", "true")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", event_log_dir.as_uri())
        .config("spark.local.dir", str(local_dir))
        .config("spark.ui.enabled", "false")
    )
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")
    return session


def environment(spark: Any) -> dict[str, Any]:
    import pandas
    import pyarrow
    import pyspark

    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = "unavailable"
    return {
        "cpu_count": os.cpu_count(),
        "gpu": gpu,
        "hostname": socket.gethostname(),
        "java_version": spark.sparkContext._jvm.java.lang.System.getProperty("java.version"),
        "pandas": pandas.__version__,
        "platform": platform.platform(),
        "pyarrow": pyarrow.__version__,
        "pyspark": pyspark.__version__,
        "rapids_plugin": "26.06.1-cuda13",
        "spark": spark.version,
    }


def result_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def run_action(query: Any, interval_ms: int) -> tuple[float, dict[str, Any], dict[str, Any]]:
    telemetry = TelemetrySampler(interval_ms)
    telemetry.start()
    started = time.perf_counter()
    rows = query.collect()
    elapsed = time.perf_counter() - started
    measurements = telemetry.stop()
    return elapsed, canonical_rows(rows), measurements


def main() -> None:
    args = parse_args()
    if args.warmups < 0 or args.trials < 1:
        raise ValueError("warmups must be non-negative and trials must be positive")
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    fact_path = str(manifest_path.parent / "fact")
    dimension_path = str(manifest_path.parent / manifest["dimension"]["path"])
    identity = {
        "dataset_id": manifest["dataset_id"],
        "git_commit": source_id(),
        "mode": args.mode,
        "python_bridge": args.python_bridge,
        "replication": args.replication,
        "seed": args.seed,
        "topology": args.topology,
    }
    if args.retry_attempt:
        identity["retry_attempt"] = args.retry_attempt
    run_id = stable_id("spark", identity)
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    event_log_dir = output_dir / "eventlog"
    local_dir = output_dir / "spark-local"
    event_log_dir.mkdir()
    local_dir.mkdir()
    spark = spark_session(args, event_log_dir, local_dir)
    expected = read_json(args.reference) if args.reference else None
    if expected is not None and "reference" in expected:
        expected = expected["reference"]
    trials: list[dict[str, Any]] = []
    final_plan = ""
    try:
        phases = ["warmup"] * args.warmups + ["measured"] * args.trials
        measured_indices = list(range(args.trials))
        random.Random(args.seed).shuffle(measured_indices)
        measured_cursor = 0
        for sequence, phase in enumerate(phases):
            query = build_query(spark, fact_path, dimension_path, args.topology)
            elapsed, value, telemetry = run_action(
                query, args.telemetry_interval_ms if phase == "measured" else 0
            )
            compare_values(expected, value) if expected is not None else None
            final_plan = query._jdf.queryExecution().executedPlan().toString()
            summary = plan_summary(final_plan)
            if args.mode == "gpu" and not summary["has_gpu"]:
                raise AssertionError("GPU mode produced no Gpu operators in the executed plan")
            if args.mode == "cpu" and summary["has_gpu"]:
                raise AssertionError("CPU mode unexpectedly produced Gpu operators")
            if args.topology != "native" and not summary["has_python"]:
                raise AssertionError("UDF topology produced no Python operator in the executed plan")
            trial_index = sequence if phase == "warmup" else measured_indices[measured_cursor]
            if phase == "measured":
                measured_cursor += 1
            trials.append(
                {
                    "correct": expected is None or True,
                    "elapsed_seconds": elapsed,
                    "phase": phase,
                    "plan": summary,
                    "result_digest": result_digest(value),
                    "sequence": sequence,
                    "telemetry": telemetry,
                    "trial_index": trial_index,
                }
            )
        if expected is None:
            expected = value
        result = {
            "benchmark": {
                "mode": args.mode,
                "python_bridge": args.python_bridge,
                "replication": args.replication,
                "seed": args.seed,
                "shuffle_partitions": args.shuffle_partitions,
                "telemetry_interval_ms": args.telemetry_interval_ms,
                "threads": args.threads,
                "topology": args.topology,
                "trials": args.trials,
                "warmups": args.warmups,
            },
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset": manifest,
            "environment": environment(spark),
            "executed_plan": final_plan,
            "experiment": EXPERIMENT_NAME,
            "git_commit": source_id(),
            "reference": expected,
            "reference_digest": result_digest(expected),
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "spark_conf": dict(spark.sparkContext.getConf().getAll()),
            "trials": trials,
        }
    finally:
        try:
            spark.stop()
        except Exception:
            # Preserve the original benchmark failure when the JVM has already died.
            pass
    result_path = output_dir / "result.json"
    atomic_new_json(result_path, result)
    print(json.dumps({"reference": result["reference"], "result": str(result_path), "run_id": run_id}, indent=2))


if __name__ == "__main__":
    main()
