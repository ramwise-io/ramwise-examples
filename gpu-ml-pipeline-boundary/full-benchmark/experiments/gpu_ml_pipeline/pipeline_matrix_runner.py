"""Run the resumable end-to-end pipeline residency matrix."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from experiments.gpu_ml_pipeline.common import (
    atomic_json,
    read_json,
    source_id as discover_source_id,
)
from experiments.gpu_ml_pipeline.matrix_runner import matrix_identity, run_json
from experiments.gpu_ml_pipeline.pipeline_benchmark import MODES

GENERATED_ROOT = Path("/data/generated/gpu-ml-pipeline")
BENCHMARK_ROOT = Path("/data/benchmarks/gpu-ml-pipeline/pipeline-matrices")


def source_id() -> str:
    return discover_source_id()


def validate_pipeline_quality(state: dict[str, Any], require_accel_gpu_log: bool) -> None:
    by_key = {
        (record["dataset"], record["mode"], record["replication"]): record
        for record in state["runs"].values()
        if record["status"] == "complete"
    }
    for (dataset, mode, replication), record in by_key.items():
        cpu = by_key.get((dataset, "cpu", replication))
        if cpu is None:
            raise RuntimeError(f"Missing CPU pipeline reference for {(dataset, mode, replication)}")
        cpu_result = read_json(Path(cpu["result"]))
        result = read_json(Path(record["result"]))
        if float(result["quality"]["roc_auc"]) < float(cpu_result["quality"]["roc_auc"]) - 0.02:
            raise RuntimeError(
                f"Pipeline quality failed for {(dataset, mode, replication)}: "
                f"CPU={cpu_result['quality']} candidate={result['quality']}"
            )
        if mode == "accel":
            logs = Path(record["stdout_log"]).read_text(encoding="utf-8") + "\n" + Path(
                record["stderr_log"]
            ).read_text(encoding="utf-8")
            lowered = logs.lower()
            if "falling back to cpu" in lowered:
                raise RuntimeError(f"Unexpected cuml.accel fallback in pipeline {dataset}")
            if require_accel_gpu_log and "ran on gpu" not in lowered:
                raise RuntimeError(f"No cuml.accel GPU dispatch confirmation in pipeline {dataset}")
        record["quality_status"] = "pass"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported matrix schema")
    commit = source_id()
    matrix_id, digest = matrix_identity(config, commit)
    matrix_dir = BENCHMARK_ROOT / matrix_id
    state_path = matrix_dir / "state.json"
    if state_path.exists():
        state = read_json(state_path)
        if state["identity_sha256"] != digest:
            raise RuntimeError("Matrix identity mismatch")
    else:
        state = {
            "schema_version": 1,
            "matrix_id": matrix_id,
            "identity_sha256": digest,
            "git_commit": commit,
            "config": config,
            "status": "pending",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "datasets": {},
            "runs": {},
        }
        atomic_json(state_path, state)
    state["status"] = "running"
    atomic_json(state_path, state)
    env = dict(os.environ, GPU_LAB_SOURCE_ID=commit, CUML_ACCEL_LOG_LEVEL="info")
    generation = config["generation"]
    for dataset in config["datasets"]:
        name = dataset["name"]
        if name in state["datasets"]:
            continue
        outcome, _ = run_json(
            [
                sys.executable,
                "-m",
                "experiments.gpu_ml_pipeline.generate_dataset",
                "--rows",
                str(dataset["rows"]),
                "--features",
                str(dataset["features"]),
                "--test-fraction",
                str(dataset.get("test_fraction", generation["test_fraction"])),
                "--clusters",
                str(dataset.get("clusters", generation["clusters"])),
                "--informative-features",
                str(dataset.get("informative_features", generation["informative_features"])),
                "--cluster-scale",
                str(dataset.get("cluster_scale", generation["cluster_scale"])),
                "--seed",
                str(dataset.get("seed", generation["seed"])),
                "--chunk-rows",
                str(dataset.get("chunk_rows", generation["chunk_rows"])),
                "--output-root",
                str(GENERATED_ROOT),
            ],
            env,
            matrix_dir / "logs" / f"generate-{name}.stdout.log",
            matrix_dir / "logs" / f"generate-{name}.stderr.log",
        )
        state["datasets"][name] = {**dataset, "manifest": outcome["manifest"], "dataset_id": outcome["dataset_id"]}
        atomic_json(state_path, state)

    benchmark = config["benchmark"]
    jobs = [
        (dataset["name"], mode, replication)
        for dataset in config["datasets"]
        for mode in dataset.get("modes", MODES)
        for replication in range(int(benchmark["replications"]))
    ]
    random.Random(int(benchmark["seed"])).shuffle(jobs)
    for dataset_name, mode, replication in jobs:
        case_id = f"{dataset_name}-{mode}-rep{replication}"
        record = state["runs"].setdefault(
            case_id,
            {"dataset": dataset_name, "mode": mode, "replication": replication, "attempts": 0, "status": "pending"},
        )
        if record["status"] == "complete" or (record["status"] == "failed" and not args.retry_failed):
            continue
        record.update({"attempts": record["attempts"] + 1, "status": "running"})
        atomic_json(state_path, state)
        attempt = int(record["attempts"])
        stdout_path = matrix_dir / "logs" / f"{case_id}-attempt{attempt}.stdout.log"
        stderr_path = matrix_dir / "logs" / f"{case_id}-attempt{attempt}.stderr.log"
        try:
            outcome, _ = run_json(
                [
                    sys.executable,
                    "-m",
                    "experiments.gpu_ml_pipeline.pipeline_benchmark",
                    "--manifest",
                    state["datasets"][dataset_name]["manifest"],
                    "--mode",
                    mode,
                    "--warmups",
                    str(benchmark["warmups"]),
                    "--trials",
                    str(benchmark["trials"]),
                    "--replication",
                    str(replication),
                    "--seed",
                    str(benchmark["seed"]),
                    "--output-root",
                    str(matrix_dir / "raw" / f"attempt-{attempt}"),
                ],
                env,
                stdout_path,
                stderr_path,
            )
            record.update(
                {
                    "status": "complete",
                    "result": outcome["result"],
                    "run_id": outcome["run_id"],
                    "stdout_log": str(stdout_path),
                    "stderr_log": str(stderr_path),
                }
            )
        except Exception as error:
            record.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "stdout_log": str(stdout_path),
                    "stderr_log": str(stderr_path),
                }
            )
            state["status"] = "failed"
            atomic_json(state_path, state)
            raise
        atomic_json(state_path, state)

    validate_pipeline_quality(
        state, bool(benchmark.get("require_accel_gpu_log", True))
    )
    atomic_json(state_path, state)
    outcome, _ = run_json(
        [
            sys.executable,
            "-m",
            "experiments.gpu_ml_pipeline.pipeline_summarize",
            "--state",
            str(state_path),
            "--output-dir",
            str(matrix_dir / "report"),
        ],
        env,
        matrix_dir / "logs" / "summarize.stdout.log",
        matrix_dir / "logs" / "summarize.stderr.log",
    )
    state.update(
        {
            "status": "complete",
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "report": outcome,
        }
    )
    atomic_json(state_path, state)
    print(json.dumps({"matrix_id": matrix_id, "state": str(state_path), "report": outcome}, indent=2))


if __name__ == "__main__":
    main()
