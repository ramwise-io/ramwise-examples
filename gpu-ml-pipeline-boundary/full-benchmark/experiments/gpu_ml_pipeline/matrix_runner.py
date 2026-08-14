"""Run the resumable fresh-process GPU ML benchmark matrix."""

from __future__ import annotations

import argparse
import hashlib
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
    canonical_json,
    quality_matches,
    read_json,
    source_id as discover_source_id,
)
from experiments.gpu_ml_pipeline.estimators import ALGORITHMS, ENGINES, allowed_engines

GENERATED_ROOT = Path("/data/generated/gpu-ml-pipeline")
BENCHMARK_ROOT = Path("/data/benchmarks/gpu-ml-pipeline/matrices")


def source_id() -> str:
    return discover_source_id()


def matrix_identity(config: dict[str, Any], commit: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        canonical_json({"config": config, "git_commit": commit}).encode()
    ).hexdigest()
    return f"{config['name']}-{digest[:10]}", digest


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported matrix schema")
    names = [dataset["name"] for dataset in config["datasets"]]
    if len(names) != len(set(names)):
        raise ValueError("Dataset names must be unique")
    for dataset in config["datasets"]:
        if int(dataset["rows"]) < 100 or int(dataset["features"]) < 2:
            raise ValueError("Every dataset needs at least 100 rows and 2 features")
        algorithms = dataset.get("algorithms", ALGORITHMS)
        engines = dataset.get("engines", ENGINES)
        unknown_algorithms = set(algorithms) - set(ALGORITHMS)
        unknown_engines = set(engines) - set(ENGINES)
        if unknown_algorithms or unknown_engines:
            raise ValueError(
                f"Unknown algorithms {unknown_algorithms} or engines {unknown_engines}"
            )


def expand_jobs(config: dict[str, Any]) -> list[tuple[str, str, str, int]]:
    jobs = []
    replications = int(config["benchmark"]["replications"])
    for dataset in config["datasets"]:
        for algorithm in dataset.get("algorithms", ALGORITHMS):
            requested_engines = dataset.get("engines", ENGINES)
            for engine in requested_engines:
                if engine not in allowed_engines(algorithm):
                    continue
                for replication in range(replications):
                    jobs.append((dataset["name"], algorithm, engine, replication))
    return jobs


def run_json(
    command: list[str],
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[dict[str, Any], str]:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, text=True, capture_output=True, env=env)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="", flush=True)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout[-4000:]}\n{completed.stderr[-8000:]}"
        )
    print(completed.stdout, end="", flush=True)
    decoder = json.JSONDecoder()
    for start, character in reversed(list(enumerate(completed.stdout))):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(completed.stdout[start:])
        except json.JSONDecodeError:
            continue
        if not completed.stdout[start + end :].strip() and isinstance(value, dict):
            return value, completed.stdout + "\n" + completed.stderr
    raise RuntimeError("Command output did not end with JSON")


def validate_results(state: dict[str, Any], require_accel_gpu_log: bool) -> None:
    by_key = {
        (record["dataset"], record["algorithm"], record["engine"], record["replication"]): record
        for record in state["runs"].values()
        if record["status"] == "complete"
    }
    for key, record in by_key.items():
        dataset, algorithm, engine, replication = key
        cpu = by_key.get((dataset, algorithm, "cpu", replication))
        if cpu is None:
            raise RuntimeError(f"Missing CPU quality reference for {key}")
        cpu_result = read_json(Path(cpu["result"]))
        result = read_json(Path(record["result"]))
        if not quality_matches(algorithm, cpu_result["quality"], result["quality"]):
            raise RuntimeError(
                f"Quality control failed for {key}: CPU={cpu_result['quality']} "
                f"candidate={result['quality']}"
            )
        if engine == "accel":
            logs = Path(record["stdout_log"]).read_text(encoding="utf-8") + "\n" + Path(
                record["stderr_log"]
            ).read_text(encoding="utf-8")
            lowered = logs.lower()
            if "falling back to cpu" in lowered:
                raise RuntimeError(f"cuml.accel CPU fallback detected for {key}")
            if require_accel_gpu_log and "ran on gpu" not in lowered:
                raise RuntimeError(f"No cuml.accel GPU dispatch confirmation for {key}")
            if not result["dispatch"]["cuml_accel_proxy"]:
                raise RuntimeError(f"cuml.accel did not install a proxy for {key}")
        record["quality_status"] = "pass"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    validate_config(config)
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
        stdout_path = matrix_dir / "logs" / f"generate-{name}.stdout.log"
        stderr_path = matrix_dir / "logs" / f"generate-{name}.stderr.log"
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
            stdout_path,
            stderr_path,
        )
        state["datasets"][name] = {
            **dataset,
            "dataset_id": outcome["dataset_id"],
            "manifest": outcome["manifest"],
        }
        atomic_json(state_path, state)

    jobs = expand_jobs(config)
    random.Random(int(config["benchmark"]["seed"])).shuffle(jobs)
    benchmark = config["benchmark"]
    for dataset_name, algorithm, engine, replication in jobs:
        case_id = f"{dataset_name}-{algorithm}-{engine}-rep{replication}"
        record = state["runs"].setdefault(
            case_id,
            {
                "dataset": dataset_name,
                "algorithm": algorithm,
                "engine": engine,
                "replication": replication,
                "attempts": 0,
                "status": "pending",
            },
        )
        if record["status"] == "complete" or (
            record["status"] == "failed" and not args.retry_failed
        ):
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
                    "experiments.gpu_ml_pipeline.benchmark",
                    "--manifest",
                    state["datasets"][dataset_name]["manifest"],
                    "--algorithm",
                    algorithm,
                    "--engine",
                    engine,
                    "--warmups",
                    str(benchmark["warmups"]),
                    "--trials",
                    str(benchmark["trials"]),
                    "--replication",
                    str(replication),
                    "--inference-batches",
                    ",".join(map(str, benchmark["inference_batches"])),
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

    validate_results(
        state,
        bool(config["benchmark"].get("require_accel_gpu_log", True)),
    )
    atomic_json(state_path, state)
    report_stdout = matrix_dir / "logs" / "summarize.stdout.log"
    report_stderr = matrix_dir / "logs" / "summarize.stderr.log"
    outcome, _ = run_json(
        [
            sys.executable,
            "-m",
            "experiments.gpu_ml_pipeline.summarize",
            "--state",
            str(state_path),
            "--output-dir",
            str(matrix_dir / "report"),
        ],
        env,
        report_stdout,
        report_stderr,
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
