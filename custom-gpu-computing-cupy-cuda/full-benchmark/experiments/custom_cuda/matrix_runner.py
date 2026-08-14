"""Run a resumable fresh-process custom CUDA benchmark matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from experiments.custom_cuda.common import atomic_json, canonical_json, read_json, source_id


BENCHMARK_ROOT = Path("/data/benchmarks/custom-cuda/matrices")
CACHE_ROOT = Path("/data/cache/custom-cuda")


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported matrix schema")
    benchmark = config.get("benchmark", {})
    for key in ("warmups", "trials", "replications", "seed"):
        if int(benchmark.get(key, 0)) < 1:
            raise ValueError(f"benchmark.{key} must be positive")
    if not any(config.get(name) for name in ("launch", "transfer", "fusion")):
        raise ValueError("Matrix must contain at least one benchmark section")
    allowed = {
        "cupy_composed",
        "cupy_partial_fusion",
        "raw_aos",
        "raw_soa",
        "raw_strided",
    }
    for case in config.get("fusion", []):
        if int(case["rows"]) < 1 or int(case["features"]) < 1:
            raise ValueError("Fusion rows and features must be positive")
        blocks = [int(value) for value in case.get("blocks", [256])]
        if any(value < 32 or value > 1024 or value % 32 for value in blocks):
            raise ValueError("Block sizes must be warp-aligned values from 32 to 1024")
        unknown = set(case.get("variants", allowed)) - allowed
        if unknown:
            raise ValueError(f"Unknown fusion variants: {sorted(unknown)}")


def matrix_identity(config: dict[str, Any], commit: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        canonical_json({"config": config, "git_commit": commit}).encode()
    ).hexdigest()
    return f"{config['name']}-{digest[:10]}", digest


def expand_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    replications = int(config["benchmark"]["replications"])
    for launches in config.get("launch", []):
        for replication in range(replications):
            jobs.append(
                {"mode": "launch", "launches": int(launches), "replication": replication}
            )
    transfer = config.get("transfer", {})
    for size in transfer.get("bytes", []):
        for direction in transfer.get("directions", ("h2d", "d2h")):
            for memory in transfer.get("memory", ("pageable", "pinned")):
                for replication in range(replications):
                    jobs.append(
                        {
                            "mode": "transfer",
                            "bytes": int(size),
                            "direction": direction,
                            "memory": memory,
                            "replication": replication,
                        }
                    )
    for case in config.get("fusion", []):
        for replication in range(replications):
            jobs.append(
                {
                    "mode": "fusion",
                    "rows": int(case["rows"]),
                    "features": int(case["features"]),
                    "blocks": [int(value) for value in case.get("blocks", [256])],
                    "variants": list(case.get("variants", [
                        "cupy_composed",
                        "cupy_partial_fusion",
                        "raw_aos",
                        "raw_soa",
                        "raw_strided",
                    ])),
                    "replication": replication,
                }
            )
    return jobs


def job_id(job: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(job).encode()).hexdigest()[:10]
    mode = job["mode"]
    if mode == "launch":
        label = f"launch-{job['launches']}"
    elif mode == "transfer":
        label = f"transfer-{job['bytes']}-{job['direction']}-{job['memory']}"
    else:
        label = f"fusion-r{job['rows']}-f{job['features']}"
    return f"{label}-rep{job['replication']}-{digest}"


def run_job(
    job: dict[str, Any],
    benchmark: dict[str, Any],
    result_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "experiments.custom_cuda.benchmark",
        "--mode",
        job["mode"],
        "--warmups",
        str(benchmark["warmups"]),
        "--trials",
        str(benchmark["trials"]),
        "--replication",
        str(job["replication"]),
        "--seed",
        str(benchmark["seed"]),
        "--output",
        str(result_path),
    ]
    if job["mode"] == "launch":
        command += ["--launches", str(job["launches"])]
    elif job["mode"] == "transfer":
        command += [
            "--bytes",
            str(job["bytes"]),
            "--direction",
            job["direction"],
            "--memory",
            job["memory"],
        ]
    else:
        command += [
            "--rows",
            str(job["rows"]),
            "--features",
            str(job["features"]),
            "--blocks",
            ",".join(map(str, job["blocks"])),
            "--variants",
            ",".join(job["variants"]),
        ]
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, text=True, capture_output=True, env=env)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="", flush=True)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout[-4000:]}\n{completed.stderr[-8000:]}"
        )
    if not result_path.exists():
        raise RuntimeError(f"Benchmark did not create {result_path}")
    return read_json(result_path)


def validate_result(result: dict[str, Any]) -> None:
    if result["mode"] == "fusion":
        references = [r for r in result["records"] if r["implementation"] == "numpy"]
        if len(references) != 1:
            raise RuntimeError("Fusion result requires exactly one NumPy reference")
        for record in result["records"]:
            status = record["quality"]["status"]
            if status not in {"reference", "pass"}:
                raise RuntimeError(f"Fusion quality failure: {record}")
    elif result["mode"] == "transfer":
        if result["records"][0]["quality_status"] != "pass":
            raise RuntimeError("Transfer quality failure")


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
            "runs": {},
        }
        atomic_json(state_path, state)
    state["status"] = "running"
    atomic_json(state_path, state)
    jobs = expand_jobs(config)
    random.Random(int(config["benchmark"]["seed"])).shuffle(jobs)
    for job in jobs:
        case = job_id(job)
        record = state["runs"].setdefault(
            case,
            {"job": job, "attempts": 0, "status": "pending"},
        )
        if record["status"] == "complete" or (
            record["status"] == "failed" and not args.retry_failed
        ):
            continue
        attempt = int(record["attempts"]) + 1
        record.update({"attempts": attempt, "status": "running"})
        atomic_json(state_path, state)
        result_path = matrix_dir / "raw" / case / f"attempt-{attempt}.json"
        stdout_path = matrix_dir / "logs" / f"{case}-attempt{attempt}.stdout.log"
        stderr_path = matrix_dir / "logs" / f"{case}-attempt{attempt}.stderr.log"
        cache_dir = CACHE_ROOT / matrix_id / case / f"attempt-{attempt}"
        env = dict(
            os.environ,
            GPU_LAB_SOURCE_ID=commit,
            CUPY_CACHE_DIR=str(cache_dir),
        )
        try:
            result = run_job(
                job,
                config["benchmark"],
                result_path,
                stdout_path,
                stderr_path,
                env,
            )
            validate_result(result)
            record.update(
                {
                    "status": "complete",
                    "result": str(result_path),
                    "run_id": result["run_id"],
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
    report_stdout = matrix_dir / "logs" / "summarize.stdout.log"
    report_stderr = matrix_dir / "logs" / "summarize.stderr.log"
    command = [
        sys.executable,
        "-m",
        "experiments.custom_cuda.summarize",
        "--state",
        str(state_path),
        "--output-dir",
        str(matrix_dir / "report"),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, env=os.environ)
    report_stdout.write_text(completed.stdout, encoding="utf-8")
    report_stderr.write_text(completed.stderr, encoding="utf-8")
    print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="", flush=True)
    if completed.returncode:
        raise RuntimeError(f"Summarizer failed: {completed.stderr[-8000:]}")
    report = json.loads(completed.stdout)
    state.update(
        {
            "status": "complete",
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "report": report,
        }
    )
    atomic_json(state_path, state)
    print(json.dumps({"matrix_id": matrix_id, "state": str(state_path), "report": report}, indent=2))


if __name__ == "__main__":
    main()

