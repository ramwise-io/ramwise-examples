"""Run a resumable profile/codec/projection Parquet benchmark matrix."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import psutil

from experiments.parquet_decompression.common import canonical_json, dataset_id, read_json
from experiments.parquet_decompression.matrix_report import build_report
from experiments.parquet_decompression.telemetry import gpu_snapshot

DATA_ROOT = Path(os.environ.get("PARQUET_BENCH_DATA_ROOT", "benchmark-data"))
GENERATED_ROOT = DATA_ROOT / "generated/parquet-decompression"
BENCHMARK_ROOT = DATA_ROOT / "benchmarks/parquet-decompression"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported matrix schema")
    for field in ("name", "profiles", "codecs", "projections", "generation", "benchmark"):
        if not config.get(field):
            raise ValueError(f"Missing matrix config field: {field}")
    if int(config.get("replications", 0)) < 1:
        raise ValueError("replications must be positive")


def parse_cpu_set(specification: str) -> list[int]:
    cpus: set[int] = set()
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"Invalid CPU range: {item}")
            cpus.update(range(start, end + 1))
        else:
            cpu = int(item)
            if cpu < 0:
                raise ValueError(f"Invalid CPU: {item}")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU affinity must not be empty")
    return sorted(cpus)


def source_git_commit() -> str:
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


def matrix_identity(
    config: dict[str, Any], git_commit: str | None = None
) -> tuple[str, str]:
    identity = {
        "config": config,
        "git_commit": git_commit if git_commit is not None else source_git_commit(),
    }
    digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    return f"{config['name']}-{digest[:10]}", digest


def enumerate_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    row_group_sizes = config.get("row_group_rows")
    if row_group_sizes is None:
        row_group_sizes = [config.get("generation", {}).get("row_group_rows", 0)]
    for index, (profile, codec, row_group_rows, projection, replication) in enumerate(
        itertools.product(
            config["profiles"],
            config["codecs"],
            row_group_sizes,
            config["projections"],
            range(1, int(config["replications"]) + 1),
        )
    ):
        row_group_label = (
            f"-rg{row_group_rows}" if len(row_group_sizes) > 1 else ""
        )
        condition_id = f"{profile}-{codec}{row_group_label}-{projection}"
        case_id = (
            condition_id
            if int(config["replications"]) == 1
            else f"{condition_id}-rep{replication}"
        )
        cases.append(
            {
                "case_id": case_id,
                "codec": codec,
                "condition_id": condition_id,
                "index": index,
                "profile": profile,
                "projection": projection,
                "replication": replication,
                "row_group_rows": int(row_group_rows),
                "seed": int(config["benchmark"]["seed"]) + index,
            }
        )
    return cases


def _atomic_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _check_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".preflight-", dir=path, delete=True):
        pass


def preflight(config: dict[str, Any], matrix_dir: Path) -> dict[str, Any]:
    _check_writable(GENERATED_ROOT)
    _check_writable(matrix_dir)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(DATA_ROOT)
    memory = psutil.virtual_memory()
    gpu = gpu_snapshot()
    if "error" in gpu:
        raise RuntimeError(f"GPU preflight failed: {gpu['error']}")

    gib = 1024**3
    thresholds = config["preflight"]
    benchmark = config["benchmark"]
    actual_affinity = sorted(psutil.Process().cpu_affinity())
    expected_specification = benchmark.get("cpu_affinity")
    expected_affinity = (
        parse_cpu_set(expected_specification) if expected_specification else actual_affinity
    )
    if actual_affinity != expected_affinity:
        raise RuntimeError(
            f"CPU affinity mismatch: expected {expected_affinity}, got {actual_affinity}"
        )
    if len(actual_affinity) < int(benchmark["threads"]):
        raise RuntimeError(
            f"CPU affinity exposes {len(actual_affinity)} CPUs for "
            f"{benchmark['threads']} requested threads"
        )
    checks = {
        "available_memory_gib": memory.available / gib,
        "cpu_affinity": actual_affinity,
        "cpu_affinity_specification": expected_specification,
        "free_disk_gib": disk.free / gib,
        "gpu": gpu,
    }
    if checks["available_memory_gib"] < thresholds["min_available_memory_gib"]:
        raise RuntimeError(f"Insufficient available memory: {checks['available_memory_gib']:.1f} GiB")
    if checks["free_disk_gib"] < thresholds["min_free_disk_gib"]:
        raise RuntimeError(f"Insufficient free disk: {checks['free_disk_gib']:.1f} GiB")
    if gpu["memory_total_bytes"] / gib < thresholds["min_gpu_memory_gib"]:
        raise RuntimeError("GPU memory is below the configured minimum")
    max_used = thresholds.get("max_gpu_memory_used_gib", 2)
    if gpu["memory_used_bytes"] / gib > max_used:
        raise RuntimeError(
            f"GPU already has {gpu['memory_used_bytes'] / gib:.1f} GiB allocated; "
            "another workload may be active"
        )
    return checks


def _run(command: list[str]) -> dict[str, Any]:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit {completed.returncode}: {' '.join(command)}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    print(completed.stdout, end="", flush=True)
    return json.loads(completed.stdout)


def dataset_manifest(config: dict[str, Any], case: dict[str, Any]) -> Path:
    generation = {
        **config["generation"],
        "codec": case["codec"],
        "profile": case["profile"],
        "row_group_rows": case["row_group_rows"],
    }
    identifier = dataset_id(generation)
    manifest = GENERATED_ROOT / identifier / "manifest.json"
    if manifest.exists():
        return manifest
    command = [
        sys.executable,
        "-m",
        "experiments.parquet_decompression.generate_dataset",
        "--output-root",
        str(GENERATED_ROOT),
    ]
    for key in ("rows", "rows_per_file", "row_group_rows", "seed"):
        command.extend([f"--{key.replace('_', '-')}", str(generation[key])])
    command.extend(["--profile", case["profile"], "--codec", case["codec"]])
    _run(command)
    if not manifest.exists():
        raise RuntimeError(f"Generator did not create expected manifest: {manifest}")
    return manifest


def benchmark_case(
    config: dict[str, Any],
    matrix_id: str,
    matrix_dir: Path,
    case: dict[str, Any],
    manifest: Path,
) -> dict[str, Any]:
    benchmark = config["benchmark"]
    command = [
        sys.executable,
        "-m",
        "experiments.parquet_decompression.benchmark",
        "--manifest",
        str(manifest),
        "--output-root",
        str(matrix_dir / "raw"),
        "--engines",
        ",".join(benchmark["engines"]),
        "--projection",
        case["projection"],
        "--warmups",
        str(benchmark["warmups"]),
        "--trials",
        str(benchmark["trials"]),
        "--threads",
        str(benchmark["threads"]),
        "--seed",
        str(case["seed"]),
        "--telemetry-interval-ms",
        str(benchmark["telemetry_interval_ms"]),
        "--host-label",
        os.environ.get("PARQUET_BENCH_HOSTNAME", "benchmark-host"),
        "--matrix-id",
        matrix_id,
        "--case-id",
        case["case_id"],
    ]
    return _run(command)


def _initial_state(
    config: dict[str, Any],
    matrix_id: str,
    digest: str,
    git_commit: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "cases": {
            case["case_id"]: {**case, "attempts": 0, "status": "pending"}
            for case in cases
        },
        "config": config,
        "identity_sha256": digest,
        "created_utc": now,
        "git_commit": git_commit,
        "matrix_id": matrix_id,
        "schema_version": 1,
        "status": "pending",
        "updated_utc": now,
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    validate_config(config)
    git_commit = source_git_commit()
    matrix_id, digest = matrix_identity(config, git_commit)
    cases = enumerate_cases(config)
    matrix_dir = BENCHMARK_ROOT / "matrices" / matrix_id
    state_path = matrix_dir / "state.json"

    if state_path.exists():
        state = read_json(state_path)
        if state.get("identity_sha256") != digest:
            raise RuntimeError("Existing matrix state does not match the source identity")
    else:
        state = _initial_state(config, matrix_id, digest, git_commit, cases)
        _atomic_state(state_path, state)

    for case in state["cases"].values():
        if case["status"] == "running":
            case["status"] = "pending"
            case["interrupted"] = True
        if case["status"] == "complete" and not Path(case["result"]).exists():
            case["status"] = "pending"
            case["missing_result"] = True
    state["preflight"] = preflight(config, matrix_dir)
    state["status"] = "running"
    _atomic_state(state_path, state)

    for case_id in [case["case_id"] for case in cases]:
        case = state["cases"][case_id]
        if case["status"] == "complete":
            print(f"SKIP complete {case_id}", flush=True)
            continue
        if case["status"] == "failed" and not args.retry_failed:
            print(f"SKIP failed {case_id}; use --retry-failed", flush=True)
            continue
        case.update(
            {
                "attempts": int(case.get("attempts", 0)) + 1,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "running",
            }
        )
        case.pop("error", None)
        _atomic_state(state_path, state)
        try:
            manifest = dataset_manifest(config, case)
            outcome = benchmark_case(config, matrix_id, matrix_dir, case, manifest)
            case.update(
                {
                    "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "manifest": str(manifest),
                    "medians_mib_s": outcome["medians_mib_s"],
                    "result": outcome["result"],
                    "run_id": outcome["run_id"],
                    "status": "complete",
                }
            )
        except Exception as error:
            case.update(
                {
                    "error": f"{type(error).__name__}: {error}",
                    "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "status": "failed",
                    "traceback": traceback.format_exc(),
                }
            )
            state["status"] = "failed"
            _atomic_state(state_path, state)
            if config.get("stop_on_failure", True):
                raise
        _atomic_state(state_path, state)

    failed = [case for case in state["cases"].values() if case["status"] == "failed"]
    pending = [case for case in state["cases"].values() if case["status"] != "complete"]
    if failed or pending:
        state["status"] = "failed" if failed else "incomplete"
        _atomic_state(state_path, state)
        raise RuntimeError(f"Matrix incomplete: {len(failed)} failed, {len(pending)} not complete")

    state["report"] = build_report(state_path, matrix_dir / "report")
    state["status"] = "complete"
    state["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_state(state_path, state)
    print(json.dumps({"matrix_id": matrix_id, "report": state["report"], "state": str(state_path)}, indent=2))


if __name__ == "__main__":
    main()
