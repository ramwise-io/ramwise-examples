"""Run a resumable analytical-pipeline scale matrix."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from experiments.analytical_pipeline.common import canonical_json, dataset_id, read_json

GENERATED_ROOT = Path("/data/generated/analytical-pipeline")
BENCHMARK_ROOT = Path("/data/benchmarks/analytical-pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--retry-failed", action="store_true")
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


def matrix_identity(config: dict[str, Any], commit: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        canonical_json({"config": config, "git_commit": commit}).encode()
    ).hexdigest()
    return f"{config['name']}-{digest[:10]}", digest


def enumerate_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for index, (profile, codec, rows) in enumerate(
        itertools.product(config["profiles"], config["codecs"], config["rows"])
    ):
        cases.append(
            {
                "case_id": f"{profile}-{codec}-r{rows}",
                "codec": codec,
                "index": index,
                "profile": profile,
                "rows": int(rows),
                "seed": int(config["benchmark"]["seed"]) + index,
            }
        )
    return cases


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported matrix schema")
    for field in ("name", "profiles", "codecs", "rows", "generation", "benchmark"):
        if not config.get(field):
            raise ValueError(f"Missing matrix field: {field}")


def atomic_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_json(command: list[str]) -> dict[str, Any]:
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
    decoder = json.JSONDecoder()
    for start, character in reversed(list(enumerate(completed.stdout))):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(completed.stdout[start:])
        except json.JSONDecodeError:
            continue
        if completed.stdout[start + end :].strip() == "" and isinstance(value, dict):
            return value
    raise RuntimeError("Command did not end with a JSON object")


def ensure_dataset(config: dict[str, Any], case: dict[str, Any]) -> Path:
    generation = {
        **config["generation"],
        "codec": case["codec"],
        "profile": case["profile"],
        "rows": case["rows"],
    }
    generation["rows_per_file"] = min(
        int(generation["rows_per_file"]), int(generation["rows"])
    )
    manifest = GENERATED_ROOT / dataset_id(generation) / "manifest.json"
    if manifest.exists():
        return manifest
    command = [
        sys.executable,
        "-m",
        "experiments.analytical_pipeline.generate_dataset",
        "--output-root",
        str(GENERATED_ROOT),
    ]
    for key in ("rows", "rows_per_file", "row_group_rows", "seed"):
        command.extend([f"--{key.replace('_', '-')}", str(generation[key])])
    command.extend(["--profile", case["profile"], "--codec", case["codec"]])
    run_json(command)
    return manifest


def benchmark_case(
    config: dict[str, Any], matrix_dir: Path, case: dict[str, Any], manifest: Path
) -> dict[str, Any]:
    benchmark = config["benchmark"]
    command = [
        sys.executable,
        "-m",
        "experiments.analytical_pipeline.benchmark",
        "--manifest",
        str(manifest),
        "--output-root",
        str(matrix_dir / "raw"),
        "--engines",
        ",".join(benchmark["engines"]),
        "--workloads",
        ",".join(benchmark["workloads"]),
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
    ]
    return run_json(command)


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    validate_config(config)
    commit = source_id()
    matrix_id, digest = matrix_identity(config, commit)
    matrix_dir = BENCHMARK_ROOT / "matrices" / matrix_id
    state_path = matrix_dir / "state.json"
    cases = enumerate_cases(config)
    if state_path.exists():
        state = read_json(state_path)
        if state["identity_sha256"] != digest:
            raise RuntimeError("Existing matrix state identity mismatch")
    else:
        state = {
            "cases": {
                case["case_id"]: {**case, "attempts": 0, "status": "pending"}
                for case in cases
            },
            "config": config,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": commit,
            "identity_sha256": digest,
            "matrix_id": matrix_id,
            "schema_version": 1,
            "status": "pending",
        }
        atomic_state(state_path, state)

    state["status"] = "running"
    atomic_state(state_path, state)
    for case in cases:
        record = state["cases"][case["case_id"]]
        if record["status"] == "complete":
            continue
        if record["status"] == "failed" and not args.retry_failed:
            continue
        record["attempts"] += 1
        record["status"] = "running"
        atomic_state(state_path, state)
        try:
            manifest = ensure_dataset(config, case)
            outcome = benchmark_case(config, matrix_dir, case, manifest)
            record.update(
                {
                    "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "manifest": str(manifest),
                    "result": outcome["result"],
                    "run_id": outcome["run_id"],
                    "status": "complete",
                }
            )
        except Exception as error:
            record.update(
                {
                    "error": f"{type(error).__name__}: {error}",
                    "status": "failed",
                    "traceback": traceback.format_exc(),
                }
            )
            state["status"] = "failed"
            atomic_state(state_path, state)
            raise
        atomic_state(state_path, state)

    incomplete = [item for item in state["cases"].values() if item["status"] != "complete"]
    if incomplete:
        state["status"] = "incomplete"
        atomic_state(state_path, state)
        raise RuntimeError(f"Matrix has {len(incomplete)} incomplete cases")
    summary_path = matrix_dir / "summary.csv"
    command = [
        sys.executable,
        "-m",
        "experiments.analytical_pipeline.summarize",
        *[state["cases"][case["case_id"]]["result"] for case in cases],
        "--output",
        str(summary_path),
    ]
    subprocess.run(command, check=True)
    state.update(
        {
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "complete",
            "summary": str(summary_path),
        }
    )
    atomic_state(state_path, state)
    print(json.dumps({"matrix_id": matrix_id, "state": str(state_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
