"""Resumable fresh-process matrix orchestration for spatial analytics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import atomic_json


def expand(config: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    defaults = config.get("defaults", {})
    for sweep in config["sweeps"]:
        axis = sweep["axis"]
        for value in sweep["values"]:
            case = dict(defaults)
            case.update(sweep.get("base", {}))
            case["operation"] = sweep["operation"]
            case[axis] = value
            case["sweep"] = sweep["name"]
            case["axis"] = axis
            case["axis_value"] = value
            cases.append(case)
    return cases


def case_id(case: dict[str, Any]) -> str:
    identity = json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(identity).hexdigest()[:10]
    value = str(case["axis_value"]).replace(".", "p")
    return f"{case['sweep']}-{value}-{digest}"


def matrix_id(config: dict[str, Any], source_commit: str) -> str:
    payload = {"config": config, "source_commit": source_commit}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def valid_result(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("schema_version") == 1 and payload.get("correctness", {}).get("matches_equal") is True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/data/benchmarks/spatial-analytics/matrices"))
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    source_commit = os.environ.get("GPU_LAB_SOURCE_COMMIT", "unknown")
    identity = matrix_id(config, source_commit)
    root = args.output_root / identity
    cases = expand(config)
    jobs = [
        (case, replication)
        for case in cases
        for replication in range(1, int(config["replications"]) + 1)
    ]
    random.Random(int(config.get("random_seed", 20260813))).shuffle(jobs)
    manifest = {
        "schema_version": 1,
        "matrix_id": identity,
        "source_commit": source_commit,
        "config": config,
        "cases": len(cases),
        "jobs": len(jobs),
    }
    atomic_json(root / "manifest.json", manifest)

    failures: list[dict[str, Any]] = []
    for position, (case, replication) in enumerate(jobs, 1):
        output = root / case_id(case) / f"replication-{replication}.json"
        if valid_result(output):
            print(f"[{position}/{len(jobs)}] resume {output.parent.name} rep={replication}", flush=True)
            continue
        failure_path = output.with_suffix(".failed.json")
        if failure_path.exists() and not args.retry_failed:
            failures.append(json.loads(failure_path.read_text()))
            print(f"[{position}/{len(jobs)}] known failure {output.parent.name} rep={replication}", flush=True)
            continue
        print(f"[{position}/{len(jobs)}] run {output.parent.name} rep={replication}", flush=True)
        command = [
            sys.executable,
            "-m",
            "experiments.spatial_analytics.benchmark",
            "--case",
            json.dumps(case, sort_keys=True),
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            failure = {
                "case": case,
                "replication": replication,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-20_000:],
                "stderr": completed.stderr[-20_000:],
            }
            atomic_json(failure_path, failure)
            failures.append(failure)
            print(completed.stderr, file=sys.stderr, flush=True)
        elif not valid_result(output):
            raise RuntimeError(f"benchmark exited successfully without a valid result: {output}")

    state = dict(manifest)
    state.update({"failures": len(failures), "complete": not failures})
    atomic_json(root / "state.json", state)
    if failures:
        raise SystemExit(f"{len(failures)} matrix jobs failed; inspect {root}")


if __name__ == "__main__":
    main()
