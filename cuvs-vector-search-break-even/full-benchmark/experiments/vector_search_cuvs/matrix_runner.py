"""Run the resumable cuVS tune-then-confirm benchmark matrix."""

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

from experiments.vector_search_cuvs.common import atomic_json, canonical_json, read_json
from experiments.vector_search_cuvs.select_params import select

GENERATED_ROOT = Path("/data/generated/vector-search-cuvs")
BENCHMARK_ROOT = Path("/data/benchmarks/vector-search-cuvs/matrices")


def source_id() -> str:
    configured = os.environ.get("GPU_LAB_SOURCE_ID")
    if configured:
        return configured
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def matrix_identity(config: dict[str, Any], commit: str) -> tuple[str, str]:
    digest = hashlib.sha256(canonical_json({"config": config, "git_commit": commit}).encode()).hexdigest()
    return f"{config['name']}-{digest[:10]}", digest


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported matrix schema")
    names = [item["name"] for item in config["datasets"]]
    if len(names) != len(set(names)):
        raise ValueError("Dataset names must be unique")
    if "brute_force" not in config["algorithms"]:
        raise ValueError("brute_force is required as the exact reference")


def run_json(command: list[str], env: dict[str, str]) -> dict[str, Any]:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, text=True, capture_output=True, env=env)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="", flush=True)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout[-4000:]}\n{completed.stderr[-8000:]}")
    print(completed.stdout, end="", flush=True)
    decoder = json.JSONDecoder()
    for start, character in reversed(list(enumerate(completed.stdout))):
        if character == "{":
            try:
                value, end = decoder.raw_decode(completed.stdout[start:])
            except json.JSONDecodeError:
                continue
            if not completed.stdout[start + end:].strip() and isinstance(value, dict):
                return value
    raise RuntimeError("Command output did not end with JSON")


def benchmark_command(config: dict[str, Any], manifest: Path, truth: Path, algorithm: str, output: Path, replication: int, *, tuning: bool, cases: Path | None = None) -> list[str]:
    benchmark = config["benchmark"]
    prefix = "tuning" if tuning else "confirmation"
    command = [
        sys.executable, "-m", "experiments.vector_search_cuvs.benchmark",
        "--manifest", str(manifest), "--truth-manifest", str(truth),
        "--algorithm", algorithm, "--output-root", str(output),
        "--batch-sizes", ",".join(map(str, benchmark["batch_sizes"])),
        "--k", str(benchmark["k"]), "--replication", str(replication),
        "--warmups", str(benchmark[f"{prefix}_warmups"]),
        "--trials", str(benchmark[f"{prefix}_trials"]),
    ]
    if cases is not None:
        command.extend(["--cases-json", str(cases)])
    return command


def unique_confirmation_cases(algorithm: str, batches: list[int], selections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if algorithm == "brute_force":
        return [{"batch_size": batch, "params": {}} for batch in batches]
    unique: dict[tuple[int, str], dict[str, Any]] = {}
    for item in selections:
        key = (int(item["batch_size"]), canonical_json(item["params"]))
        unique[key] = {"batch_size": key[0], "params": item["params"]}
    return list(unique.values())


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
            "schema_version": 1, "matrix_id": matrix_id, "identity_sha256": digest,
            "git_commit": commit, "config": config, "status": "pending",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "datasets": {}, "tuning": {}, "selections": {}, "confirmation": {},
        }
        atomic_json(state_path, state)
    env = dict(os.environ, GPU_LAB_SOURCE_ID=commit)
    state["status"] = "running"
    atomic_json(state_path, state)

    for dataset in config["datasets"]:
        name = dataset["name"]
        if name in state["datasets"]:
            continue
        generation = config["generation"]
        outcome = run_json([
            sys.executable, "-m", "experiments.vector_search_cuvs.generate_dataset",
            "--rows", str(dataset["rows"]), "--dimensions", str(dataset["dimensions"]),
            "--queries", str(generation["queries"]), "--clusters", str(generation["clusters"]),
            "--noise", str(generation["noise"]), "--seed", str(generation["seed"]),
            "--chunk-rows", str(generation["chunk_rows"]), "--output-root", str(GENERATED_ROOT),
        ], env)
        manifest = Path(outcome["manifest"])
        truth = manifest.parent / f"truth-k{config['benchmark']['k']}" / "manifest.json"
        run_json([sys.executable, "-m", "experiments.vector_search_cuvs.generate_truth", "--manifest", str(manifest), "--k", str(config["benchmark"]["k"]), "--output", str(truth)], env)
        state["datasets"][name] = {**dataset, "manifest": str(manifest), "truth": str(truth)}
        atomic_json(state_path, state)

    tuning_cases = [(dataset["name"], algorithm) for dataset in config["datasets"] for algorithm in config["algorithms"]]
    random.Random(config["benchmark"]["seed"]).shuffle(tuning_cases)
    for name, algorithm in tuning_cases:
        case_id = f"{name}-{algorithm}"
        record = state["tuning"].setdefault(case_id, {"dataset": name, "algorithm": algorithm, "attempts": 0, "status": "pending"})
        if record["status"] == "complete" or (record["status"] == "failed" and not args.retry_failed):
            continue
        record.update({"attempts": record["attempts"] + 1, "status": "running"})
        atomic_json(state_path, state)
        try:
            data = state["datasets"][name]
            outcome = run_json(benchmark_command(config, Path(data["manifest"]), Path(data["truth"]), algorithm, matrix_dir / "raw" / "tuning" / f"attempt-{record['attempts']}", -1, tuning=True), env)
            record.update({"status": "complete", "result": outcome["result"], "run_id": outcome["run_id"]})
        except Exception as error:
            record.update({"status": "failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
            state["status"] = "failed"
            atomic_json(state_path, state)
            raise
        atomic_json(state_path, state)

    targets = [float(value) for value in config["benchmark"]["recall_targets"]]
    recall_margin = float(config["benchmark"].get("recall_confirmation_margin", 0.0))
    for dataset in config["datasets"]:
        name = dataset["name"]
        state["selections"].setdefault(name, {})
        for algorithm in config["algorithms"]:
            result = read_json(Path(state["tuning"][f"{name}-{algorithm}"]["result"]))
            state["selections"][name][algorithm] = select(
                result,
                [1.0] if algorithm == "brute_force" else targets,
                0.0 if algorithm == "brute_force" else recall_margin,
            )
    atomic_json(state_path, state)

    confirmation_cases = []
    for dataset in config["datasets"]:
        for algorithm in config["algorithms"]:
            for replication in range(config["benchmark"]["replications"]):
                confirmation_cases.append((dataset["name"], algorithm, replication))
    random.Random(config["benchmark"]["seed"] + 1).shuffle(confirmation_cases)
    for name, algorithm, replication in confirmation_cases:
        case_id = f"{name}-{algorithm}-rep{replication}"
        record = state["confirmation"].setdefault(case_id, {"dataset": name, "algorithm": algorithm, "replication": replication, "attempts": 0, "status": "pending"})
        if record["status"] == "complete" or (record["status"] == "failed" and not args.retry_failed):
            continue
        record.update({"attempts": record["attempts"] + 1, "status": "running"})
        case_path = matrix_dir / "cases" / f"{name}-{algorithm}.json"
        cases = unique_confirmation_cases(algorithm, config["benchmark"]["batch_sizes"], state["selections"][name][algorithm])
        if not cases:
            record.update({"status": "skipped", "reason": "No tuning parameter met any recall target"})
            atomic_json(state_path, state)
            continue
        atomic_json(case_path, {"cases": cases})
        atomic_json(state_path, state)
        try:
            data = state["datasets"][name]
            outcome = run_json(benchmark_command(config, Path(data["manifest"]), Path(data["truth"]), algorithm, matrix_dir / "raw" / "confirmation" / f"attempt-{record['attempts']}", replication, tuning=False, cases=case_path), env)
            record.update({"status": "complete", "result": outcome["result"], "run_id": outcome["run_id"]})
        except Exception as error:
            record.update({"status": "failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
            state["status"] = "failed"
            atomic_json(state_path, state)
            raise
        atomic_json(state_path, state)

    incomplete = [record for record in state["confirmation"].values() if record["status"] not in {"complete", "skipped"}]
    if incomplete:
        raise RuntimeError(f"{len(incomplete)} confirmation cases incomplete")
    outcome = run_json([sys.executable, "-m", "experiments.vector_search_cuvs.summarize", "--state", str(state_path), "--output-dir", str(matrix_dir / "report")], env)
    state.update({"status": "complete", "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "report": outcome})
    atomic_json(state_path, state)
    print(json.dumps({"matrix_id": matrix_id, "state": str(state_path), "report": outcome}, indent=2))


if __name__ == "__main__":
    main()
