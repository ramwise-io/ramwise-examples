"""Run a resumable, randomized Spark RAPIDS matrix in isolated JVM processes."""

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

from experiments.spark_rapids_fallback.common import canonical_json, read_json

GENERATED_ROOT = Path("/data/generated/analytical-pipeline")
BENCHMARK_ROOT = Path("/data/benchmarks/spark-rapids-fallback/matrices")


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


def atomic_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resolve_manifest(rows: int) -> Path:
    matches = []
    for path in GENERATED_ROOT.glob("wide-zstd-*/manifest.json"):
        manifest = read_json(path)
        config = manifest["config"]
        if (
            config["rows"] == rows
            and config["seed"] == 73
            and config["profile"] == "wide"
            and config["codec"] == "zstd"
            and config["row_group_rows"] == 262_144
        ):
            matches.append((config["rows_per_file"], str(path), path))
    if not matches:
        raise FileNotFoundError(f"No deterministic wide Zstd manifest for {rows} rows")
    # Prefer the established crossover layout: no more than 2.5M rows per file.
    eligible = [item for item in matches if item[0] <= 2_500_000]
    return max(eligible or matches)[2]


def cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for rows in config["rows"]:
        for topology in config["topologies"]:
            for mode in config["modes"]:
                for python_bridge in config.get("python_bridges", ["forced-cpu"]):
                    for replication in range(config["replications"]):
                        result.append(
                            {
                                "case_id": f"r{rows}-{topology}-{mode}-{python_bridge}-rep{replication}",
                                "mode": mode,
                                "python_bridge": python_bridge,
                                "replication": replication,
                                "rows": rows,
                                "topology": topology,
                            }
                        )
    random.Random(config["benchmark"]["seed"]).shuffle(result)
    return result


def run_json(command: list[str], env: dict[str, str]) -> dict[str, Any]:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, text=True, capture_output=True, env=env)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout[-4000:]}\n{completed.stderr[-8000:]}"
        )
    decoder = json.JSONDecoder()
    for start, character in reversed(list(enumerate(completed.stdout))):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(completed.stdout[start:])
        except json.JSONDecodeError:
            continue
        if not completed.stdout[start + end :].strip() and isinstance(value, dict):
            return value
    raise RuntimeError("Benchmark output did not end with JSON")


def benchmark_command(
    config: dict[str, Any],
    manifest: Path,
    reference: Path | None,
    output_root: Path,
    *,
    mode: str,
    topology: str,
    python_bridge: str,
    replication: int,
    seed: int,
    warmups: int,
    trials: int,
    retry_attempt: int = 0,
) -> list[str]:
    benchmark = config["benchmark"]
    command = [
        sys.executable,
        "-m",
        "experiments.spark_rapids_fallback.benchmark",
        "--manifest",
        str(manifest),
        "--mode",
        mode,
        "--topology",
        topology,
        "--python-bridge",
        python_bridge,
        "--output-root",
        str(output_root),
        "--warmups",
        str(warmups),
        "--trials",
        str(trials),
        "--threads",
        str(benchmark["threads"]),
        "--shuffle-partitions",
        str(benchmark["shuffle_partitions"]),
        "--seed",
        str(seed),
        "--replication",
        str(replication),
        "--telemetry-interval-ms",
        str(benchmark["telemetry_interval_ms"]),
    ]
    if reference is not None:
        command.extend(["--reference", str(reference)])
    if retry_attempt:
        command.extend(["--retry-attempt", str(retry_attempt)])
    return command


def ensure_reference(
    config: dict[str, Any], matrix_dir: Path, rows: int, manifest: Path, env: dict[str, str]
) -> Path:
    reference = matrix_dir / "references" / f"r{rows}.json"
    if reference.exists():
        return reference
    outcome = run_json(
        benchmark_command(
            config,
            manifest,
            None,
            matrix_dir / "reference-raw",
            mode="cpu",
            topology="native",
            python_bridge="forced-cpu",
            replication=10_000 + rows,
            seed=config["benchmark"]["seed"] + rows,
            warmups=0,
            trials=1,
        ),
        env,
    )
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(
        json.dumps(outcome["reference"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    commit = source_id()
    matrix_id, digest = matrix_identity(config, commit)
    matrix_dir = BENCHMARK_ROOT / matrix_id
    state_path = matrix_dir / "state.json"
    ordered_cases = cases(config)
    if state_path.exists():
        state = read_json(state_path)
        if state["identity_sha256"] != digest:
            raise RuntimeError("Matrix identity mismatch")
    else:
        state = {
            "cases": {
                case["case_id"]: {**case, "attempts": 0, "status": "pending"}
                for case in ordered_cases
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

    manifests = {rows: resolve_manifest(rows) for rows in config["rows"]}
    env = dict(os.environ, GPU_LAB_SOURCE_ID=commit)
    references = {
        rows: ensure_reference(config, matrix_dir, rows, manifests[rows], env)
        for rows in config["rows"]
    }
    state["status"] = "running"
    atomic_state(state_path, state)
    for index, case in enumerate(ordered_cases):
        record = state["cases"][case["case_id"]]
        if record["status"] == "complete":
            continue
        if record["status"] == "failed" and not args.retry_failed:
            continue
        record.update({"attempts": record["attempts"] + 1, "status": "running"})
        atomic_state(state_path, state)
        seed = config["benchmark"]["seed"] + index + case["replication"] * 10_000
        try:
            outcome = run_json(
                benchmark_command(
                    config,
                    manifests[case["rows"]],
                    references[case["rows"]],
                    matrix_dir / "raw",
                    mode=case["mode"],
                    topology=case["topology"],
                    python_bridge=case["python_bridge"],
                    replication=case["replication"],
                    seed=seed,
                    warmups=config["benchmark"]["warmups"],
                    trials=config["benchmark"]["trials"],
                    retry_attempt=record["attempts"] if record["attempts"] > 1 else 0,
                ),
                env,
            )
            record.update(
                {
                    "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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

    incomplete = [record for record in state["cases"].values() if record["status"] != "complete"]
    if incomplete:
        state["status"] = "incomplete"
        atomic_state(state_path, state)
        raise RuntimeError(f"{len(incomplete)} cases remain incomplete")
    summary = matrix_dir / "summary.csv"
    command = [
        sys.executable,
        "-m",
        "experiments.spark_rapids_fallback.summarize",
        *[state["cases"][case["case_id"]]["result"] for case in ordered_cases],
        "--output",
        str(summary),
    ]
    subprocess.run(command, check=True, env=env)
    state.update(
        {
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "complete",
            "summary": str(summary),
        }
    )
    atomic_state(state_path, state)
    print(json.dumps({"matrix_id": matrix_id, "state": str(state_path), "summary": str(summary)}, indent=2))


if __name__ == "__main__":
    main()
