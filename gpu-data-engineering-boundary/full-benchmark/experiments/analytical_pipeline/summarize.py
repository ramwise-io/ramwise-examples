"""Summarize immutable analytical-pipeline result JSON files."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

from experiments.analytical_pipeline.common import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        measured = [trial for trial in result["trials"] if trial["phase"] == "measured"]
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for trial in measured:
            groups.setdefault((trial["workload"], trial["engine"]), []).append(trial)
        medians = {
            key: statistics.median(item["elapsed_seconds"] for item in trials)
            for key, trials in groups.items()
        }
        for (workload, engine), trials in sorted(groups.items()):
            telemetry = [trial["gpu_telemetry"]["summary"] for trial in trials]
            device_memory_max = max(
                (item.get("memory_used_bytes_max", 0) for item in telemetry),
                default=0,
            )
            gpu_utilization = statistics.fmean(
                item.get("utilization_gpu_percent_mean", 0.0) for item in telemetry
            )
            cpu_cores = statistics.fmean(
                item.get("process_cpu_cores_mean", 0.0) for item in telemetry
            )
            process_rss_peak = max(
                (item.get("process_rss_bytes_max", 0) for item in telemetry),
                default=0,
            )
            elapsed = medians[(workload, engine)]
            fastest = min(
                value for (candidate_workload, _), value in medians.items()
                if candidate_workload == workload
            )
            rows.append(
                {
                    "run_id": result["run_id"],
                    "profile": result["dataset"]["config"]["profile"],
                    "codec": result["dataset"]["config"]["codec"],
                    "rows": result["dataset"]["config"]["rows"],
                    "parquet_bytes": result["dataset"]["parquet_bytes"],
                    "estimated_logical_bytes": result["dataset"]["estimated_logical_bytes"],
                    "workload": workload,
                    "engine": engine,
                    "median_seconds": elapsed,
                    "min_seconds": min(item["elapsed_seconds"] for item in trials),
                    "max_seconds": max(item["elapsed_seconds"] for item in trials),
                    "relative_to_fastest": elapsed / fastest,
                    "device_memory_used_max_bytes": device_memory_max,
                    "gpu_utilization_mean_percent": gpu_utilization,
                    "process_cpu_cores_mean": cpu_cores,
                    "process_rss_peak_bytes": process_rss_peak,
                    "trials": len(trials),
                    "correct": all(item["correct"] for item in trials),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    results = [read_json(path) for path in args.results]
    rows = summarize_results(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
