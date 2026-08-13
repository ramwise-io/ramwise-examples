"""Derive a compact CSV from immutable Spark RAPIDS result JSON files."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

from experiments.spark_rapids_fallback.common import read_json


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        measured = [trial for trial in result["trials"] if trial["phase"] == "measured"]
        telemetry = [trial["telemetry"]["summary"] for trial in measured]
        plan = measured[-1]["plan"]
        rows.append(
            {
                "run_id": result["run_id"],
                "rows": result["dataset"]["config"]["rows"],
                "parquet_bytes": result["dataset"]["parquet_bytes"],
                "logical_bytes": result["dataset"]["estimated_logical_bytes"],
                "topology": result["benchmark"]["topology"],
                "udf_width": __import__("experiments.spark_rapids_fallback.workload", fromlist=["udf_width"]).udf_width(result["benchmark"]["topology"]),
                "mode": result["benchmark"]["mode"],
                "python_bridge": result["benchmark"].get("python_bridge", "forced-cpu"),
                "replication": result["benchmark"]["replication"],
                "median_seconds": statistics.median(trial["elapsed_seconds"] for trial in measured),
                "min_seconds": min(trial["elapsed_seconds"] for trial in measured),
                "max_seconds": max(trial["elapsed_seconds"] for trial in measured),
                "gpu_plan_lines": plan["gpu_plan_lines"],
                "python_plan_lines": plan["python_plan_lines"],
                "transitions": plan["transitions"],
                "gpu_utilization_mean_percent": statistics.fmean(item.get("gpu_utilization_percent_mean", 0.0) for item in telemetry),
                "gpu_memory_used_max_bytes": max(item.get("gpu_memory_used_bytes_max", 0) for item in telemetry),
                "gpu_energy_joules_estimate": statistics.fmean(item.get("gpu_energy_joules_estimate", 0.0) for item in telemetry),
                "process_tree_cpu_cores_mean": statistics.fmean(item.get("process_tree_cpu_cores_mean", 0.0) for item in telemetry),
                "process_tree_rss_max_bytes": max(item.get("process_tree_rss_bytes_max", 0) for item in telemetry),
                "correct": all(trial["correct"] for trial in measured),
                "trials": len(measured),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize_results([read_json(path) for path in args.results])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
