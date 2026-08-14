"""Summarize end-to-end GPU ML pipeline residency results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.gpu_ml_pipeline.common import atomic_json, read_json

STAGE_NAMES = (
    "load",
    "host_to_device",
    "device_to_host",
    "scale",
    "feature_engineering",
    "pca",
    "train",
    "inference",
    "materialize",
)


def break_even_reuses(cpu_fixed: float, mode_fixed: float, cpu_repeat: float, mode_repeat: float) -> int | None:
    savings = cpu_repeat - mode_repeat
    if savings <= 0:
        return None
    return max(0, math.ceil((mode_fixed - cpu_fixed) / savings))


def med(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    state = read_json(args.state.resolve())
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in state["runs"].values():
        if record["status"] == "complete":
            result = read_json(Path(record["result"]))
            grouped[(result["dataset"]["dataset_id"], result["condition"]["mode"])].append(result)
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    for key, items in grouped.items():
        config = items[0]["dataset"]["config"]
        trial_sets = [trial for item in items for trial in item["trials"]]
        stage_medians = {
            f"{name}_seconds": med([float(trial["stages"].get(name, 0.0)) for trial in trial_sets])
            for name in STAGE_NAMES
        }
        total = med([float(item["total_median_seconds"]) for item in items])
        totals = [float(item["total_median_seconds"]) for item in items]
        repeated = stage_medians.get("inference_seconds", 0.0) + stage_medians.get("materialize_seconds", 0.0)
        aggregates[key] = {
            "dataset_id": key[0],
            "rows": int(config["rows"]),
            "features": int(config["features"]),
            "input_gib": int(config["rows"]) * int(config["features"]) * 4 / 2**30,
            "clusters": int(config["clusters"]),
            "informative_features": int(config["informative_features"]),
            "cluster_scale": float(config["cluster_scale"]),
            "mode": key[1],
            "replications": len(items),
            "total_seconds": total,
            "total_seconds_min": min(totals),
            "total_seconds_max": max(totals),
            "fixed_seconds": total - repeated,
            "repeat_inference_seconds": repeated,
            "roc_auc": med([float(item["quality"]["roc_auc"]) for item in items]),
            "transfer_count": med([float(trial["transfers"]["count"]) for trial in trial_sets]),
            "host_to_device_gib": med([float(trial["transfers"]["host_to_device"]) for trial in trial_sets]) / 2**30,
            "device_to_host_gib": med([float(trial["transfers"]["device_to_host"]) for trial in trial_sets]) / 2**30,
            "peak_gpu_memory_gib": max(
                [float(trial["gpu"].get("memory_used_max_bytes", 0)) for trial in trial_sets],
                default=0.0,
            ) / 2**30,
            **stage_medians,
        }
    rows = []
    for key in sorted(aggregates):
        row = aggregates[key]
        cpu = aggregates[(key[0], "cpu")]
        rows.append(
            {
                **row,
                "speedup_vs_cpu": cpu["total_seconds"] / row["total_seconds"],
                "quality_gap_vs_cpu": row["roc_auc"] - cpu["roc_auc"],
                "break_even_inference_reuses": break_even_reuses(
                    cpu["fixed_seconds"],
                    row["fixed_seconds"],
                    cpu["repeat_inference_seconds"],
                    row["repeat_inference_seconds"],
                ),
            }
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "pipeline_results.csv"
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"matrix_id": state["matrix_id"], "rows": len(rows), "pipeline_results": str(result_path)}
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
