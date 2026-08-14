"""Summarize confirmed GPU ML estimator and inference measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.gpu_ml_pipeline.common import atomic_json, primary_quality_metric, read_json


def break_even_batches(
    cpu_fixed: float,
    candidate_fixed: float,
    cpu_inference: float,
    candidate_inference: float,
) -> int | None:
    savings = cpu_inference - candidate_inference
    if savings <= 0:
        return None
    return max(0, math.ceil((candidate_fixed - cpu_fixed) / savings))


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    state = read_json(args.state.resolve())
    results = [
        read_json(Path(record["result"]))
        for record in state["runs"].values()
        if record["status"] == "complete"
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[
            (
                result["dataset"]["dataset_id"],
                result["condition"]["algorithm"],
                result["condition"]["engine"],
            )
        ].append(result)
    aggregates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, items in grouped.items():
        config = items[0]["dataset"]["config"]
        algorithm = items[0]["condition"]["algorithm"]
        metric, _ = primary_quality_metric(algorithm)
        peaks = [
            float(trial["gpu"].get("memory_used_max_bytes", 0))
            for item in items
            for trial in item["timing"]["fit_trials"]
        ]
        energies = [
            float(trial["gpu"].get("energy_joules_estimate", 0))
            for item in items
            for trial in item["timing"]["fit_trials"]
        ]
        aggregates[key] = {
            "dataset_id": key[0],
            "rows": int(config["rows"]),
            "features": int(config["features"]),
            "input_gib": int(config["rows"]) * int(config["features"]) * 4 / 2**30,
            "clusters": int(config["clusters"]),
            "informative_features": int(config["informative_features"]),
            "cluster_scale": float(config["cluster_scale"]),
            "algorithm": algorithm,
            "engine": key[2],
            "replications": len(items),
            "load_seconds": median([x["timing"]["load_seconds"] for x in items]),
            "explicit_transfer_seconds": median(
                [x["timing"]["explicit_transfer_seconds"] for x in items]
            ),
            "fit_seconds": median([x["timing"]["fit_median_seconds"] for x in items]),
            "fit_seconds_min": min(
                float(x["timing"]["fit_median_seconds"]) for x in items
            ),
            "fit_seconds_max": max(
                float(x["timing"]["fit_median_seconds"]) for x in items
            ),
            "one_shot_fit_seconds": median(
                [x["timing"]["one_shot_fit_seconds"] for x in items]
            ),
            "one_shot_fit_seconds_min": min(
                float(x["timing"]["one_shot_fit_seconds"]) for x in items
            ),
            "one_shot_fit_seconds_max": max(
                float(x["timing"]["one_shot_fit_seconds"]) for x in items
            ),
            "quality_metric": metric,
            "quality_value": median([float(x["quality"][metric]) for x in items]),
            "peak_gpu_memory_gib": max(peaks, default=0.0) / 2**30,
            "fit_energy_joules": median(energies) if energies else 0.0,
        }
    condition_rows = []
    for key in sorted(aggregates):
        row = aggregates[key]
        cpu = aggregates[(key[0], key[1], "cpu")]
        condition_rows.append(
            {
                **row,
                "fit_speedup_vs_cpu": cpu["fit_seconds"] / row["fit_seconds"],
                "one_shot_speedup_vs_cpu": cpu["one_shot_fit_seconds"]
                / row["one_shot_fit_seconds"],
                "quality_gap_vs_cpu": row["quality_value"] - cpu["quality_value"],
            }
        )

    inference_grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        for measurement in result["timing"]["inference"]:
            inference_grouped[
                (
                    result["dataset"]["dataset_id"],
                    result["condition"]["algorithm"],
                    result["condition"]["engine"],
                    int(measurement["batch_size"]),
                )
            ].append(measurement)
    inference_aggregates = {
        key: median([x["end_to_end_seconds"] for x in values])
        for key, values in inference_grouped.items()
    }
    inference_rows = []
    for key in sorted(inference_aggregates):
        dataset_id, algorithm, engine, batch = key
        cpu_key = (dataset_id, algorithm, "cpu", batch)
        if cpu_key not in inference_aggregates:
            continue
        seconds = inference_aggregates[key]
        cpu_seconds = inference_aggregates[cpu_key]
        measured_seconds = [float(x["end_to_end_seconds"]) for x in inference_grouped[key]]
        fixed = aggregates[(dataset_id, algorithm, engine)]["one_shot_fit_seconds"]
        cpu_fixed = aggregates[(dataset_id, algorithm, "cpu")]["one_shot_fit_seconds"]
        inference_rows.append(
            {
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "engine": engine,
                "batch_size": batch,
                "latency_ms": seconds * 1000,
                "latency_ms_min": min(measured_seconds) * 1000,
                "latency_ms_max": max(measured_seconds) * 1000,
                "rows_per_second": batch / seconds,
                "speedup_vs_cpu": cpu_seconds / seconds,
                "break_even_batches": break_even_batches(
                    cpu_fixed, fixed, cpu_seconds, seconds
                ),
            }
        )
    output = args.output_dir.resolve()
    condition_path = output / "condition_results.csv"
    inference_path = output / "inference_results.csv"
    write_csv(condition_path, condition_rows)
    write_csv(inference_path, inference_rows)
    summary = {
        "matrix_id": state["matrix_id"],
        "condition_rows": len(condition_rows),
        "inference_rows": len(inference_rows),
        "condition_results": str(condition_path),
        "inference_results": str(inference_path),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
