"""Aggregate confirmed cuVS runs and calculate exact-search break-even points."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.vector_search_cuvs.common import atomic_json, canonical_json, read_json


def median(values: list[float]) -> float:
    return statistics.median(values)


def break_even_queries(build_seconds: float, exact_batch_seconds: float, ann_batch_seconds: float, batch: int) -> float | None:
    saved_seconds_per_query = (exact_batch_seconds - ann_batch_seconds) / batch
    return build_seconds / saved_seconds_per_query if saved_seconds_per_query > 0 else None


def aggregate(state: dict[str, Any]) -> list[dict[str, Any]]:
    selection_targets: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for dataset_name, algorithms in state["selections"].items():
        for algorithm, selections in algorithms.items():
            for selected in selections:
                key = (dataset_name, algorithm, int(selected["batch_size"]), canonical_json(selected["params"]))
                selection_targets[key].append(float(selected["target_recall"]))

    groups: dict[tuple[str, str, int, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record in state["confirmation"].values():
        if record["status"] != "complete":
            continue
        result = read_json(Path(record["result"]))
        dataset_name = record["dataset"]
        algorithm = record["algorithm"]
        for search in result["searches"]:
            key = (dataset_name, algorithm, int(search["batch_size"]), canonical_json(search["params"]))
            groups[key].append((result, search))

    rows: list[dict[str, Any]] = []
    for key, values in groups.items():
        dataset_name, algorithm, batch, params_json = key
        first = values[0][0]
        build_seconds = median([float(result["build_seconds"]) for result, _ in values])
        latency_ms = median([float(search["latency_ms_per_batch"]) for _, search in values])
        build_memory = median([
            float(result["memory"]["after_build"]["device_used_bytes"])
            for result, _ in values
        ])
        direct_search_memory = [
            search["sustained_telemetry"]["gpu"].get("memory_used_max_bytes")
            for _, search in values
        ]
        search_memory = median([
            float(measured if measured is not None else result["memory"]["after_build"]["device_used_bytes"])
            for (result, _), measured in zip(values, direct_search_memory)
        ])
        rows.append({
            "dataset": dataset_name,
            "dataset_id": first["dataset"]["dataset_id"],
            "rows": int(first["dataset"]["config"]["rows"]),
            "dimensions": int(first["dataset"]["config"]["dimensions"]),
            "algorithm": algorithm,
            "batch_size": batch,
            "params": params_json,
            "tuning_target_recalls": ";".join(f"{x:.2f}" for x in sorted(set(selection_targets.get(key, [1.0] if algorithm == "brute_force" else [])))),
            "replications": len(values),
            "build_seconds_median": build_seconds,
            "corpus_bytes": int(first["dataset"]["corpus"]["bytes"]),
            "index_bytes_median": int(median([float(result["serialized_index_bytes"]) for result, _ in values])),
            "device_memory_after_build_bytes_median": int(build_memory),
            "device_memory_search_max_bytes_median": int(search_memory),
            "search_memory_telemetry_replications": sum(value is not None for value in direct_search_memory),
            "recall_at_10_median": median([float(search["recall_at_k"]) for _, search in values]),
            "recall_at_10_min": min(float(search["recall_at_k"]) for _, search in values),
            "latency_ms_per_batch_median": latency_ms,
            "latency_ms_per_query_median": latency_ms / batch,
            "queries_per_second_median": batch / (latency_ms / 1000),
        })

    for row in rows:
        tuned = [float(value) for value in row["tuning_target_recalls"].split(";") if value]
        row["confirmed_target_recalls"] = ";".join(
            f"{target:.2f}" for target in tuned if row["recall_at_10_min"] >= target
        )

    exact = {(row["dataset"], row["batch_size"]): row for row in rows if row["algorithm"] == "brute_force"}
    rates = state["config"]["benchmark"]["query_rates_per_second"]
    for row in rows:
        if row["algorithm"] == "brute_force":
            row["break_even_status"] = "reference"
            row["break_even_queries"] = 0.0
            for rate in rates:
                row[f"break_even_seconds_at_{rate}_qps"] = 0.0
            continue
        reference = exact[(row["dataset"], row["batch_size"])]
        count = break_even_queries(
            row["build_seconds_median"],
            reference["latency_ms_per_batch_median"] / 1000,
            row["latency_ms_per_batch_median"] / 1000,
            row["batch_size"],
        )
        row["break_even_status"] = "finite" if count is not None else "ann-not-faster"
        row["break_even_queries"] = count
        for rate in rates:
            row[f"break_even_seconds_at_{rate}_qps"] = count / rate if count is not None else None
    return sorted(rows, key=lambda row: (row["rows"], row["dimensions"], row["dataset"], row["batch_size"], row["algorithm"], row["params"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    state = read_json(args.state)
    rows = aggregate(state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "summary.csv"
    atomic_json(json_path, {"matrix_id": state["matrix_id"], "rows": rows})
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
