"""Aggregate custom CUDA benchmark replications into publication tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.custom_cuda.common import atomic_json, read_json


def med(values: list[float]) -> float:
    return float(statistics.median(values))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_launch(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for result in results:
        for record in result["records"]:
            grouped[(int(result["condition"]["launches"]), record["implementation"])].append((result, record))
    rows = []
    for (launches, implementation), items in sorted(grouped.items()):
        wall = [float(record["wall_seconds_per_launch"]) for _, record in items]
        device = [float(record["device_seconds_per_launch"]) for _, record in items]
        compile_times = [float(result["compile_seconds"]) for result, _ in items]
        rows.append(
            {
                "launches_per_trial": launches,
                "implementation": implementation,
                "replications": len(items),
                "wall_microseconds_per_launch": med(wall) * 1e6,
                "wall_microseconds_per_launch_min": min(wall) * 1e6,
                "wall_microseconds_per_launch_max": max(wall) * 1e6,
                "device_microseconds_per_launch": med(device) * 1e6,
                "device_microseconds_per_launch_min": min(device) * 1e6,
                "device_microseconds_per_launch_max": max(device) * 1e6,
                "cold_compile_seconds": med(compile_times),
            }
        )
    return rows


def aggregate_transfer(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        record = result["records"][0]
        key = (int(record["bytes"]), record["direction"], record["memory"])
        grouped[key].append(record)
    rows = []
    for (size, direction, memory), items in sorted(grouped.items()):
        wall = [float(item["wall_seconds"]) for item in items]
        device = [float(item["device_seconds"]) for item in items]
        rows.append(
            {
                "bytes": size,
                "kib": size / 2**10,
                "direction": direction,
                "memory": memory,
                "replications": len(items),
                "wall_seconds": med(wall),
                "wall_seconds_min": min(wall),
                "wall_seconds_max": max(wall),
                "device_seconds": med(device),
                "device_seconds_min": min(device),
                "device_seconds_max": max(device),
                "wall_gib_per_second": size / med(wall) / 2**30,
                "device_gib_per_second": size / med(device) / 2**30,
            }
        )
    return rows


def aggregate_fusion(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str, Any, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for result in results:
        condition = result["condition"]
        for record in result["records"]:
            key = (
                int(condition["rows"]),
                int(condition["features"]),
                record["implementation"],
                record["block_size"],
                record["layout"],
            )
            grouped[key].append((result, record))
    rows = []
    for key, items in sorted(grouped.items(), key=lambda entry: tuple(str(v) for v in entry[0])):
        row_count, features, implementation, block_size, layout = key
        resident = [float(record["resident_wall_seconds"]) for _, record in items]
        device = [
            float(record["resident_device_seconds"])
            for _, record in items
            if record["resident_device_seconds"] is not None
        ]
        one_shot = [
            float(record["one_shot_seconds"])
            for _, record in items
            if record.get("one_shot_seconds") is not None
        ]
        if implementation.startswith("raw_"):
            compile_times = [
                float(result["compile_seconds"]["raw_module"])
                for result, _ in items
            ]
        elif implementation == "cupy_partial_fusion":
            compile_times = [
                float(result["compile_seconds"]["cupy_partial_fusion"])
                for result, _ in items
            ]
        else:
            compile_times = []
        conversions = [
            float(result["layout_conversion"]["aos_to_soa_wall_seconds"])
            for result, _ in items
        ]
        kernel = next((record.get("kernel") for _, record in items if record.get("kernel")), None)
        quality_rows = [record["quality"] for _, record in items]
        rows.append(
            {
                "rows": row_count,
                "features": features,
                "input_bytes": row_count * features * 4,
                "implementation": implementation,
                "layout": layout,
                "block_size": block_size,
                "replications": len(items),
                "resident_wall_seconds": med(resident),
                "resident_wall_seconds_min": min(resident),
                "resident_wall_seconds_max": max(resident),
                "resident_device_seconds": med(device) if device else None,
                "one_shot_seconds": med(one_shot) if one_shot else None,
                "one_shot_seconds_min": min(one_shot) if one_shot else None,
                "one_shot_seconds_max": max(one_shot) if one_shot else None,
                "cold_compile_seconds": med(compile_times) if compile_times else None,
                "aos_to_soa_conversion_seconds": med(conversions),
                "max_abs_score_error": max(
                    float(item.get("max_abs_score_error", 0.0)) for item in quality_rows
                ),
                "flag_mismatches": max(
                    int(item.get("flag_mismatches", 0)) for item in quality_rows
                ),
                "registers_per_thread": kernel.get("registers_per_thread") if kernel else None,
                "static_shared_memory_bytes": kernel.get("static_shared_memory_bytes") if kernel else None,
                "active_blocks_per_multiprocessor": kernel.get("active_blocks_per_multiprocessor") if kernel else None,
                "theoretical_occupancy": kernel.get("theoretical_occupancy") if kernel else None,
            }
        )
    references = {
        (row["rows"], row["features"]): row
        for row in rows
        if row["implementation"] == "numpy"
    }
    for row in rows:
        reference = references[(row["rows"], row["features"])]
        row["resident_speedup_vs_numpy"] = (
            reference["resident_wall_seconds"] / row["resident_wall_seconds"]
        )
        row["one_shot_speedup_vs_numpy"] = (
            reference["one_shot_seconds"] / row["one_shot_seconds"]
            if row["one_shot_seconds"]
            else None
        )
    return rows


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
    by_mode = {
        mode: [result for result in results if result["mode"] == mode]
        for mode in ("launch", "transfer", "fusion")
    }
    launch_rows = aggregate_launch(by_mode["launch"])
    transfer_rows = aggregate_transfer(by_mode["transfer"])
    fusion_rows = aggregate_fusion(by_mode["fusion"])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    launch_path = output / "launch_results.csv"
    transfer_path = output / "transfer_results.csv"
    fusion_path = output / "fusion_results.csv"
    write_csv(launch_path, launch_rows)
    write_csv(transfer_path, transfer_rows)
    write_csv(fusion_path, fusion_rows)
    environments = [result["environment"] for result in results]
    summary = {
        "matrix_id": state["matrix_id"],
        "git_commit": state["git_commit"],
        "runs": len(results),
        "launch_rows": len(launch_rows),
        "transfer_rows": len(transfer_rows),
        "fusion_rows": len(fusion_rows),
        "launch_results": str(launch_path),
        "transfer_results": str(transfer_path),
        "fusion_results": str(fusion_path),
        "environment": environments[0] if environments else {},
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
