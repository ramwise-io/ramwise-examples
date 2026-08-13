"""Aggregate multiple immutable benchmark runs without hiding run-level variance."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from experiments.parquet_decompression.common import read_json, write_new_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("PARQUET_BENCH_DATA_ROOT", "benchmark-data"))
        / "benchmarks/parquet-decompression/aggregate",
    )
    return parser.parse_args()


def validate_compatible(results: list[dict[str, Any]]) -> None:
    first = results[0]
    fields = (
        ("experiment",),
        ("schema_version",),
        ("cache_state",),
        ("columns",),
        ("dataset", "dataset_id"),
        ("policy", "threads"),
        ("policy", "timed_region"),
    )
    for result in results[1:]:
        for path in fields:
            left: Any = first
            right: Any = result
            for key in path:
                left = left[key]
                right = right[key]
            if left != right:
                raise ValueError(f"Incompatible result field: {'.'.join(path)}")


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    engines = sorted(
        {
            trial["engine"]
            for result in results
            for trial in result["trials"]
            if trial["phase"] == "measured"
        }
    )
    rows = []
    for engine in engines:
        trial_values = []
        run_medians = []
        for result in results:
            values = [
                trial["throughput_mib_s"]
                for trial in result["trials"]
                if trial["phase"] == "measured" and trial["engine"] == engine
            ]
            if not values:
                raise ValueError(f"Run {result['run_id']} has no measured {engine} trials")
            trial_values.extend(values)
            run_medians.append(statistics.median(values))
        rows.append(
            {
                "engine": engine,
                "max_run_median_mib_s": max(run_medians),
                "median_mib_s": statistics.median(trial_values),
                "median_of_run_medians_mib_s": statistics.median(run_medians),
                "min_run_median_mib_s": min(run_medians),
                "run_median_stdev_mib_s": statistics.stdev(run_medians) if len(run_medians) > 1 else 0.0,
                "runs": len(results),
                "trial_stdev_mib_s": statistics.stdev(trial_values) if len(trial_values) > 1 else 0.0,
                "trials": len(trial_values),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    results = [read_json(path) for path in args.results]
    validate_compatible(results)
    rows = summarize(results)
    aggregate_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{uuid.uuid4().hex[:8]}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / f"{aggregate_id}-summary.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {
        "aggregate_id": aggregate_id,
        "dataset_id": results[0]["dataset"]["dataset_id"],
        "source_run_ids": [result["run_id"] for result in results],
        "source_git_commits": sorted({result["git_commit"] for result in results}),
        "summary": rows,
    }
    json_path = args.output_dir / f"{aggregate_id}.json"
    write_new_json(json_path, aggregate)

    import matplotlib.pyplot as pyplot

    ordered = sorted(rows, key=lambda row: row["median_of_run_medians_mib_s"], reverse=True)
    centers = [row["median_of_run_medians_mib_s"] for row in ordered]
    lower = [center - row["min_run_median_mib_s"] for center, row in zip(centers, ordered, strict=True)]
    upper = [row["max_run_median_mib_s"] - center for center, row in zip(centers, ordered, strict=True)]
    figure, axis = pyplot.subplots(figsize=(8, 4.5))
    axis.bar([row["engine"] for row in ordered], centers, yerr=[lower, upper], capsize=5)
    axis.set_ylabel("Median throughput (MiB/s)")
    axis.set_title(f"{len(results)}-run Parquet pilot (error bars: run-median range)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    chart_path = args.output_dir / f"{aggregate_id}-throughput.png"
    figure.savefig(chart_path, dpi=160)
    pyplot.close(figure)

    print(f"Aggregate: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Chart: {chart_path}")
    print("\n| Engine | Median of run medians MiB/s | Run range | Runs | Trials |")
    print("|---|---:|---:|---:|---:|")
    for row in ordered:
        print(
            f"| {row['engine']} | {row['median_of_run_medians_mib_s']:.1f} | "
            f"{row['min_run_median_mib_s']:.1f}-{row['max_run_median_mib_s']:.1f} | "
            f"{row['runs']} | {row['trials']} |"
        )


if __name__ == "__main__":
    main()
