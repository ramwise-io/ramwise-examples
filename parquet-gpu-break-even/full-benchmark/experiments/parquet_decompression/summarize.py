"""Create a CSV and throughput chart from an immutable benchmark result."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("PARQUET_BENCH_DATA_ROOT", "benchmark-data"))
        / "benchmarks/parquet-decompression/charts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.result.open(encoding="utf-8") as handle:
        result = json.load(handle)
    measured = [trial for trial in result["trials"] if trial["phase"] == "measured"]
    engines = sorted({trial["engine"] for trial in measured})
    rows = []
    for engine in engines:
        values = [trial["throughput_mib_s"] for trial in measured if trial["engine"] == engine]
        elapsed = [trial["elapsed_seconds"] for trial in measured if trial["engine"] == engine]
        rows.append(
            {
                "engine": engine,
                "mean_mib_s": statistics.fmean(values),
                "median_mib_s": statistics.median(values),
                "median_seconds": statistics.median(elapsed),
                "stdev_mib_s": statistics.stdev(values) if len(values) > 1 else 0.0,
                "trials": len(values),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{result['run_id']}-summary.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as pyplot

    figure, axis = pyplot.subplots(figsize=(8, 4.5))
    axis.bar([row["engine"] for row in rows], [row["median_mib_s"] for row in rows])
    axis.set_ylabel("Median throughput (MiB/s)")
    axis.set_title(f"Parquet read throughput: {result['dataset']['dataset_id']}")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    chart_path = args.output_dir / f"{result['run_id']}-throughput.png"
    figure.savefig(chart_path, dpi=160)
    pyplot.close(figure)

    print(f"CSV: {csv_path}")
    print(f"Chart: {chart_path}")
    print("\n| Engine | Median MiB/s | Median seconds | Trials |")
    print("|---|---:|---:|---:|")
    for row in sorted(rows, key=lambda item: item["median_mib_s"], reverse=True):
        print(
            f"| {row['engine']} | {row['median_mib_s']:.1f} | "
            f"{row['median_seconds']:.4f} | {row['trials']} |"
        )


if __name__ == "__main__":
    main()
