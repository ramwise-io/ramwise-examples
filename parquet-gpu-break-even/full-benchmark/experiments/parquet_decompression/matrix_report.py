"""Build cross-case CSV and charts from a Parquet matrix state file."""

from __future__ import annotations

import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.parquet_decompression.common import read_json


def build_summary(state: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for case in state["cases"].values():
        if case["status"] != "complete":
            continue
        result = read_json(Path(case["result"]))
        row_group_rows = case.get("row_group_rows")
        if row_group_rows is None:
            row_group_rows = result["dataset"]["config"]["row_group_rows"]
        row_group_rows = int(row_group_rows)
        grouped[
            (case["profile"], case["codec"], row_group_rows, case["projection"])
        ].append(result)

    rows: list[dict[str, Any]] = []
    for (profile, codec, row_group_rows, projection), results in sorted(grouped.items()):
        engines = sorted(
            {
                trial["engine"]
                for result in results
                for trial in result["trials"]
                if trial["phase"] == "measured"
            }
        )
        for engine in engines:
            run_medians = []
            elapsed_medians = []
            telemetry_run_medians: dict[str, list[float]] = defaultdict(list)
            for result in results:
                trials = [
                    trial
                    for trial in result["trials"]
                    if trial["phase"] == "measured" and trial["engine"] == engine
                ]
                run_medians.append(
                    statistics.median(trial["throughput_mib_s"] for trial in trials)
                )
                elapsed_medians.append(
                    statistics.median(trial["elapsed_seconds"] for trial in trials)
                )
                for field in (
                    "affinity_cpu_utilization_percent_mean",
                    "background_affinity_cpu_cores_mean",
                    "host_cpu_utilization_percent_mean",
                    "process_cpu_cores_mean",
                ):
                    values = [
                        trial.get("gpu_telemetry", {}).get("summary", {}).get(field)
                        for trial in trials
                    ]
                    values = [value for value in values if value is not None]
                    if values:
                        telemetry_run_medians[field].append(statistics.median(values))
            row = {
                "codec": codec,
                "dataset_id": results[0]["dataset"]["dataset_id"],
                "engine": engine,
                "median_elapsed_seconds": statistics.median(elapsed_medians),
                "median_mib_s": statistics.median(run_medians),
                "parquet_mib": results[0]["dataset"]["parquet_bytes"] / 1024**2,
                "profile": profile,
                "projection": projection,
                "replications": len(results),
                "row_group_rows": row_group_rows,
                "run_median_max_mib_s": max(run_medians),
                "run_median_min_mib_s": min(run_medians),
                "trials_per_replication": sum(
                    1
                    for trial in results[0]["trials"]
                    if trial["phase"] == "measured" and trial["engine"] == engine
                ),
            }
            for field, values in telemetry_run_medians.items():
                if len(values) == len(results):
                    row[field] = statistics.median(values)
            rows.append(row)
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_report(state_path: Path, output_dir: Path) -> dict[str, str]:
    state = read_json(state_path)
    rows = build_summary(state)
    if not rows:
        raise ValueError("No completed matrix cases to report")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    _atomic_json(json_path, {"matrix_id": state["matrix_id"], "rows": rows})
    _write_csv(csv_path, rows)

    import matplotlib.pyplot as pyplot

    projections = list(dict.fromkeys(row["projection"] for row in rows))
    engines = sorted({row["engine"] for row in rows})
    conditions = sorted(
        {(row["profile"], row["codec"], row["row_group_rows"]) for row in rows}
    )
    lookup = {
        (
            row["profile"],
            row["codec"],
            row["row_group_rows"],
            row["projection"],
            row["engine"],
        ): row
        for row in rows
    }

    figure, axes = pyplot.subplots(
        len(projections),
        1,
        figsize=(13, 4.5 * len(projections)),
        squeeze=False,
    )
    x_values = list(range(len(conditions)))
    for axis, projection in zip(axes[:, 0], projections, strict=True):
        for engine in engines:
            values = [
                lookup[(profile, codec, row_group_rows, projection, engine)][
                    "median_mib_s"
                ]
                for profile, codec, row_group_rows in conditions
            ]
            axis.plot(x_values, values, marker="o", label=engine)
        axis.set_yscale("log")
        axis.set_ylabel("Median MiB/s (log scale)")
        axis.set_title(f"Projection: {projection}")
        axis.set_xticks(
            x_values,
            [
                f"{profile}\n{codec}\nrg={row_group_rows:,}"
                for profile, codec, row_group_rows in conditions
            ],
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend(ncol=len(engines), fontsize=8)
    figure.suptitle(f"Parquet matrix throughput: {state['matrix_id']}")
    figure.tight_layout()
    throughput_path = output_dir / "throughput.png"
    figure.savefig(throughput_path, dpi=160)
    pyplot.close(figure)

    speedup_rows = []
    for profile, codec, row_group_rows in conditions:
        for projection in projections:
            condition_rows = [
                lookup[(profile, codec, row_group_rows, projection, engine)]
                for engine in engines
            ]
            cudf = next(row for row in condition_rows if row["engine"] == "cudf")
            cpu = max(
                (row for row in condition_rows if row["engine"] != "cudf"),
                key=lambda row: row["median_mib_s"],
            )
            speedup_rows.append(
                {
                    "condition": f"{profile}-{codec}-rg{row_group_rows}-{projection}",
                    "cpu_engine": cpu["engine"],
                    "speedup": cudf["median_mib_s"] / cpu["median_mib_s"],
                }
            )
    figure, axis = pyplot.subplots(figsize=(13, 5))
    axis.bar(
        [row["condition"] for row in speedup_rows],
        [row["speedup"] for row in speedup_rows],
    )
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_ylabel("cuDF / fastest CPU median throughput")
    axis.set_xticks(
        range(len(speedup_rows)),
        [row["condition"] for row in speedup_rows],
        rotation=55,
        ha="right",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    speedup_path = output_dir / "cudf-speedup.png"
    figure.savefig(speedup_path, dpi=160)
    pyplot.close(figure)

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "speedup_chart": str(speedup_path),
        "throughput_chart": str(throughput_path),
    }
