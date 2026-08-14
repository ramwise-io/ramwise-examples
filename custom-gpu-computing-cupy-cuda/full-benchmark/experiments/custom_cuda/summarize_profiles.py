"""Derive a compact publication table from private Nsight Compute reports."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from experiments.custom_cuda.common import atomic_json


PROFILE_PATTERN = re.compile(
    r"^r(?P<rows>\d+)-f(?P<features>\d+)-(?P<layout>aos|soa)-b(?P<block>\d+)\.ncu-rep$"
)


def parse_export(text: str) -> list[dict[str, str]]:
    start = text.find('"ID","Process ID"')
    if start < 0:
        raise ValueError("Nsight CSV header not found")
    return list(csv.DictReader(io.StringIO(text[start:])))


def metric(rows: list[dict[str, str]], section: str, name: str) -> tuple[float, str]:
    matches = [
        row
        for row in rows
        if row.get("Section Name") == section and row.get("Metric Name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {section}/{name} metric, found {len(matches)}")
    return float(matches[0]["Metric Value"].replace(",", "")), matches[0]["Metric Unit"]


def duration_seconds(value: float, unit: str) -> float:
    factors = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    if unit not in factors:
        raise ValueError(f"Unsupported duration unit: {unit}")
    return value * factors[unit]


def summarize_report(report: Path, ncu: Path) -> dict[str, Any]:
    match = PROFILE_PATTERN.match(report.name)
    if not match:
        raise ValueError(f"Unrecognized profile name: {report.name}")
    completed = subprocess.run(
        [str(ncu), "--import", str(report), "--page", "details", "--csv"],
        text=True,
        capture_output=True,
        check=True,
    )
    rows = parse_export(completed.stdout)
    duration, duration_unit = metric(
        rows, "GPU Speed Of Light Throughput", "Duration"
    )
    values = match.groupdict()
    return {
        "profile": report.stem,
        "rows": int(values["rows"]),
        "features": int(values["features"]),
        "layout": values["layout"],
        "block_size": int(values["block"]),
        "kernel": rows[0]["Kernel Name"],
        "profiled_duration_seconds": duration_seconds(duration, duration_unit),
        "memory_throughput_gbyte_s": metric(
            rows, "Memory Workload Analysis", "Memory Throughput"
        )[0],
        "dram_throughput_percent": metric(
            rows, "GPU Speed Of Light Throughput", "DRAM Throughput"
        )[0],
        "l1_tex_hit_rate_percent": metric(
            rows, "Memory Workload Analysis", "L1/TEX Hit Rate"
        )[0],
        "l2_hit_rate_percent": metric(
            rows, "Memory Workload Analysis", "L2 Hit Rate"
        )[0],
        "mem_busy_percent": metric(
            rows, "Memory Workload Analysis", "Mem Busy"
        )[0],
        "mem_pipes_busy_percent": metric(
            rows, "Memory Workload Analysis", "Mem Pipes Busy"
        )[0],
        "compute_sm_throughput_percent": metric(
            rows, "GPU Speed Of Light Throughput", "Compute (SM) Throughput"
        )[0],
        "registers_per_thread": int(
            metric(rows, "Launch Statistics", "Registers Per Thread")[0]
        ),
        "waves_per_sm": metric(rows, "Launch Statistics", "Waves Per SM")[0],
        "theoretical_occupancy_percent": metric(
            rows, "Occupancy", "Theoretical Occupancy"
        )[0],
        "achieved_occupancy_percent": metric(
            rows, "Occupancy", "Achieved Occupancy"
        )[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ncu", type=Path, required=True)
    args = parser.parse_args()
    reports = sorted(
        path
        for path in args.input_dir.resolve().glob("*.ncu-rep")
        if PROFILE_PATTERN.match(path.name)
    )
    if not reports:
        raise RuntimeError("No recognized Nsight reports found")
    rows = [summarize_report(report, args.ncu.resolve()) for report in reports]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "profile_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    version = subprocess.run(
        [str(args.ncu.resolve()), "--version"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    summary = {
        "profiles": len(rows),
        "profile_results": str(csv_path),
        "ncu_version": version,
        "note": "Profiled duration is replay/instrumentation time; use benchmark CSV for performance claims.",
    }
    atomic_json(output / "profile_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

