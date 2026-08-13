"""Create lossless, publication-safe summaries from a raw spatial matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


TIMING_STAGES = (
    "ingest_seconds",
    "index_build_seconds",
    "candidate_seconds",
    "exact_seconds",
    "query_seconds",
    "materialize_seconds",
)
TOTAL_STAGES = ("ingest_seconds", "index_build_seconds", "query_seconds", "materialize_seconds")


def seconds(result: dict[str, Any], key: str) -> float:
    return float(result.get(key, 0.0))


def total(result: dict[str, Any]) -> float:
    return sum(seconds(result, key) for key in TOTAL_STAGES)


def break_even_queries(
    cpu: dict[str, Any], gpu: dict[str, Any], include_materialization: bool = False
) -> int | None:
    """Queries after which GPU fixed cost is recovered; None means never."""
    cpu_repeated = seconds(cpu, "query_seconds")
    gpu_repeated = seconds(gpu, "query_seconds")
    if include_materialization:
        cpu_repeated += seconds(cpu, "materialize_seconds")
        gpu_repeated += seconds(gpu, "materialize_seconds")
    per_query_savings = cpu_repeated - gpu_repeated
    if per_query_savings <= 0:
        return None
    gpu_fixed = seconds(gpu, "ingest_seconds") + seconds(gpu, "index_build_seconds")
    cpu_fixed = seconds(cpu, "ingest_seconds") + seconds(cpu, "index_build_seconds")
    return max(0, math.ceil((gpu_fixed - cpu_fixed) / per_query_savings))


def describe(values: Iterable[float], prefix: str) -> dict[str, float]:
    measured = list(values)
    return {
        f"{prefix}_median": statistics.median(measured),
        f"{prefix}_min": min(measured),
        f"{prefix}_max": max(measured),
    }


def case_fields(case: dict[str, Any]) -> dict[str, Any]:
    window = case.get("window")
    return {
        "sweep": case["sweep"],
        "operation": case["operation"],
        "axis": case["axis"],
        "axis_value": case["axis_value"],
        "points": case["points"],
        "features": case.get("features"),
        "vertices": case.get("vertices"),
        "radius_fraction": case.get("radius_fraction"),
        "window_low": window[0] if window else None,
        "window_high": window[1] if window else None,
        "cpu_workers": case.get("cpu_workers", 1),
        "warmups": case["warmups"],
        "trials": case["trials"],
    }


def load_matrix(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((root / "manifest.json").read_text())
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(directory.glob("replication-*.json")):
            if path.name.endswith(".failed.json"):
                continue
            payload = json.loads(path.read_text())
            payload["_case_id"] = directory.name
            successes.append(payload)
        for path in sorted(directory.glob("replication-*.failed.json")):
            payload = json.loads(path.read_text())
            payload["_case_id"] = directory.name
            failures.append(payload)
    return manifest, successes, failures


def confirmed_rows(root: Path) -> list[dict[str, Any]]:
    manifest, successes, failures = load_matrix(root)
    expected = int(manifest["config"]["replications"])
    failed_ids = {row["_case_id"] for row in failures}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for payload in successes:
        grouped.setdefault(payload["_case_id"], []).append(payload)

    output: list[dict[str, Any]] = []
    for case_id, group in sorted(grouped.items()):
        if case_id in failed_ids or len(group) != expected:
            continue
        group.sort(key=lambda row: int(row.get("replication", 0)))
        case = group[0]["case"]
        row: dict[str, Any] = {
            "matrix_id": manifest["matrix_id"],
            "source_commit": manifest["source_commit"],
            "case_id": case_id,
            **case_fields(case),
            "replications": len(group),
        }
        for engine in ("cpu", "gpu"):
            for stage in TIMING_STAGES:
                row.update(describe((seconds(item[engine], stage) for item in group), f"{engine}_{stage}"))
            row.update(describe((total(item[engine]) for item in group), f"{engine}_end_to_end_seconds"))

        cpu_total = row["cpu_end_to_end_seconds_median"]
        gpu_total = row["gpu_end_to_end_seconds_median"]
        cpu_query = row["cpu_query_seconds_median"]
        gpu_query = row["gpu_query_seconds_median"]
        row["gpu_end_to_end_speedup"] = cpu_total / gpu_total
        row["gpu_resident_query_speedup"] = cpu_query / gpu_query

        resident_break_even = [
            break_even_queries(item["cpu"], item["gpu"]) for item in group
        ]
        materialized_break_even = [
            break_even_queries(item["cpu"], item["gpu"], include_materialization=True)
            for item in group
        ]
        row["reused_index_break_even_queries_median"] = (
            statistics.median(value for value in resident_break_even if value is not None)
            if all(value is not None for value in resident_break_even)
            else None
        )
        row["materialized_reuse_break_even_queries_median"] = (
            statistics.median(value for value in materialized_break_even if value is not None)
            if all(value is not None for value in materialized_break_even)
            else None
        )

        first = group[0]
        row["matches"] = first["cpu"]["matches"]
        for engine in ("cpu", "gpu"):
            candidate_values = [item[engine].get("candidate_pairs") for item in group]
            row[f"{engine}_candidate_pairs"] = (
                statistics.median(candidate_values) if all(value is not None for value in candidate_values) else None
            )
            row[f"{engine}_rss_bytes_max"] = max(
                item[engine]["memory"]["rss_bytes"] for item in group
            )
            row[f"{engine}_gpu_used_bytes_max"] = max(
                item[engine]["memory"]["gpu_used_bytes"] for item in group
            )
        correctness = [item["correctness"] for item in group]
        row["maximum_absolute_distance_error"] = max(
            (float(item.get("maximum_absolute_distance_error", 0.0)) for item in correctness),
            default=0.0,
        )
        row["maximum_relative_distance_error"] = max(
            (float(item.get("maximum_relative_distance_error", 0.0)) for item in correctness),
            default=0.0,
        )
        output.append(row)
    return output


def excluded_rows(root: Path) -> list[dict[str, Any]]:
    manifest, successes, failures = load_matrix(root)
    expected = int(manifest["config"]["replications"])
    by_id: dict[str, dict[str, Any]] = {}
    for payload in successes:
        entry = by_id.setdefault(
            payload["_case_id"],
            {"case": payload["case"], "successes": 0, "failures": []},
        )
        entry["successes"] += 1
    for payload in failures:
        entry = by_id.setdefault(
            payload["_case_id"],
            {"case": payload["case"], "successes": 0, "failures": []},
        )
        entry["failures"].append(payload)

    output: list[dict[str, Any]] = []
    for case_id, entry in sorted(by_id.items()):
        if entry["successes"] == expected and not entry["failures"]:
            continue
        returncodes = sorted({int(item["returncode"]) for item in entry["failures"]})
        messages = []
        for item in entry["failures"]:
            last = item.get("stderr", "").strip().splitlines()
            messages.append(last[-1] if last else "process exited without stderr")
        output.append(
            {
                "matrix_id": manifest["matrix_id"],
                "source_commit": manifest["source_commit"],
                "case_id": case_id,
                **case_fields(entry["case"]),
                "expected_replications": expected,
                "successful_replications": entry["successes"],
                "failed_replications": len(entry["failures"]),
                "returncodes": ";".join(map(str, returncodes)),
                "failure_summary": " | ".join(sorted(set(messages))),
                "publication_status": "excluded_from_performance_claims",
            }
        )
    return output


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--excluded-csv", type=Path)
    args = parser.parse_args()
    values = confirmed_rows(args.root)
    excluded = excluded_rows(args.root)
    if args.csv:
        write_csv(args.csv, values)
    if args.excluded_csv:
        write_csv(args.excluded_csv, excluded)
    for row in values:
        print(
            f"{row['sweep']:24} {str(row['axis_value']):>10} "
            f"CPU={row['cpu_end_to_end_seconds_median']:.6f}s "
            f"GPU={row['gpu_end_to_end_seconds_median']:.6f}s "
            f"speedup={row['gpu_end_to_end_speedup']:.2f}x "
            f"matches={row['matches']}"
        )
    if excluded:
        print(f"excluded {len(excluded)} incomplete or failed cases")


if __name__ == "__main__":
    main()
