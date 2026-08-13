"""Select the fastest measured search parameters meeting recall targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.vector_search_cuvs.common import atomic_json, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--targets", default="0.80,0.90,0.95,0.99")
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select(result: dict, targets: list[float], margin: float = 0.0) -> list[dict]:
    selected = []
    batches = sorted({row["batch_size"] for row in result["searches"]})
    for batch in batches:
        rows = [row for row in result["searches"] if row["batch_size"] == batch]
        for target in targets:
            threshold = min(1.0, target + margin)
            eligible = [row for row in rows if row["recall_at_k"] >= threshold]
            if eligible:
                best = min(eligible, key=lambda row: row["latency_ms_per_batch"])
                selected.append({
                    "algorithm": result["benchmark"]["algorithm"],
                    "batch_size": batch,
                    "dataset_id": result["dataset"]["dataset_id"],
                    "dimensions": result["dataset"]["config"]["dimensions"],
                    "params": best["params"],
                    "recall_at_k": best["recall_at_k"],
                    "rows": result["dataset"]["config"]["rows"],
                    "target_recall": target,
                    "tuning_threshold": threshold,
                    "tuning_latency_ms_per_batch": best["latency_ms_per_batch"],
                })
    return selected


def main() -> None:
    args = parse_args()
    targets = [float(value) for value in args.targets.split(",")]
    rows = []
    for path in args.results:
        rows.extend(select(read_json(path), targets, args.margin))
    atomic_json(args.output, {"schema_version": 1, "selections": rows})
    print(json.dumps({"output": str(args.output), "selections": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
