"""Split a private cuVS summary into substantiated public results and controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    published, unconfirmed = [], []
    for row in rows:
        if row["algorithm"] == "brute_force" or row["confirmed_target_recalls"].strip():
            published.append(row)
        else:
            unconfirmed.append(row)
    return published, unconfirmed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.summary)
    if not rows:
        raise ValueError("summary has no rows")
    published, controls = split(rows)
    fieldnames = list(rows[0])
    write_rows(args.output_dir / "published_results.csv", published, fieldnames)
    write_rows(args.output_dir / "unconfirmed_controls.csv", controls, fieldnames)
    print(json.dumps({"published": len(published), "unconfirmed_controls": len(controls)}, indent=2))


if __name__ == "__main__":
    main()
