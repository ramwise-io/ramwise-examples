#!/usr/bin/env python3
"""Reproduce the SQLite vs plain-JSONL local-store benchmark (standard library only).

    python benchmark.py                  # the locked protocol: 10k, 100k, 500k rows
    python benchmark.py --sizes 10000    # quick check of the script on your machine

Outputs go to ./output/raw and ./output/derived by default (never over
./canonical-run, which holds the published Linux run). Pass ``--output-root`` to
write elsewhere. Exact timings are environment-specific; the script also writes
run metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import platform
import random
import sqlite3
import statistics
import tempfile
import time
from datetime import datetime, timezone

SEED = 20260827
SIZES = [10_000, 100_000, 500_000]
LOOKUP_REPS = 7
AGG_REPS = 3
UPDATE_REPS = 5
APPEND_BATCHES = 9
APPEND_BATCH_SIZE = 1_000
PROTOCOL_LOCK_COMMIT = "8eb28aa42211d373974d3c63a0ec4745a57f946d"

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parent


def record(i: int) -> dict[str, int | str]:
    return {
        "id": i,
        "category": f"c{i % 20:02d}",
        "value": (i * 37) % 100000,
        "name": f"record-{i:07d}",
    }


def encode(row: dict[str, int | str]) -> str:
    return json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n"


def timed(fn):
    start = time.perf_counter_ns()
    result = fn()
    return (time.perf_counter_ns() - start) / 1_000_000, result


def build_jsonl(path: pathlib.Path, n: int) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for i in range(n):
            f.write(encode(record(i)))
        f.flush()
        os.fsync(f.fileno())


def build_sqlite(path: pathlib.Path, n: int) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute(
        "CREATE TABLE records("
        "id INTEGER PRIMARY KEY,"
        "category TEXT NOT NULL,"
        "value INTEGER NOT NULL,"
        "name TEXT NOT NULL)"
    )
    con.execute("CREATE INDEX idx_records_category ON records(category)")

    batch: list[tuple[int, str, int, str]] = []
    for i in range(n):
        row = record(i)
        batch.append((row["id"], row["category"], row["value"], row["name"]))
        if len(batch) == 5000:
            con.executemany("INSERT INTO records VALUES (?,?,?,?)", batch)
            batch.clear()
    if batch:
        con.executemany("INSERT INTO records VALUES (?,?,?,?)", batch)

    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return con


def sqlite_total_bytes(path: pathlib.Path) -> int:
    total = 0
    for suffix in ("", "-wal", "-shm"):
        candidate = pathlib.Path(str(path) + suffix)
        if candidate.exists():
            total += candidate.stat().st_size
    return total


def jsonl_lookup(path: pathlib.Path, target: int) -> dict | None:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] == target:
                return row
    return None


def sqlite_lookup(con: sqlite3.Connection, target: int):
    return con.execute(
        "SELECT id, category, value, name FROM records WHERE id=?", (target,)
    ).fetchone()


def jsonl_aggregate(path: pathlib.Path, category: str) -> tuple[int, int]:
    count = 0
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["category"] == category:
                count += 1
                total += row["value"]
    return count, total


def sqlite_aggregate(con: sqlite3.Connection, category: str):
    return con.execute(
        "SELECT COUNT(*), SUM(value) FROM records WHERE category=?", (category,)
    ).fetchone()


def jsonl_update(path: pathlib.Path, target: int, new_value: int) -> None:
    tmp = path.with_suffix(".tmp")
    found = False
    with path.open("r", encoding="utf-8") as src, tmp.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line in src:
            row = json.loads(line)
            if row["id"] == target:
                row["value"] = new_value
                dst.write(encode(row))
                found = True
            else:
                dst.write(line)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(tmp, path)
    if not found:
        raise KeyError(target)


def sqlite_update(con: sqlite3.Connection, target: int, new_value: int) -> None:
    con.execute("UPDATE records SET value=? WHERE id=?", (new_value, target))
    con.commit()


def jsonl_append(path: pathlib.Path, start_id: int, count: int) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for i in range(start_id, start_id + count):
            f.write(encode(record(i)))
        f.flush()
        os.fsync(f.fileno())


def sqlite_append(con: sqlite3.Connection, start_id: int, count: int) -> None:
    rows = []
    for i in range(start_id, start_id + count):
        row = record(i)
        rows.append((row["id"], row["category"], row["value"], row["name"]))
    con.executemany("INSERT INTO records VALUES (?,?,?,?)", rows)
    con.commit()


def cpu_model() -> str:
    proc = pathlib.Path("/proc/cpuinfo")
    if proc.exists():
        for line in proc.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def write_csv(path: pathlib.Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        help="write raw/ and derived/ beneath this directory (default: ./output)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        help="initial row counts to run; the published run used 10000 100000 500000",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sizes = args.sizes or SIZES
    output_root = args.output_root or (ROOT / "output")
    raw_dir = output_root / "raw"
    derived_dir = output_root / "derived"
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    summaries: list[dict] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = pathlib.Path(temp_dir)

        for n in sizes:
            jsonl_path = temp / f"records-{n}.jsonl"
            sqlite_path = temp / f"records-{n}.sqlite"

            jsonl_build_ms, _ = timed(lambda: build_jsonl(jsonl_path, n))
            sqlite_build_ms, con = timed(lambda: build_sqlite(sqlite_path, n))

            jsonl_initial_bytes = jsonl_path.stat().st_size
            sqlite_initial_bytes = sqlite_total_bytes(sqlite_path)

            assert jsonl_aggregate(jsonl_path, "c07") == tuple(
                sqlite_aggregate(con, "c07")
            )

            rng = random.Random(SEED + n)
            lookup_targets = [rng.randrange(n) for _ in range(LOOKUP_REPS)]
            lookup_jsonl: list[float] = []
            lookup_sqlite: list[float] = []

            for rep, target in enumerate(lookup_targets, 1):
                jsonl_ms, jsonl_row = timed(
                    lambda target=target: jsonl_lookup(jsonl_path, target)
                )
                sqlite_ms, sqlite_row = timed(
                    lambda target=target: sqlite_lookup(con, target)
                )
                assert jsonl_row is not None
                assert jsonl_row["id"] == sqlite_row[0]
                assert jsonl_row["value"] == sqlite_row[2]
                lookup_jsonl.append(jsonl_ms)
                lookup_sqlite.append(sqlite_ms)
                raw_rows.extend(
                    [
                        {
                            "initial_rows": n,
                            "operation": "lookup",
                            "representation": "jsonl",
                            "rep": rep,
                            "target_or_start": target,
                            "elapsed_ms": f"{jsonl_ms:.6f}",
                        },
                        {
                            "initial_rows": n,
                            "operation": "lookup",
                            "representation": "sqlite",
                            "rep": rep,
                            "target_or_start": target,
                            "elapsed_ms": f"{sqlite_ms:.6f}",
                        },
                    ]
                )

            aggregate_jsonl: list[float] = []
            aggregate_sqlite: list[float] = []
            for rep in range(1, AGG_REPS + 1):
                jsonl_ms, jsonl_result = timed(
                    lambda: jsonl_aggregate(jsonl_path, "c07")
                )
                sqlite_ms, sqlite_result = timed(
                    lambda: sqlite_aggregate(con, "c07")
                )
                assert jsonl_result == tuple(sqlite_result)
                aggregate_jsonl.append(jsonl_ms)
                aggregate_sqlite.append(sqlite_ms)
                raw_rows.extend(
                    [
                        {
                            "initial_rows": n,
                            "operation": "aggregate_c07",
                            "representation": "jsonl",
                            "rep": rep,
                            "target_or_start": "c07",
                            "elapsed_ms": f"{jsonl_ms:.6f}",
                        },
                        {
                            "initial_rows": n,
                            "operation": "aggregate_c07",
                            "representation": "sqlite",
                            "rep": rep,
                            "target_or_start": "c07",
                            "elapsed_ms": f"{sqlite_ms:.6f}",
                        },
                    ]
                )

            update_targets = [rng.randrange(n) for _ in range(UPDATE_REPS)]
            update_jsonl: list[float] = []
            update_sqlite: list[float] = []
            for rep, target in enumerate(update_targets, 1):
                new_value = 900000 + rep
                jsonl_ms, _ = timed(
                    lambda target=target, value=new_value: jsonl_update(
                        jsonl_path, target, value
                    )
                )
                sqlite_ms, _ = timed(
                    lambda target=target, value=new_value: sqlite_update(
                        con, target, value
                    )
                )
                assert jsonl_lookup(jsonl_path, target)["value"] == new_value
                assert sqlite_lookup(con, target)[2] == new_value
                update_jsonl.append(jsonl_ms)
                update_sqlite.append(sqlite_ms)
                raw_rows.extend(
                    [
                        {
                            "initial_rows": n,
                            "operation": "update_one",
                            "representation": "jsonl",
                            "rep": rep,
                            "target_or_start": target,
                            "elapsed_ms": f"{jsonl_ms:.6f}",
                        },
                        {
                            "initial_rows": n,
                            "operation": "update_one",
                            "representation": "sqlite",
                            "rep": rep,
                            "target_or_start": target,
                            "elapsed_ms": f"{sqlite_ms:.6f}",
                        },
                    ]
                )

            append_jsonl: list[float] = []
            append_sqlite: list[float] = []
            for rep in range(1, APPEND_BATCHES + 1):
                start_id = n + (rep - 1) * APPEND_BATCH_SIZE
                jsonl_ms, _ = timed(
                    lambda start_id=start_id: jsonl_append(
                        jsonl_path, start_id, APPEND_BATCH_SIZE
                    )
                )
                sqlite_ms, _ = timed(
                    lambda start_id=start_id: sqlite_append(
                        con, start_id, APPEND_BATCH_SIZE
                    )
                )
                append_jsonl.append(jsonl_ms)
                append_sqlite.append(sqlite_ms)
                raw_rows.extend(
                    [
                        {
                            "initial_rows": n,
                            "operation": "append_1000",
                            "representation": "jsonl",
                            "rep": rep,
                            "target_or_start": start_id,
                            "elapsed_ms": f"{jsonl_ms:.6f}",
                        },
                        {
                            "initial_rows": n,
                            "operation": "append_1000",
                            "representation": "sqlite",
                            "rep": rep,
                            "target_or_start": start_id,
                            "elapsed_ms": f"{sqlite_ms:.6f}",
                        },
                    ]
                )

            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            summaries.append(
                {
                    "initial_rows": n,
                    "jsonl_build_ms": f"{jsonl_build_ms:.6f}",
                    "sqlite_build_ms": f"{sqlite_build_ms:.6f}",
                    "jsonl_initial_bytes": jsonl_initial_bytes,
                    "sqlite_initial_bytes": sqlite_initial_bytes,
                    "jsonl_lookup_median_ms": f"{statistics.median(lookup_jsonl):.6f}",
                    "sqlite_lookup_median_ms": f"{statistics.median(lookup_sqlite):.6f}",
                    "jsonl_aggregate_median_ms": f"{statistics.median(aggregate_jsonl):.6f}",
                    "sqlite_aggregate_median_ms": f"{statistics.median(aggregate_sqlite):.6f}",
                    "jsonl_update_median_ms": f"{statistics.median(update_jsonl):.6f}",
                    "sqlite_update_median_ms": f"{statistics.median(update_sqlite):.6f}",
                    "jsonl_append1000_median_ms": f"{statistics.median(append_jsonl):.6f}",
                    "sqlite_append1000_median_ms": f"{statistics.median(append_sqlite):.6f}",
                    "jsonl_final_bytes": jsonl_path.stat().st_size,
                    "sqlite_final_bytes": sqlite_total_bytes(sqlite_path),
                }
            )
            con.close()

    write_csv(
        raw_dir / "benchmark-timings.csv",
        raw_rows,
        [
            "initial_rows",
            "operation",
            "representation",
            "rep",
            "target_or_start",
            "elapsed_ms",
        ],
    )
    write_csv(
        derived_dir / "benchmark-summary.csv",
        summaries,
        list(summaries[0].keys()),
    )

    metadata = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "README.md#what-the-benchmark-does",
        "protocol_lock_commit": PROTOCOL_LOCK_COMMIT,
        "run_class": "local rerun of the published protocol",
        "exploratory_run_excluded": True,
        "seed": SEED,
        "sizes": sizes,
        "lookup_reps": LOOKUP_REPS,
        "aggregate_reps": AGG_REPS,
        "update_reps": UPDATE_REPS,
        "append_batches": APPEND_BATCHES,
        "append_batch_size": APPEND_BATCH_SIZE,
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "filesystem_for_temp_storage": (
            "not automatically detected; record manually for a canonical run"
        ),
        "execution_environment_note": (
            "Exact timings are environment-specific; inspect the platform and rerun "
            "on target hardware/filesystem before generalizing."
        ),
        "sqlite_journal_mode": "WAL",
        "sqlite_synchronous": "FULL",
    }
    (raw_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(derived_dir / "benchmark-summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
