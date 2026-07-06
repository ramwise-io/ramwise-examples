"""Incremental load patterns, made runnable on SQLite.

Companion code for:
https://ramwise.dev/blog/incremental-load-patterns/

The post's thesis: "incremental load" is not one technique but a family, and
you should pick the weakest one that still guarantees correctness for how your
source behaves. This module implements the common patterns against a tiny
SQLite source/target so you can watch them work -- and, importantly, watch the
naive timestamp watermark silently LOSE rows, then watch the composite
`(timestamp, id)` watermark fix it.

Pure standard library (sqlite3). No dependencies.

Run:  python incremental.py
"""
from __future__ import annotations

import sqlite3


def make_source() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, amount REAL, modified_at TEXT)"
    )
    db.executemany(
        "INSERT INTO orders (id, amount, modified_at) VALUES (?, ?, ?)",
        [
            (1, 10.0, "2026-01-01T09:00:00"),
            (2, 20.0, "2026-01-01T09:00:00"),  # same timestamp as 1, 3, 4
            (3, 30.0, "2026-01-01T09:00:00"),  # (one transaction wrote these)
            (4, 40.0, "2026-01-01T09:00:00"),
            (5, 50.0, "2026-01-02T11:30:00"),
        ],
    )
    db.commit()
    return db


# --- Pattern 3: naive timestamp watermark (has the tie bug) -----------------
def load_naive_watermark(src, last_ts, *, batch_limit):
    """Pull rows strictly newer than the saved timestamp.

    `batch_limit` simulates a run that stops partway through -- e.g. a crash, a
    timeout, or a page boundary -- which is exactly when the tie bug bites.
    """
    rows = src.execute(
        "SELECT id, amount, modified_at FROM orders "
        "WHERE modified_at > ? ORDER BY modified_at LIMIT ?",
        (last_ts, batch_limit),
    ).fetchall()
    new_watermark = max((r[2] for r in rows), default=last_ts)
    return rows, new_watermark


# --- Pattern 3, fixed: composite (timestamp, id) keyset watermark -----------
def load_composite_watermark(src, last_ts, last_id, *, batch_limit):
    rows = src.execute(
        "SELECT id, amount, modified_at FROM orders "
        "WHERE modified_at > ? OR (modified_at = ? AND id > ?) "
        "ORDER BY modified_at, id LIMIT ?",
        (last_ts, last_ts, last_id, batch_limit),
    ).fetchall()
    if rows:
        last = rows[-1]
        return rows, last[2], last[0]
    return rows, last_ts, last_id


def _drain_naive(src, batch_limit):
    """Run the naive loader repeatedly until it stops returning rows."""
    seen, ts = [], ""
    while True:
        rows, ts = load_naive_watermark(src, ts, batch_limit=batch_limit)
        if not rows:
            break
        seen.extend(r[0] for r in rows)
    return seen


def _drain_composite(src, batch_limit):
    seen, ts, rid = [], "", 0
    while True:
        rows, ts, rid = load_composite_watermark(src, ts, rid, batch_limit=batch_limit)
        if not rows:
            break
        seen.extend(r[0] for r in rows)
    return seen


if __name__ == "__main__":
    all_ids = [1, 2, 3, 4, 5]

    # batch_limit=2 forces the run to stop in the middle of the tie group
    # (ids 1..4 share one timestamp).
    naive = _drain_naive(make_source(), batch_limit=2)
    composite = _drain_composite(make_source(), batch_limit=2)

    print("source rows:            ", all_ids)
    print("naive watermark loaded: ", sorted(set(naive)),
          "  <- dropped", sorted(set(all_ids) - set(naive)), "silently")
    print("composite watermark:    ", sorted(set(composite)),
          "  <- all rows, no loss")
    print()
    print("The naive watermark saves max(modified_at) after a partial batch, then")
    print("asks for `> that timestamp` -- so the tied rows it never reached are")
    print("excluded forever. The composite `(ts, id)` keyset never skips them.")
