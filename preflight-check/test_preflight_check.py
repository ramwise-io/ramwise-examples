"""Tests: the sample feed is a fixture — 3 golden, 4 broken, with the exact findings."""
from pathlib import Path
from preflight_check import check_fileset

FEED = str(Path(__file__).parent / "feed")
BASE = str(Path(__file__).parent / "expected.schema")


def test_name_mode_partitions_the_set():
    r = check_fileset(FEED, BASE, "name")
    assert (len(r.golden), len(r.broken), len(r.skipped)) == (3, 4, 0)
    broken = {v.path: [f.reason_code for f in v.findings] for v in r.broken}
    assert broken["orders_2026-01-08.csv"] == ["EXTRA_COLUMN"]
    # a rename is reported as two facts, never guessed as one
    assert set(broken["orders_2026-01-09.csv"]) == {"MISSING_COLUMN", "EXTRA_COLUMN"}
    assert broken["orders_2026-01-10.csv"] == ["MISSING_COLUMN"]
    assert broken["orders_2026-01-11.csv"] == ["TYPE_CHANGE"]


def test_position_mode_warns_but_does_not_fail_on_possible_swap():
    r = check_fileset(FEED, BASE, "position")
    swap = next(v for v in r.broken if v.path == "orders_2026-01-09.csv")
    assert swap.ok  # WARN only — a positional load is unaffected by the name
    assert [f.reason_code for f in swap.findings] == ["POSITIONAL_NAME_MISMATCH"]
