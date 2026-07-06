"""Tests for semantic_sql. Run with `python test_semantic_sql.py` or `pytest`."""
from __future__ import annotations

from semantic_sql import ask, schema_map, setup_db, verify

SCHEMA = schema_map()


def test_gate_accepts_valid_select():
    ok, _ = verify(
        "SELECT customers.ctry FROM customers WHERE customers.seg = 'business'", SCHEMA
    )
    assert ok


def test_gate_rejects_writes():
    for bad in [
        "UPDATE orders SET amt = 0",
        "DROP TABLE customers",
        "SELECT 1; DELETE FROM orders",
    ]:
        ok, _ = verify(bad, SCHEMA)
        assert not ok, bad


def test_gate_rejects_unknown_table():
    ok, reason = verify("SELECT patients.name FROM patients", SCHEMA)
    assert not ok and "unknown table" in reason


def test_gate_catches_hallucinated_column():
    ok, reason = verify("SELECT orders.revenue FROM orders", SCHEMA)
    assert not ok and "hallucinated column" in reason


def test_valid_question_runs_and_returns_revenue():
    _, rows = ask(setup_db(), "total revenue by country for paid orders")
    result = dict(rows)
    # paid orders: US = 100 + 300 = 400, IN = 250
    assert result == {"US": 400.0, "IN": 250.0}


def test_hallucinated_draft_is_blocked_before_running():
    blocked = False
    try:
        ask(setup_db(), "revenue by country but hallucinate a column")
    except PermissionError:
        blocked = True
    assert blocked


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
