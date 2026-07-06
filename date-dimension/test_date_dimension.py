"""Tests for date_dimension. Run with `python test_date_dimension.py` or `pytest`."""
from __future__ import annotations

import datetime as dt

from date_dimension import dynamic_fields, static_core


def _by_value(rows):
    return {r["date_value"]: r for r in rows}


def test_static_core_is_deterministic():
    a = list(static_core(dt.date(2026, 1, 1), dt.date(2026, 3, 31)))
    b = list(static_core(dt.date(2026, 1, 1), dt.date(2026, 3, 31)))
    assert a == b  # same input -> same rows, forever


def test_row_count_matches_days():
    rows = list(static_core(dt.date(2026, 1, 1), dt.date(2026, 1, 31)))
    assert len(rows) == 31


def test_known_attributes_and_fiscal_april_start():
    rows = _by_value(static_core(dt.date(2026, 1, 1), dt.date(2026, 12, 31)))
    may = rows["2026-05-15"]
    assert (may["year"], may["quarter"], may["month"]) == (2026, 2, 5)
    # April-start fiscal year: May 2026 is FY2026, fiscal Q1
    assert (may["fiscal_year"], may["fiscal_quarter"]) == (2026, 1)
    mar = rows["2026-03-15"]
    # March 2026 falls in the PRIOR fiscal year, fiscal Q4
    assert (mar["fiscal_year"], mar["fiscal_quarter"]) == (2025, 4)


def test_dynamic_fields_depend_on_today():
    today = dt.date(2026, 6, 15)
    assert dynamic_fields(dt.date(2026, 6, 15), today)["is_today"] is True
    assert dynamic_fields(dt.date(2026, 6, 14), today)["is_past"] is True
    assert dynamic_fields(dt.date(2026, 6, 20), today)["days_from_today"] == 5
    assert dynamic_fields(dt.date(2026, 6, 1), today)["is_current_month"] is True
    assert dynamic_fields(dt.date(2026, 5, 31), today)["is_current_month"] is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
