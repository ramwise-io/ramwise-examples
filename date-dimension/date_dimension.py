"""A date dimension split into a static core and a dynamic edge.

Companion code for:
https://ramwise.dev/blog/date-dimension-not-static/

The lesson: a date dimension is only "static" if you keep it static on purpose.
Deterministic attributes (year, quarter, ISO week, fiscal period...) are true
forever and safe to persist. Today-relative attributes (is_today,
is_current_month, days_from_today...) depend on when you look and must NOT be
stored -- compute them at the edge, at query time, or your history quietly
changes overnight.

Pure standard library (datetime).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

DYNAMIC_VIEW_SQL = (Path(__file__).parent / "dynamic_view.sql").read_text()


def static_core(start: dt.date, end: dt.date, *, fiscal_start_month: int = 4):
    """Yield one row per day of DETERMINISTIC attributes -- persist these."""
    d = start
    while d <= end:
        quarter = (d.month - 1) // 3 + 1
        fiscal_year = d.year if d.month >= fiscal_start_month else d.year - 1
        fiscal_quarter = ((d.month - fiscal_start_month) % 12) // 3 + 1
        yield {
            "date_key": int(d.strftime("%Y%m%d")),
            "date_value": d.isoformat(),
            "year": d.year,
            "quarter": quarter,
            "month": d.month,
            "day": d.day,
            "iso_week": d.isocalendar()[1],
            "day_name": d.strftime("%A"),
            "is_weekend": d.weekday() >= 5,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter,
        }
        d += dt.timedelta(days=1)


def dynamic_fields(date_value: dt.date, today: dt.date) -> dict:
    """TODAY-relative attributes -- never persist these; compute at query time."""
    return {
        "is_today": date_value == today,
        "is_past": date_value < today,
        "is_current_month": (date_value.year, date_value.month)
        == (today.year, today.month),
        "days_from_today": (date_value - today).days,
    }


if __name__ == "__main__":
    core = list(static_core(dt.date(2026, 1, 1), dt.date(2026, 12, 31)))
    print(f"static core: {len(core)} rows generated (deterministic, persist once)\n")

    sample = core[100]  # some day in April 2026
    print("one static-core row (safe to store forever):")
    for key, value in sample.items():
        print(f"  {key:16} {value}")

    today = dt.date.today()
    dv = dt.date.fromisoformat(sample["date_value"])
    print("\ndynamic fields for that date, computed against today"
          f" ({today.isoformat()}):")
    for key, value in dynamic_fields(dv, today).items():
        print(f"  {key:16} {value}")
    print("\nStore the top block. Compute the bottom block at the edge.")
