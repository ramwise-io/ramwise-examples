# date-dimension

A date dimension split into a **static core** and a **dynamic edge** — the fix
from **[Your Date Dimension Is Not Static](https://ramwise.dev/blog/date-dimension-not-static/)**.
Pure standard library (`datetime`).

The trap: someone adds `is_current_month` / `days_from_today` columns to the
"static" date dimension and persists them — and now the same historical report
drifts overnight, because those values depend on *today*.

The fix: persist only the deterministic attributes (year, quarter, ISO week,
fiscal period…); compute the today-relative ones at the edge.

## Run it

```bash
python date_dimension.py        # prints a static-core row + its dynamic fields
python test_date_dimension.py   # tests (also works under: pytest)
```

- `static_core(start, end)` yields the rows you **persist once**. Same input →
  same rows, forever (`test_static_core_is_deterministic`).
- `dynamic_fields(date, today)` are the values you **never store** — compute
  them at query time.
- [`dynamic_view.sql`](dynamic_view.sql) is the lakehouse-SQL version: a view
  over the core that re-evaluates `current_date()` on every query, so it needs
  no refresh job — with the UTC-timezone and Direct-Lake caveats noted inline.

The rule: *if a column's value depends on today's date, don't store it in the
core.*
