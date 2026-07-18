# preflight-check — companion for *"The Files That Break Your Bulk Load"* on ramwise.dev

A tiny **script** that does a structural pre-flight check on a bulk-load fileset. Point it at a
folder of CSVs and an expected schema, and it names the files that will break or corrupt your
load — with the exact reason for each.

This is the working script from the blog post — about fifteen lines of real logic — plus a
folder of deliberately-broken example files to run it against. It ships both modes (name and
position); it's genuinely all you need for the common case.

## Run it (30 seconds)

```bash
pip install duckdb
python preflight_check.py feed expected.schema name
```

Expected output:

```
7 files: 3 conform, 4 broken, 0 unreadable

GOLDEN (3) — safe to load:
  ✓ orders_2026-01-05.csv
  ✓ orders_2026-01-06.csv
  ✓ orders_2026-01-07.csv

BROKEN (4) — fix or quarantine:
  ✗ orders_2026-01-08.csv
      [ERROR] unexpected column 'discount' (baseline doesn't declare it)
  ✗ orders_2026-01-09.csv
      [ERROR] expected column 'customer_id' is missing
      [ERROR] unexpected column 'account_id' (baseline doesn't declare it)
  ✗ orders_2026-01-10.csv
      [ERROR] expected column 'revenue' is missing
  ✗ orders_2026-01-11.csv
      [ERROR] column 'quantity': expected int, found text
```

## What's in `feed/`

Seven order-feed CSVs — three clean, four deliberately broken, each demonstrating a real
schema-drift case from the post:

| File | What's wrong |
|---|---|
| `orders_2026-01-05/06/07.csv` | nothing — the golden shape |
| `orders_2026-01-08.csv` | an extra column (`discount`) appeared |
| `orders_2026-01-09.csv` | `customer_id` was renamed to `account_id` |
| `orders_2026-01-10.csv` | `revenue` was dropped |
| `orders_2026-01-11.csv` | `quantity` became text (`"many"`) |

Open them and poke around — that's the point.

## The baseline is dumb on purpose

`expected.schema` is one column per line, `name,type`. That's the whole format. Generate it
from a `CREATE TABLE` with `awk`, or hand-write it.

```
order_id,int
customer_id,int
sku,text
quantity,int
revenue,decimal
```

Types are coarse classes (`int`, `decimal`, `text`, `bool`, `date`, `timestamp`) — structural
equivalence cares that a column is "int-ish", not `DECIMAL(10,2)`.

## Two modes

- **name mode** (default) — for loads that bind columns by name (Spark name-union, ADF byName,
  pandas concat). Catches missing / extra / renamed / retyped columns as errors.
- **position mode** — for loads that bind by ordinal position (fixed-width, ordinal COPY INTO,
  headerless). Catches count + type-at-position drift; if headers exist, surfaces a name
  disagreement as a **warning** (a possible same-type swap the positional load can't see).

```bash
python preflight_check.py <folder> <baseline> position
```

## What it deliberately does NOT do

- It reports **structure**, never **data quality** — missing/renamed/retyped columns, never
  "this value is invalid."
- It **never guesses at a rename** — it shows the evidence (missing X + unexpected Y) and lets
  you decide.
- It's **honest about the physics limit** — a same-type swap in a headerless positional load
  is undetectable from structure, and it says so rather than faking it.

See the blog post for why those refusals are the actual point.

Free and open source.
