# sqlite-vs-jsonl-local-first — companion for *"The Row Count Was the Wrong Way to Choose Between SQLite and JSONL"* on ramwise.dev

A standard-library-only benchmark that compares **plain, unindexed JSON Lines** with **SQLite** for
the same four-field records, across four operations: random-ID lookup, one-category aggregate,
single-record update, and a 1,000-record append batch.

It demonstrates one claim: for this schema and these access paths, the choice between the two
tracks **which operations the application needs**, not a row-count threshold. It does not show that
SQLite is faster at everything (append is the counterexample), and it says nothing about concurrency,
crash recovery, browsers, compressed JSONL, or JSONL with a sidecar index.

This is the harness that produced the published numbers, with two changes: it writes to `./output`
instead of a research-repo path, and it accepts `--sizes` for a quick check. The measurement logic is
unchanged. Nothing was sanitised or omitted from the run in [`canonical-run/`](canonical-run/).

## Run it

```bash
python benchmark.py --sizes 10000        # about a second; checks the script on your machine
python benchmark.py                      # the published protocol: 10k, 100k, 500k rows
```

Nothing to install (I checked the quick run on Python 3.14; the published run used 3.13.5). The full run takes a few minutes, most of it the 500,000-row
JSONL rewrites. Results land in `output/derived/benchmark-summary.csv` (medians) and
`output/raw/benchmark-timings.csv` (every repetition).

## What to expect

- **Byte sizes are deterministic.** At 10,000 rows you should see `jsonl_initial_bytes` 667688 and
  `sqlite_initial_bytes` 466944, matching [`canonical-run/`](canonical-run/benchmark-summary.csv). If they
  differ, the SQLite or Python version is producing a different file layout.
- **Timings are not.** The shape should hold on most machines: JSONL lookup, aggregate and update
  grow with file size, while SQLite's stay small. The exact milliseconds and ratios will not match,
  and the append comparison may flip. Page cache, filesystem, and `fsync` cost all move them.
- The script asserts that both stores return the same lookup, aggregate and post-update values. A
  mismatch aborts the run instead of recording a time.

## What the benchmark does

| | JSONL | SQLite |
|---|---|---|
| Format | UTF-8, one compact JSON object per line, no index | `records(id INTEGER PRIMARY KEY, category, value, name)` plus an index on `category` |
| Lookup | Parse lines from the top until the ID matches | Primary-key lookup |
| Aggregate | Parse every line, count and sum category `c07` | Indexed `COUNT(*)`, `SUM(value)` |
| Update | Rewrite to a temp file, `fsync`, `os.replace` | `UPDATE` and commit |
| Append (1,000 records) | Write at EOF, one `flush` + `fsync` | One insert batch, one commit |
| Durability | `fsync` on every write | `journal_mode=WAL`, `synchronous=FULL` |

Records are generated deterministically (`category = "c" + i mod 20`, `value = i*37 mod 100000`).
Repetitions per size: 7 lookups, 3 aggregates, 5 updates, 9 append batches. Reported values are
medians, and every repetition is in the raw CSV.

JSONL lookup stops at the match, so its latency depends on where the random target falls in the file.
That is why a handful of lookups per size can vary widely in the raw timings.

## The canonical run

[`canonical-run/`](canonical-run/) is the locked run behind the post: Python 3.13.5, SQLite 3.46.1,
Linux 6.18.35 on an AMD EPYC 9V74 host, ext4, run on 2026-08-27. It is a single ephemeral host, no
OS page-cache flush, no cold-disk measurement.

## Limits

- One schema, one SQLite configuration, one JSON implementation, one host.
- JSONL has no index. A sidecar index would be a different, legitimate architecture and is not tested.
- Single writer, single connection. No injected crashes, no concurrent processes.
- The update comparison includes a design choice: JSONL does a whole-file rewrite. An append-only
  change log would make updates cheaper and reads more complicated.
