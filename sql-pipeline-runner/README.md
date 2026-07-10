# sql-pipeline-runner

Runnable example behind **[Keep the Runner Dumb](https://ramwise.dev/blog/keep-the-runner-dumb/)**.
Pure standard library (`sqlite3`), no dependencies.

A data pipeline can be one SQL file whose steps run top to bottom, with
data-quality **assertions as first-class steps** that halt the run *before* bad
data is written. The runner stays deliberately dumb: it doesn't parse your SQL,
hold state, or schedule anything. It compiles the directives to a plan, runs the
SQL on an in-memory database, gates on the assertions, emits a report, and exits
with a code that says *who to wake up*.

A pipeline is just directive-annotated SQL ([`example.sql`](example.sql)):

```sql
--@pipeline load_orders

--@step clean
CREATE TABLE out AS SELECT id, amount FROM raw WHERE amount > 0

--@assert no_null_id EXPECT_NO_ROWS
SELECT * FROM out WHERE id IS NULL
```

```bash
python pipeline_runner.py
```
```
clean source -> passes, exports:
  status=success  exit=0
  output rows: [{'id': 1, 'amount': 10.0}, {'id': 2, 'amount': 20.0}]

dirty source -> assert halts BEFORE export (exit 10):
  status=assertion_failed  exit=10
  assert no_null_id: failed  evidence=[{'id': None, 'amount': 15.0}]
  output rows: []
```

The design in miniature:

- **Assertions gate the run.** A null id trips `no_null_id` *before* the export,
  so the bad data never lands — unlike a test suite that runs afterward on data
  you already wrote.
- **Exit codes are a contract.** `10` = the data arrived dirty (page the source
  owner); `20` = the script is broken (page the pipeline owner). Most pipelines
  blur both into a generic non-zero.
- **No state.** Work happens in an in-memory database that dies with the process,
  so a failed run corrupts nothing and "re-run from the top" is the whole resume
  story.
- **It won't parse your SQL.** The text between directives is handed straight to
  the engine.

The real tool ([pktl](https://ramwise.dev/blog/keep-the-runner-dumb/)) runs on
DuckDB; this teaching version uses stdlib SQLite so it runs with nothing
installed. The pattern is engine-agnostic.

## Run the tests

```bash
python test_pipeline_runner.py   # also works under: pytest
```

The tests assert that clean data passes and exports, that a null id halts the
run with exit 10 and emits nothing, that a broken SQL step exits 20, and that
`compile` is pure (same text → same plan).
