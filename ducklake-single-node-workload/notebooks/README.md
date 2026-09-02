# Notebook map

These are the marimo Python files used by the measured workload.

| Notebook | Work performed |
|---|---|
| `run_workload.py` | Calls the eight steps in order and reports the complete result. |
| `step_01_generate_sources.py` | Generates deterministic Parquet, CSV, and JSONL inputs. |
| `step_02_ingest_batch.py` | Loads order, customer, and product data into DuckLake. |
| `step_03_prepare_model.py` | Standardizes orders, quarantines bad rows, and builds analytical tables. |
| `step_04_apply_incremental.py` | Applies inserts, updates, deletes, and late-arriving changes. |
| `step_05_process_events.py` | Parses nested JSON events, quarantines invalid records, and builds sessions. |
| `step_06_validate_quality.py` | Runs count, key, dimension, quarantine, and revenue checks. |
| `step_07_query_history.py` | Compares the pre-change snapshot with current state. |
| `step_08_publish_results.py` | Saves the measurement CSV, JSON summary, and per-run SVG. |

## Small runner helpers

The measured files retain a few imports from the job runner used for the experiment:

- `lake.connect()` opens the current project's DuckLake catalog.
- `workflow.run_step()` executes another marimo app and records it as a child step.
- `notebook_output` wraps normal marimo Markdown and table display.
- `run_artifacts.path()` returns a directory belonging to the current historical run.
- `anatini_projects...` is the runner's import namespace for project notebooks.

These calls are orchestration and display glue. The data generation, SQL, checks, and timing logic are visible in the files themselves. The HTML exports preserve the resulting code and output without requiring that runner.
