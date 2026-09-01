# DuckLake Workload Lab

This project exercises a complete single-node data-engineering workload through
the ordinary Anatini product path. It is evidence for a bounded engineering
question, not a vendor leaderboard.

## What the Job runs

1. generate deterministic Parquet, CSV, and JSONL source files;
2. ingest them into DuckLake raw tables;
3. quarantine invalid rows and build an analytical model;
4. apply inserts, updates, and deletes;
5. process nested events into sessions;
6. execute quality contracts;
7. query the pre-incremental snapshot;
8. retain CSV, JSON, SVG, logs, notebook HTML, and DuckLake measurements.

## Profiles

| Profile | Orders | Changes | Events |
|---|---:|---:|---:|
| quick | 100,000 | 5,000 | 200,000 |
| small | 1,000,000 | 50,000 | 2,000,000 |
| medium | 10,000,000 | 500,000 | 20,000,000 |
| large | 50,000,000 | 2,500,000 | 100,000,000 |

Run profiles in ascending order. Large is conditional on medium staying inside
the documented resource and time stop conditions.

## Interpretation

Exact timings are specific to the recorded Windows host, versions, Job profile,
and warm-cache policy. The useful result is the workload envelope and the point
where time, memory, storage, or operability stops being comfortable.
