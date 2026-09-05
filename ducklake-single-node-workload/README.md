# DuckLake Single-Node Workload

Companion to **[DuckLake on One Machine](https://ramwise.dev/blog/ducklake-single-node-workload/)**.

This repository preserves the public parts of a complete data-engineering workload built around DuckDB, DuckLake, and marimo. It is a workflow study rather than a one-query engine benchmark.

One root notebook calls eight child notebooks that:

1. generate deterministic Parquet, CSV, and JSONL sources;
2. ingest batch facts and dimensions;
3. prepare, quarantine, and model orders;
4. apply inserts, updates, deletes, and late changes;
5. process semi-structured events and sessions;
6. run quality and reconciliation checks;
7. compare current and historical DuckLake snapshots; and
8. publish measurements, tables, and a chart.

The study ran four profiles from 100,000 base orders and 200,000 events through 50 million base orders and 100 million events. Each profile has five measured runs. All 20 measured runs passed all 16 checks.

## Follow the notebook evidence

Read the four static marimo snapshots in this order:

- [Quick pipeline](https://ramwise.dev/notebooks/ducklake-single-node-workload/quick-pipeline.html) — follows the root workload through all eight notebook steps.
- [Quick results](https://ramwise.dev/notebooks/ducklake-single-node-workload/quick-results.html) — presents measurements and correctness evidence for the smallest profile.
- [Large pipeline](https://ramwise.dev/notebooks/ducklake-single-node-workload/large-pipeline.html) — follows the same workload at the largest measured scale.
- [Large results](https://ramwise.dev/notebooks/ducklake-single-node-workload/large-results.html) — presents the large-run timings, resource envelope, and final checks.

Those links open rendered notebooks on ramwise.dev. Downloadable copies are kept under [`exports/`](exports/) as retained artifacts; GitHub displays their HTML source rather than rendering it. The hosted pages use the same files with pinned, self-hosted marimo assets and do not execute the workload again.

The publication figures are under [`charts/`](charts/). Derived CSV and JSON evidence is under [`results/`](results/). The exact marimo source used by the measured job is under [`notebooks/`](notebooks/).

## About the notebook source

The notebooks are preserved as measured, not rewritten into a second benchmark after the fact. Most of their work is ordinary DuckDB SQL against a DuckLake catalog. A few small helper calls came from the filesystem-backed job runner used for the experiment: opening the project lake, calling child notebooks, displaying output, and retaining result files.

Those helpers are described in [`notebooks/README.md`](notebooks/README.md). The source is useful for inspecting the workload and adapting its SQL, but this folder is not advertised as a standalone Python package or a turnkey benchmark command.

## Evidence boundary

The committed result files contain the run index, individual step measurements, profile summaries, and step summaries used by the article. The full retained run bundles include host-specific operational evidence and remain in the research record.

Generated source data, the 61 GiB historical lake, package environments, caches, and credentials are intentionally absent. They are unnecessary for checking the reported aggregates and would make this a poor public example.

See [`METHODOLOGY.md`](METHODOLOGY.md) for the environment, timing boundaries, workload sizes, and limitations.

## Result in one sentence

On the documented high-end Windows workstation, the largest workflow took roughly five minutes and reached 3.33 GiB median sampled peak RSS; repeated snapshots made storage housekeeping the part that could not be ignored.
