# DuckLake Single-Node Workload Lab

Companion to **[I Ran a DuckLake Workspace From 200K to 100M Events](https://ramwise.dev/blog/ducklake-single-node-workload/)**.

This is the public evidence companion for a normal data-engineering workflow built from DuckLake, DuckDB, marimo, and Anatini. It is deliberately not a one-query engine microbenchmark.

One root notebook calls eight child notebooks that:

1. generate deterministic Parquet, CSV, and JSONL sources;
2. ingest batch facts and dimensions;
3. prepare, quarantine, and model orders;
4. apply inserts, updates, deletes, and late changes;
5. process semi-structured events and sessions;
6. run quality and reconciliation checks;
7. compare current and historical DuckLake snapshots; and
8. publish measurements, tables, and a chart.

The locked study ran four profiles from 100,000 base orders / 200,000 events through 50 million base orders / 100 million events. Each profile has five measured Runs. All 20 measured Runs succeeded and passed all 16 checks.

## Start with the output

The static marimo exports contain the code and the output developers saw in the retained Runs:

- [`exports/quick-pipeline.html`](exports/quick-pipeline.html) — median quick pipeline Run;
- [`exports/quick-results.html`](exports/quick-results.html) — quick results notebook;
- [`exports/large-pipeline.html`](exports/large-pipeline.html) — median large pipeline Run;
- [`exports/large-results.html`](exports/large-results.html) — large results notebook.

Download and open an HTML file locally. It is self-contained and does not execute the workload again.

The four publication charts are under [`charts/`](charts/). Sanitized derived evidence is under [`results/`](results/). The exact project source—manifest, Job, marimo notebooks, SQL, test, and method notes—is under [`project/`](project/).

## What is and is not public here

The committed results are derived summaries: the Run index, all step measurements, profile summaries, and step summaries. The full retained raw Run bundles contain host-specific operational evidence and remain in the research record; they are not required to inspect or regenerate the public charts.

The project snapshot excludes generated source data, the DuckLake catalog/Parquet lake, retained Runs, caches, package environments, and secrets. Those are workspace/runtime state, not teaching source.

## Reproduce the analysis

The analysis and charts were generated mechanically from the locked raw Runs. The public CSVs are sufficient to inspect every reported aggregate. See [`METHODOLOGY.md`](METHODOLOGY.md) for the environment, measurement boundary, profile sizes, and limitations.

To rerun the full workload, import the project source into a compatible Anatini workspace, install the documented Python dependencies, and run the saved Job. The source uses Anatini's notebook workflow helpers and project-local DuckLake connection, so it is not presented as a standalone pip package.

## Result in one sentence

On the documented high-end Windows workstation, the largest workflow remained a roughly five-minute retained Run with 3.33 GiB median sampled peak RSS—but repeated snapshots made storage maintenance a real part of the design.
