# Published study methodology

This is the sanitized public record for the 50-million-row Parquet CPU/GPU
study. It is sufficient to understand and rerun the experimental design without
publishing the private lab repository or infrastructure details.

## Hardware and software envelope

- NVIDIA RTX PRO 4000 Blackwell SFF, 24 GB
- NVIDIA driver 595.71.05
- CUDA 13.1 and RAPIDS 26.08
- Intel Core Ultra 9 285HX
- eight performance cores exposed to the benchmark process
- Ubuntu 26.04, Python 3.12
- local NVMe data volume

The public dependency definition is in `environment.yml`. Results apply
to this versioned stack; library and driver changes can change absolute timing.

## Dataset

All data was generated deterministically with seed 47. The final studies used
50,000,000 rows split into twenty files of 2,500,000 rows each.

The mixed profile contained integer, floating-point, boolean, and dictionary-
friendly string columns with deterministic null masks. The wide profile added
sixteen floating-point metric columns with deterministic null masks. The
confirmation study used Snappy and Zstandard. The row-group study fixed the
wide profile and Zstandard while varying only rows per group.

## Timed region

The timer covered Parquet read, decompression, and materialization into the
engine's returned frame. cuDF work was synchronized before the timer stopped.
DuckDB returned an Arrow table. Full correctness validation ran outside every
timed region.

Files were warmed into the OS page cache before measurement. This is explicitly
a warm-cache reader/decode/materialization study, not a cold-storage benchmark.
Reported projected MiB/s used total compressed file size as its numerator, so
elapsed seconds—not that derived rate—is the physical projected-read metric.

## Controls

- PyArrow, pandas, Polars, DuckDB, and cuDF were tested.
- Each engine received one warmup and five measured trials per replication.
- Measured engine order was seeded and randomized.
- Every condition had three independent process-level replications.
- The container was restricted to eight performance cores; engines requested
  eight threads.
- CPU process use, selected-core use, whole-host use, and NVIDIA telemetry were
  sampled every 20 ms.
- Every returned frame was checked against a PyArrow reference signature using
  row count, selected columns, null counts, extrema, sums, and distinct counts.

## Final evidence integrity

The isolated confirmation completed 24 cases and 720 timed/warmup trial blocks.
The row-group sweep completed another 24 cases and 720 blocks. Both had zero
correctness failures and zero telemetry sampler errors. Background activity on
the selected cores stayed below 0.36 core in the final studies.

The derived medians used by the post are in `results/published_results.csv`. Raw JSON
and per-sample host telemetry are retained privately because they include
infrastructure metadata and are not necessary to reproduce the article's
tables.

## Limits

The study did not test cold storage, remote object stores, many tiny files,
host-to-device conversion of an existing CPU dataframe, datasets larger than
memory, or out-of-core GPU execution. Three independent replications support the
large effects reported, but not narrow formal confidence intervals for small
differences. A dataset-size crossover sweep remains the most useful follow-up.
