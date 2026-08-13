# Full benchmark

This folder is the sanitized public version of the complete 50-million-row
benchmark harness used by the post. It includes the deterministic generator,
five-engine benchmark, correctness signatures, CPU/GPU telemetry, resumable
matrix runner, report builder, configs, tests, and pinned environment.

It intentionally excludes private infrastructure details and raw per-sample
host telemetry. The derived condition medians used by the article are under
`results/`.

## Build and test

```bash
docker build -t ramwise/parquet-gpu-study:26.08 .
docker run --rm --gpus all \
  --mount type=bind,source="$PWD",target=/benchmark,readonly \
  --workdir /benchmark \
  ramwise/parquet-gpu-study:26.08 \
  python -m pytest -q -p no:cacheprovider
```

## Run a study matrix

The defaults assume Linux logical CPUs `0-7` are the eight cores you want to
dedicate. Inspect your topology with `lscpu -e` and override the set when
needed:

```bash
PARQUET_BENCH_CPUSET=0-7 bash run_matrix.sh confirmation
PARQUET_BENCH_CPUSET=0-7 bash run_matrix.sh rowgroup
```

Generated datasets, raw JSON, checkpoints, and charts go under the ignored
`benchmark-data/` directory. The full run needs an NVIDIA/RAPIDS-compatible
system, at least 20 GB GPU memory, roughly 32 GB free host memory, and at least
100 GB free disk for the confirmation matrix. Review `METHODOLOGY.md` before
comparing results across machines.
