# Full analytical-pipeline benchmark

This is the sanitized runnable harness behind the article and output-complete
notebook. It contains deterministic data generation, equivalent query
implementations, strict correctness checks, randomized trials, process/GPU
telemetry, resumable matrices, and the exact resolved Linux package set.

The test host used Ubuntu 26.04. The benchmark ran inside the digest-pinned
Ubuntu 24.04 CUDA base image in `Dockerfile`.

## Build and test

```bash
docker build -t ramwise/gpu-data-engineering-boundary:26.08 .
docker run --rm --gpus all \
  --mount type=bind,source="$PWD",target=/benchmark,readonly \
  --workdir /benchmark \
  ramwise/gpu-data-engineering-boundary:26.08
```

## Run a matrix

The runner writes generated inputs and immutable raw output beneath `/data`.
Mount a dedicated data directory there; the full matrices require substantial
disk, RAM, VRAM, and time.

```bash
mkdir -p benchmark-data
docker run --rm --gpus all \
  --mount type=bind,source="$PWD",target=/benchmark,readonly \
  --mount type=bind,source="$PWD/benchmark-data",target=/data \
  --workdir /benchmark \
  -e CUDF_POLARS__EXECUTOR__FALLBACK_MODE=raise \
  ramwise/gpu-data-engineering-boundary:26.08 \
  python -m experiments.analytical_pipeline.matrix_runner \
    --config configs/crossover.json
```

Available configs are `crossover.json`, `codec.json`, and `out_of_core.json`.
The 200-million-row out-of-core matrix also used pinned host memory and a 16
GiB cuDF spill limit; see `METHODOLOGY.md` before comparing runs.

The public notebook reads sanitized derived tables. Raw JSON and host-specific
telemetry remain private because they include infrastructure identifiers.
