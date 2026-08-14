# Full custom CuPy/CUDA benchmark

This is the sanitized runnable harness behind the article and output-complete
notebook. It covers launch calibration, pageable/pinned transfers, a fused
numeric scoring workload, intentionally different layouts, block-size sweeps,
correctness checks, fresh-process replications, and optional Nsight Compute
captures.

The test host used Ubuntu 26.04. The benchmark container uses the
digest-pinned Ubuntu 24.04 CUDA image in `Dockerfile` and the exact explicit
Conda specification in `conda-linux-64.lock`. Read `METHODOLOGY.md` before
comparing results.

## Build and test

```bash
docker build -t ramwise/custom-cuda-study:26.08 .
docker run --rm --network none --gpus all \
  --mount type=bind,source="$PWD",target=/benchmark,readonly \
  --workdir /benchmark \
  ramwise/custom-cuda-study:26.08
```

## Run the publication matrix

```bash
mkdir -p benchmark-data/{benchmarks,cache}
docker run --rm --network none --gpus all --cpuset-cpus=0-7 \
  --mount type=bind,source="$PWD",target=/benchmark,readonly \
  --mount type=bind,source="$PWD/benchmark-data",target=/data \
  --workdir /benchmark \
  -e GPU_LAB_SOURCE_ID="$(git rev-parse HEAD)" \
  -e GPU_LAB_HOSTNAME="your-host" -e GPU_LAB_HOST_OS="your-host-os" \
  -e GPU_LAB_CPUSET=0-7 -e GPU_LAB_IMAGE=ramwise/custom-cuda-study:26.08 \
  -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=8 \
  ramwise/custom-cuda-study:26.08 \
  python -m experiments.custom_cuda.matrix_runner --config configs/full.json
```

The full run is substantial. If this directory is outside a Git checkout,
replace `$(git rev-parse HEAD)` with an immutable source label of your own.

## Optional profiling

Build the separate profiler layer:

```bash
docker build -f Dockerfile.profiler -t ramwise/custom-cuda-profiler:2025.4.1 .
```

Use `experiments.custom_cuda.profile_target` under Nsight Compute and then
derive the selected table with `experiments.custom_cuda.summarize_profiles`.
The exact sections used by the study were `LaunchStats`, `Occupancy`,
`MemoryWorkloadAnalysis`, and `SpeedOfLight`, with one warmup launch skipped.
On hosts that restrict GPU performance counters, the short-lived profiler
container may require root plus `SYS_ADMIN`. Do not grant that capability to
the ordinary benchmark container.

The public notebook reads the sanitized condition tables. Raw JSON, private
paths, logs, and `.ncu-rep` files are intentionally excluded.

