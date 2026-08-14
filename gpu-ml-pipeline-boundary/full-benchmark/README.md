# Full GPU ML pipeline benchmark

This is the sanitized runnable harness behind the article and output-complete
notebook. It contains deterministic dense-numeric generation, estimator and
end-to-end pipeline matrices, quality gates, GPU-dispatch validation,
fresh-process replications, timing, transfer accounting, and telemetry.

The test host used Ubuntu 26.04. The benchmark ran inside the digest-pinned
Ubuntu 24.04 CUDA base image in `Dockerfile`. `conda-linux-64.lock` fixes the
complete RAPIDS, scikit-learn, XGBoost, and Python environment; the three
additional manifold-learning wheels and their hashes are fixed separately.

## Build and test

```bash
docker build -t ramwise/gpu-ml-pipeline-boundary:26.08 .
docker run --rm --network none --gpus all \
  --mount type=bind,source="$PWD",target=/benchmark,readonly \
  --workdir /benchmark \
  ramwise/gpu-ml-pipeline-boundary:26.08
```

## Run the estimator matrix

```bash
mkdir -p benchmark-data
docker run --rm --network none --gpus all --cpuset-cpus=0-7 --shm-size=16g \
  --ulimit=memlock=-1 --ulimit=stack=67108864 \
  --mount type=bind,source="$PWD",target=/benchmark,readonly \
  --mount type=bind,source="$PWD/benchmark-data",target=/data \
  --workdir /benchmark \
  -e GPU_LAB_SOURCE_ID="$(git rev-parse HEAD)" \
  -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=8 \
  ramwise/gpu-ml-pipeline-boundary:26.08 \
  python -m experiments.gpu_ml_pipeline.matrix_runner \
    --config configs/estimators.json
```

Replace the module and config with
`experiments.gpu_ml_pipeline.pipeline_matrix_runner` and
`configs/pipeline.json` for the end-to-end residency matrix. Both full runs are
substantial. If the bundle is outside a Git checkout, replace
`$(git rev-parse HEAD)` with an immutable source label of your own. Read
`METHODOLOGY.md` before comparing results.

The public notebook reads sanitized condition and replication summaries. Raw
JSON, sampled telemetry, private paths, and host identifiers remain private.
