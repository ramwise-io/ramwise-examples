# Full Spark RAPIDS fallback benchmark

This is the sanitized runnable harness used for the published companion.
Generated datasets, raw event logs, driver logs, and immutable result JSON stay
outside Git under `/data`.

The Dockerfile pins the CUDA base-image digest and pins Spark 3.5.8 and RAPIDS
Accelerator 26.06.1 artifacts by checksum. `python-freeze.txt` records the
exact resolved Python environment from the validated image;
`requirements.lock` is the exact install set used by the container. The
benchmark requires Linux, Docker with the NVIDIA runtime, an
NVIDIA GPU supported by Spark RAPIDS, and the deterministic analytical-pipeline
Parquet corpus described in `METHODOLOGY.md`.

Build and test:

```bash
docker build -t spark-rapids-fallback .
docker run --rm --gpus all --shm-size=12g \
  -v "$PWD:/workspace/gpu-lab:ro" -v /path/to/gpu-lab-data:/data \
  spark-rapids-fallback python3 -m pytest tests -q
```

Run a matrix from the repository root mounted at `/workspace/gpu-lab`:

```bash
python3 -m experiments.spark_rapids_fallback.matrix_runner \
  --config experiments/spark_rapids_fallback/matrix_scale.json
```

`matrix_bridge.json` is an optional diagnostic control for RAPIDS' accelerated
Python columnar bridge. It is not part of the published timing evidence; see
the methodology for the observed stability limitation.
