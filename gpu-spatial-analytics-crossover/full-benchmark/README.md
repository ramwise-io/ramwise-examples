# Full GPU spatial benchmark

This is a sanitized copy of the harness used for the published study. The
benchmark host was Ubuntu 26.04; the container is NVIDIA's signed RAPIDS 25.04
CUDA 12.8 image pinned by digest. The extracted `conda-linux-64.lock` records
the exact Linux package set, including Python 3.12.9, cuSpatial/cuDF 25.04,
CuPy 13.4.1, Shapely 2.0.7, and NumPy 2.0.2.

NVIDIA now classifies cuSpatial as an inactive RAPIDS project. The image and
its contents are governed by the NVIDIA Deep Learning Container License.

Build and run the smoke matrix:

```bash
docker build -t spatial-crossover:25.04 .
mkdir -p "$PWD/output"
docker run --rm --network none --gpus all --cpuset-cpus=0-7 \
  --shm-size=16g --ulimit=memlock=-1 --ulimit=stack=67108864 \
  -v "$PWD:/workspace:ro" -v "$PWD/output:/data" -w /workspace \
  -e CUDA_VISIBLE_DEVICES=0 -e GPU_LAB_SOURCE_COMMIT=public-bundle \
  spatial-crossover:25.04 \
  python -m experiments.spatial_analytics.matrix_runner \
  --config experiments/spatial_analytics/matrix_smoke.json \
  --output-root /data/matrices
```

Run the API semantics pilot before starting the full matrix:

```bash
docker run --rm --gpus all \
  spatial-crossover:25.04 \
  python -m experiments.spatial_analytics.api_semantics
```

Use `matrix_full.json` only after the smoke matrix and API semantics pilot pass.
See `METHODOLOGY.md` for exact timing and publication rules.
