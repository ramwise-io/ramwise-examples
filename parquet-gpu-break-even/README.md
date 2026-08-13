# parquet-gpu-break-even

Companion code for the ramwise.dev Parquet CPU/GPU benchmark post.

The example separates two things that should not be confused:

- `benchmark_parquet.py` is a small correctness-checked benchmark readers can
  run locally. It uses PyArrow and adds cuDF automatically when available.
- `parquet_gpu_study.ipynb` reconstructs the published row-group charts from
  `published_results.csv`, then optionally runs the small local benchmark.

The published 50-million-row study used a stricter harness in the
[`gpu-lab`](https://github.com/ramca-cyber/gpu-lab) repository: isolated CPU
cores, randomized measured trials, full correctness signatures, independent
replications, and continuous CPU/GPU telemetry. This companion intentionally
teaches the core method without duplicating that orchestration layer.

## Run the small example

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python benchmark_parquet.py --rows 1000000 --projection core
pytest -q
```

If cuDF is installed in a compatible NVIDIA/RAPIDS environment, the same
command reports both PyArrow and cuDF. Otherwise it produces a valid CPU-only
result. For the GPU path, use the RAPIDS installation instructions appropriate
for your CUDA driver rather than adding `cudf` to this generic requirements
file.

## Read the result honestly

This is a warm-cache end-to-end read/materialization comparison, not a storage
benchmark. cuDF timing includes the returned GPU frame; the correctness check
converts it to Arrow after the timer stops. The first read is a warmup. Every
measured frame is checked for row count, selected columns, and numeric sums.

Do not compare your laptop's one-million-row result directly with the post's
50-million-row lab result. Small projected reads can favor PyArrow because GPU
startup overhead has not yet been amortized. Locating that crossover is a
useful experiment in its own right.
