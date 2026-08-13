# GPU Data Engineering Boundary

Companion to [The GPU Boundary Was a Query Plan, Not a Row Count](https://ramwise.dev/blog/gpu-data-engineering-boundary/).

This folder is the public, output-complete companion to a controlled benchmark
of four analytical execution paths:

- DuckDB on CPU;
- Polars CPU streaming;
- Polars GPU with CPU fallback disabled; and
- native cuDF, plus an explicitly partitioned cuDF path for the larger-than-VRAM case.

Open [`gpu_data_engineering_boundary.ipynb`](gpu_data_engineering_boundary.ipynb)
to inspect the measured crossover curves, workload winners, codec sensitivity,
larger-than-VRAM behavior, memory evidence, and the conclusions supported by
the study. The committed notebook contains its outputs; it does not require a
GPU merely to read and reproduce the charts.

The public [`results/published_results.csv`](results/published_results.csv)
contains derived condition medians only. Raw per-trial host telemetry and
private infrastructure metadata are intentionally excluded.
[`results/failed_controls.csv`](results/failed_controls.csv) preserves the
single-shot native-cuDF out-of-memory result without publishing private logs.

[`full-benchmark/`](full-benchmark/) contains the sanitized runnable harness,
three study configs, methodology, tests, digest-pinned container definition,
and exact Conda explicit specification.

## Rebuild the notebook output

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python build_notebook.py
jupyter execute gpu_data_engineering_boundary.ipynb --inplace
```

The benchmark itself was run in a pinned Linux GPU container. Re-executing this
notebook reads the committed derived results and regenerates the analysis; it
does not rerun the multi-engine benchmark.
