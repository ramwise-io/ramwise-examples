# Spark RAPIDS Fallback Boundary

Companion to [Spark RAPIDS Won Until I Split the Plan in Two](https://ramwise.dev/blog/spark-rapids-fallback-boundary/),
a Ramwise study of realistic PySpark ETL with and without the RAPIDS
Accelerator.

Open [`spark_rapids_fallback_boundary.ipynb`](spark_rapids_fallback_boundary.ipynb)
to inspect the scale crossover, the physical-plan fallback anatomy, and the
point where partial acceleration became slower than keeping the complete Spark
plan on CPU. The committed notebook includes all outputs and requires no GPU
to read.

The public evidence retains all 78 process-level condition rows:

- `results/scale_results.csv`: 5 sizes × 2 modes × 3 replications;
- `results/fallback_results.csv`: 8 plan topologies × 2 modes × 3 replications.
- `results/bridge_control_status.csv`: status-only audit for the excluded,
  unstable accelerated-bridge control; no timings are published from it.

Each row is the median of two measured actions after one warmup. The full
study therefore contains 156 measured actions, plus warmups and per-size
correctness references. Raw event logs, verbose driver logs, private host
paths, and per-sample telemetry remain in the internal experiment workspace.

`full-benchmark/` contains the sanitized runnable harness, both exact matrix
configs, an optional accelerated-bridge diagnostic, methodology, tests, and a
checksum-pinned container definition with an exact Python dependency lock.

## Open or rerun the notebook

The committed notebook is already output-complete. Open it directly, or rerun
the analysis against the committed derived evidence:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter execute spark_rapids_fallback_boundary.ipynb --inplace
```

The notebook regenerates analysis from the committed derived evidence. It
does not rerun Spark or require an NVIDIA GPU.
