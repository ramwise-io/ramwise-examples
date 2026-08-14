# GPU ML Pipeline Boundary

Companion to [I Moved an Entire ML Pipeline to the GPU](https://ramwise.dev/blog/gpu-ml-pipeline-boundary/).

The study compares CPU libraries, unchanged supported APIs under `cuml.accel`,
and native GPU implementations for PCA, KMeans, Logistic Regression, Random
Forest, UMAP, HDBSCAN, and XGBoost. A second matrix measures a complete
preprocessing, feature-engineering, PCA, training, inference, and
materialization pipeline under five CPU/GPU placement strategies.

Open `gpu_ml_pipeline_boundary.ipynb` to inspect the measured row and feature
crossovers, estimator-versus-lifecycle gap, inference behavior, transfer
penalties, and quality controls. The committed notebook contains its outputs;
reading and rerunning its analysis does not require a GPU.

`results/` contains only derived condition medians, fresh-process replication
ranges, and documented quality metrics. Raw per-trial telemetry and private
infrastructure metadata are intentionally excluded. `full-benchmark/` contains
the sanitized runnable harness, publication configs, tests, digest-pinned
container definition, and exact resolved environment.

## Open or rerun the notebook

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter execute gpu_ml_pipeline_boundary.ipynb --inplace
```

The notebook reads the committed derived evidence. It does not rerun the large
GPU benchmark.
