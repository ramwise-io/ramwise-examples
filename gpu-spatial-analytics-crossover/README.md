# GPU Spatial Analytics Crossover

Companion to [100 Million Points on One GPU](https://ramwise.dev/blog/gpu-spatial-analytics-crossover/),
the Ramwise field note on CPU/GPU spatial analytics boundaries.

This parent folder contains an output-complete notebook, sanitized derived
evidence, and the full reproducible benchmark harness. The
study separates spatial index construction, bounding-box candidate generation,
exact geometry refinement, and result materialization across point scale,
feature count, polygon complexity, and selectivity.

The reader [`gpu_spatial_analytics_crossover.ipynb`](gpu_spatial_analytics_crossover.ipynb)
does not need a GPU. It reads committed CSVs derived from three fresh-process
confirmation replications. The separate
[`full-benchmark/`](full-benchmark/) directory needs a supported NVIDIA GPU,
driver, and NVIDIA Container Toolkit.

## Important software boundary

NVIDIA lists cuSpatial as an inactive RAPIDS project; 25.04 is its final stable
release. This experiment uses NVIDIA's signed RAPIDS 25.04 CUDA 12.8 image by
immutable digest and evaluates that historical release on a current RTX PRO
4000 Blackwell SFF host. It is not an endorsement of an actively advancing
library.

## Evidence boundary

Public CSVs contain 33 derived confirmation medians, replication ranges,
candidate counts, memory readings, crossover calculations, and sanitized
exclusions. JSON controls record quadtree coverage and nearest-line numerical
boundaries. Raw per-trial JSON, host paths, and internal
orchestration state remain in the private experiment workspace. The results
README records the exact timing-matrix and derivation revisions.

## Open or rerun the notebook

The committed notebook is already output-complete. Open it directly, or rerun
the analysis against the committed derived evidence:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter execute gpu_spatial_analytics_crossover.ipynb --inplace
```
