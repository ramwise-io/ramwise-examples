# cuVS Vector Search Break-Even

Companion to [How Many Queries Pay for a Vector Index?](https://ramwise.dev/blog/gpu-vector-search-break-even/),
the Ramwise field note on exact and approximate GPU vector search.

This folder contains an output-complete notebook that analyzes exact cuVS
brute force, IVF-Flat, two IVF-PQ compression levels, and CAGRA across corpus
size, vector dimensionality, query batch size, and recall target. The main
question is economic rather than just algorithmic: how many queries must an
ANN index serve before its build cost is recovered by faster search?

The committed [`cuvs_vector_search_break_even.ipynb`](cuvs_vector_search_break_even.ipynb)
reads a sanitized table of confirmed medians. It does not need a GPU merely to
display or regenerate the analysis. The separate
[`full-benchmark/`](full-benchmark/) directory contains the GPU harness and
exact environment used for the study.

## Evidence boundary

The public result table contains derived confirmation medians, replication
minima, index and VRAM sizes, and break-even calculations. Raw per-trial NVML
telemetry, machine paths, and internal orchestration state remain in the
private experiment workspace. [`results/README.md`](results/README.md) records
the matrix ID, measurement and derivation revisions, and evidence split.

## Open or rerun the notebook

The committed notebook is already output-complete. Open it directly, or rerun
the analysis against the committed derived evidence:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter execute cuvs_vector_search_break_even.ipynb --inplace
```

The benchmark itself requires a supported NVIDIA GPU, driver, and NVIDIA
Container Toolkit. Re-executing the analysis notebook does not rerun it.
