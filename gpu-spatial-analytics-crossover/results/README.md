# Published evidence

The performance matrix is `ea50b93f1817b252`, timed from private source commit
`621212fd37cb221cc8322c50cc4f5ea663a34064`. The publication-safe reducer is
private derivation commit `8682dc1`.

- `published_results.csv` has 33 cases for which all three fresh-process
  replications completed and passed correctness checks. It includes medians,
  min/max replication ranges, timing stages, candidate counts, memory samples,
  and index-reuse break-even calculations.
- `excluded_cases.csv` has the three configurations excluded from performance
  claims: two with one Shapely/GEOS process exit and one with three strict
  nearest-distance validation failures.
- `quadtree_diagnostic.json` demonstrates that a 10M-point tree with leaf
  capacity 128 lost point coverage, while capacity 256 accounted for every
  point and reproduced the CPU pair count.
- `nearest_100k_diagnostic.json` and `nearest_1m_diagnostic.json` show the
  verified and failed nearest-line numerical ranges across four candidate-box
  radii.

Raw trials, host paths, and internal orchestration state remain private. Every
public row retains the matrix ID, timing-source commit, case ID, geometry
parameters, replication count, and relevant correctness fields.

The analysis notebook was generated with the benchmark image's `nbformat` and
executed against these committed files with the existing Jupyter-capable
RAPIDS 26.08 lab image. The 25.04 benchmark image has no `ipykernel`; notebook
execution does not import cuSpatial or recompute any timed result.
