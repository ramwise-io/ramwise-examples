# Spark RAPIDS Fallback Boundary

This study measures when a mixed CPU/GPU Spark physical plan stops benefiting
from the RAPIDS Accelerator. It does not treat plugin installation or
`spark.rapids.sql.enabled=true` as evidence that a query ran on the GPU.

The benchmark records the post-execution physical plan, Python CPU islands,
row/column transitions, event log, correctness result, wall time, process-tree
CPU/RSS, and NVML telemetry for every measured action.

## Workload

A deterministic wide Parquet fact table is filtered, broadcast-joined to a
100,000-row customer dimension, projected, grouped, and sorted. The same
identity-style vectorized Python UDF is placed progressively earlier in the
pipeline and widened from one to twelve inputs. This changes the location and
volume of the CPU island without changing the result.

RAPIDS can partially accelerate the columnar bridge around `ArrowEvalPythonExec`.
The main fallback matrix explicitly disables that replacement so the UDF is a
visible CPU island with row/column transitions. A separate pilot retains the
accelerated-bridge behavior as a control; the Python function itself is not
described as GPU-executed in either case.

The CPU and GPU modes use the same Spark 3.5.8 image and load the same RAPIDS
26.06.1 CUDA 13 plugin. Only `spark.rapids.sql.enabled` differs. Adaptive query
execution is disabled so plan topology is stable and auditable on this
single-node local-mode laboratory setup.

## Matrices

- `matrix_scale.json`: fully supported native plan, 1M through 50M rows.
- `matrix_fallback.json`: fixed 50M rows, fallback location, width, and two
  separated Python islands.
- `matrix_bridge.json`: current RAPIDS accelerated Python-columnar bridge
  control for selected 50M-row CPU-island plans.

The first bridge-control matrix was excluded from performance reporting: one
initial replication and the 12-column retry terminated the JVM with the same
`libcuda.so.1` SIGSEGV. Completed control cases remain in the private audit
store, and retry attempts receive distinct run IDs so partial failures are
preserved rather than overwritten.

Local mode is appropriate for this controlled single-GPU experiment, not a
production Spark deployment recommendation.
