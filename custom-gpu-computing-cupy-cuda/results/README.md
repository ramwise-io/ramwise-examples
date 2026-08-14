# Published results

These files are derived condition-level evidence from the publication matrix:

- `published_launch_results.csv`: Python/CUDA launch cost across queued launch
  counts;
- `published_transfer_results.csv`: H2D and D2H copy medians and replication
  ranges for pageable and pinned host memory;
- `published_fusion_results.csv`: NumPy, composed CuPy, partial fusion, and raw
  CUDA timings, correctness, layout conversion, kernel attributes, and
  CPU-relative speedups; and
- `published_profile_results.csv`: selected Nsight Compute launch, occupancy,
  and memory-workload metrics from five representative captures.

The timing tables contain three independent fresh-process replications. Some
NumPy reference rows contain six because the same reference condition also ran
inside the separate block-size sweep. Profiler duration is intentionally
included only as profiling context: replay and counter instrumentation perturb
it, so performance claims use the ordinary benchmark table.

All configured GPU outputs passed the documented numeric quality gate. Raw
per-trial JSON, NVTX/profiler reports, logs, matrix identifiers, hostnames, and
private paths remain in the internal lab.

