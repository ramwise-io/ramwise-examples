# Measurement method

- Fixed seed: `20260901`.
- One warm-up and five measured Runs per safe profile.
- Ordinary warm local-system conditions; the OS page cache is not flushed.
- Anatini Job resource profile: 6 GB DuckDB memory limit and four threads.
- Step time uses `perf_counter_ns`; end-to-end duration comes from the Run.
- Every correctness failure aborts and remains visible.
- Measurements are retained in the Run and appended to
  `benchmark.measurements` only by the final publication notebook.
- Summary statistics are median, minimum, and maximum. No observation is silently
  discarded.

The canonical research protocol and final interpretation live in the sibling
`flip-the-default-research` investigation. This project owns the executable
workload and its Run evidence.
