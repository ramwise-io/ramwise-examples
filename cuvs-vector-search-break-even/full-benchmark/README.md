# Full GPU benchmark

This directory is a sanitized copy of the exact cuVS experiment harness used
for the published study. It retains source revision identifiers and derived
evidence without exposing the private lab repository or raw host telemetry.

The digest-pinned container installs the exact Conda explicit specification.
The published lock resolves Python 3.12.13, CuPy 14.1.1, cuVS/libcuVS 26.08.01
(build `25b1be43`), and CUDA 13.1. The benchmark host used Ubuntu 26.04; the
benchmark container intentionally uses the pinned Ubuntu 24.04 CUDA base.
Run the smoke gate before the full study:

```bash
bash run_matrix.sh smoke
bash run_matrix.sh full
```

The full matrix performs parameter tuning followed by three independent
confirmation builds. Generated datasets and raw results are written outside
the source checkout and are intentionally not committed.

See [`METHODOLOGY.md`](METHODOLOGY.md) for measurement boundaries and the
break-even definition.
