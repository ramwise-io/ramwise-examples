# parquet-gpu-break-even

Companion code for the ramwise.dev Parquet CPU/GPU benchmark post.

The example separates two things that should not be confused:

- [`notebook/`](notebook/) contains a small correctness-checked benchmark and
  notebook readers can run locally. It uses PyArrow and adds cuDF automatically
  when available.
- [`full-benchmark/`](full-benchmark/) contains the sanitized complete study
  harness, exact final configs, pinned environment, methodology, tests, and
  derived results.

The published 50-million-row study used a stricter internal harness: isolated
CPU cores, randomized measured trials, full correctness signatures,
independent replications, and continuous CPU/GPU telemetry. This folder is the
public reproduction boundary. It includes sanitized study configs, the pinned
software environment, methodology, derived result tables, and a compact
runnable implementation of the timed method. Internal server paths, service
inventory, and raw host telemetry are intentionally excluded.

## Run the small example

```bash
cd notebook
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python benchmark_parquet.py --rows 1000000 --projection core
pytest -q
```

If cuDF is installed in a compatible NVIDIA/RAPIDS environment, the same
command reports both PyArrow and cuDF. Otherwise it produces a valid CPU-only
result. For the GPU path, use the RAPIDS installation instructions appropriate
for your CUDA driver rather than adding `cudf` to this generic requirements
file.

## Read the result honestly

This is a warm-cache end-to-end read/materialization comparison, not a storage
benchmark. cuDF timing includes the returned GPU frame; the correctness check
converts it to Arrow after the timer stops. The first read is a warmup. Every
measured frame is checked for row count, selected columns, and numeric sums.

Do not compare your laptop's one-million-row result directly with the post's
50-million-row lab result. Small projected reads can favor PyArrow because GPU
startup overhead has not yet been amortized. Locating that crossover is a
useful experiment in its own right.

## Published study material

- [`full-benchmark/METHODOLOGY.md`](full-benchmark/METHODOLOGY.md) describes the
  full experimental protocol, hardware/software envelope, integrity counts,
  and limits.
- [`full-benchmark/environment.yml`](full-benchmark/environment.yml) pins the
  public Conda environment used by the benchmark image.
- [`full-benchmark/configs/`](full-benchmark/configs/) contains sanitized copies
  of the two final matrix definitions.
- [`full-benchmark/results/published_results.csv`](full-benchmark/results/published_results.csv)
  contains only the derived condition medians used by the article and
  notebook—no internal host metadata.
