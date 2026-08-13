# Published methodology

## Hardware and software envelope

- NVIDIA RTX PRO 4000 Blackwell SFF, 24 GB
- NVIDIA driver 595.71.05; CUDA 13.1; RAPIDS 26.08
- Intel Core Ultra 9 285HX; eight requested engine threads
- Ubuntu 26.04 host; digest-pinned Ubuntu 24.04 CUDA container
- Python 3.12.13, DuckDB 1.5.5, Polars 1.42.1, cuDF 26.08.00,
  PyArrow 23.0.1
- exact resolved packages in `conda-linux-64.lock`
- local NVMe input, warmed into the OS page cache

## Data and workloads

The deterministic fact table contains numeric, categorical string, and text
columns. The wide profile adds twelve floating-point metrics. A 100,000-row
customer dimension supplies segment and region attributes.

The crossover matrix covers 100,000 through 50 million rows and seven plans:
projected scan/group-by, join/group-by, string features, filtered top-k, and
feature aggregation over 2, 6, or 12 metric columns. The codec matrix fixes 20
million rows and compares Snappy with Zstandard. The out-of-core matrix uses a
200-million-row wide Zstandard dataset: 19.7 GB compressed and 34.6 GB logical.

## Timed and validation boundaries

The timer begins before Parquet input and stops after a CPU-backed Arrow result
is materialized. GPU work is synchronized before stopping the timer. Every
accepted result is canonicalized and checked against the DuckDB reference
outside the timed region.

Each condition has one warmup and three measured trials. Measured engine and
workload order is seeded and randomized. Process RSS/CPU and NVIDIA device
telemetry are sampled every 20 ms. Whole-device GPU memory is retained only as
private host context because unrelated resident services prevent honest
per-process attribution.

Polars GPU runs with CPU fallback disabled. The default cuDF-Polars streaming
executor is therefore measured as a real GPU path, not a transparent CPU run.

## Larger-than-VRAM controls

The one-shot native-cuDF read is retained as a failed control: with generic
spilling enabled, it attempted a 23.3 GB device allocation and exhausted VRAM.
The separate `cudf-streaming` condition is an explicit bounded algorithm that
reads one file at a time and merges exact partial aggregates. It is not labeled
as automatic native-cuDF out-of-core execution.

The successful out-of-core matrix used:

```text
CUDF_SPILL=on
CUDF_SPILL_DEVICE_LIMIT=17179869184
RAPIDSMPF_PINNED_MEMORY=true
RAPIDSMPF_PINNED_MAX_POOL_SIZE=32GiB
CUDF_POLARS__EXECUTOR__FALLBACK_MODE=raise
```

## Limits

These results apply to warm-cache local-NVMe synthetic workloads on one
workstation and fixed versions. They do not cover remote object storage, cold
cache, many tiny files, energy, or multi-GPU execution. Three trials identify
the large effects measured here but do not justify narrow formal confidence
intervals for small differences.
