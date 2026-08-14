# Methodology

## Question

When does custom CUDA improve a useful data-processing operation over NumPy or
ordinary CuPy, and which part of the gain comes from launch fusion, memory
residency, layout, or launch configuration?

The study does not treat the fastest kernel-only number as the answer. Cold
compilation, host/device transfer, layout conversion, Python submission, and
output materialization are reported separately.

## Hardware and environment

- Ubuntu 26.04 host operating system.
- NVIDIA RTX PRO 4000 Blackwell SFF Edition, 24 GB VRAM, compute capability
  12.0.
- Eight pinned logical CPU cores (`0-7`) for NumPy comparisons.
- RAPIDS 26.08 container with Python 3.12, CuPy 14.1.1, NumPy 2.4.6, a CUDA
  13.1 local runtime, CUDA 13.2 linked/build toolchain, and NVRTC 13.1.
- Matrix identity binds the complete configuration to the Git source commit.

The server's normal Open WebUI, SearXNG, Ollama, and Dockge services remain
running. The study measures the workstation as operated rather than claiming
exclusive datacenter-GPU conditions.

## Operation

For each row `r` and feature `f`, the operation computes:

```text
z[r,f] = clip((x[r,f] - mean[f]) * inverse_std[f], -5, 5)
score[r] = sum(z[r,f] * weight[f])
flag[r] = score[r] > 0
```

NumPy produces the reference score and flag arrays. GPU score arrays must match
with `rtol=3e-4` and `atol=3e-4`. A flag difference is accepted only when the
reference score lies within the observed numeric error of the zero threshold.

## Implementations

- `numpy`: vectorized host implementation.
- `cupy_composed`: separate standardize, clip, multiply, reduction, comparison,
  and cast operations with ordinary CuPy expressions.
- `cupy_partial_fusion`: an `ElementwiseKernel` fuses standardize, clip, and
  weighting; CuPy still performs reduction and thresholding separately.
- `raw_aos`: one thread scores one row from C-contiguous row-major input. Its
  code is simple, but adjacent threads in a warp read different rows.
- `raw_soa`: identical arithmetic over feature-major input, so adjacent threads
  read adjacent values for a given feature.
- `raw_strided`: the row-major kernel reads from a padded row stride, making the
  separation between adjacent threads still larger.

The feature-major input is generated as a valid upstream layout before the
one-shot timer. Separately reported AoS-to-SoA conversion timing answers whether
changing an already resident row-major input would repay itself.

## Timing

Every process performs one or more unmeasured warmups, then measured trials.
Publication matrices use seven trials per process and three independent
fresh-process replications. Trial order among resident GPU implementations is
seeded and randomized.

- `resident_wall_seconds`: Python wall time from immediately before the CUDA
  event/operation until the ending event synchronizes. Inputs are already on
  the relevant device layout; output allocation is included.
- `resident_device_seconds`: CUDA-event elapsed time over the same operation.
- `one_shot_seconds`: pageable host input transfer, GPU work, score and flag
  materialization back to NumPy, and final synchronization. The NumPy reference
  requires no transfer, so its resident and one-shot values are identical.
- `cold_compile_seconds`: attributed `RawModule` or `ElementwiseKernel`
  compilation using an empty per-run CuPy cache. A one-element type-compatible
  probe prevents workload execution from contaminating the partial-fusion
  compile timer. Compilation is excluded from all steady timings.
- `aos_to_soa_conversion_seconds`: explicit device transpose/materialization,
  reported separately rather than silently charged to one layout.

The one-shot comparison for `raw_soa` assumes feature-major host input already
exists. Readers starting with row-major input must add the measured conversion
or retain that layout.

## Launch and transfer calibration

The launch test repeatedly applies either a one-element CuPy add or a
handwritten one-block CUDA kernel. It reports both device-event and wall time
per launch across 1, 10, 100, and 1,000 queued launches.

The transfer test preallocates device and host buffers, then measures H2D and
D2H copies for pageable and CUDA-pinned host memory. CUDA-event duration and
synchronized wall duration are both retained. Buffer allocation is excluded;
the result characterizes transfer, not allocator policy.

## Configuration and profiling

The full matrix sweeps row count at 16 features, feature width at 262,144 rows,
and block sizes from 32 through 1,024 for representative narrow and wide
conditions. Kernel attributes record registers, static shared memory, active
blocks per SM, and calculated theoretical occupancy.

Nsight Compute is used only for representative stable conditions because
counter collection can replay kernels and materially perturb execution. Raw
profiler reports remain private evidence; the public table contains selected
launch, occupancy, and memory-workload metrics with the exact command.
The short-lived profiler container runs as root with `SYS_ADMIN` because the
host driver restricts performance-counter access. The normal benchmark
container remains non-root, and no host driver setting is changed.

## Interpretation limits

The operation is dense, numeric, single-GPU, and synthetic. It does not cover
irregular graphs, atomics under contention, tensor-core matrix multiplication,
multi-GPU communication, unified-memory oversubscription, or custom C++ host
extensions. Results establish mechanisms on this workstation, not universal
CUDA constants.
