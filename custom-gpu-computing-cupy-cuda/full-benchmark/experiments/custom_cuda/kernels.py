from __future__ import annotations

from typing import Any


TOUCH_SOURCE = r"""
extern "C" __global__
void touch(float* value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        value[0] += 1.0f;
    }
}
"""


SCORE_SOURCE = r"""
__device__ __forceinline__ float transform_value(
    float value,
    float mean,
    float inverse_std,
    float weight
) {
    float z = (value - mean) * inverse_std;
    z = fminf(5.0f, fmaxf(-5.0f, z));
    return z * weight;
}

extern "C" __global__
void score_aos(
    const float* __restrict__ values,
    const float* __restrict__ means,
    const float* __restrict__ inverse_stds,
    const float* __restrict__ weights,
    float* __restrict__ scores,
    unsigned char* __restrict__ flags,
    int rows,
    int features,
    float threshold
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float score = 0.0f;
    int base = row * features;
    for (int feature = 0; feature < features; ++feature) {
        score += transform_value(
            values[base + feature], means[feature], inverse_stds[feature], weights[feature]
        );
    }
    scores[row] = score;
    flags[row] = score > threshold;
}

extern "C" __global__
void score_soa(
    const float* __restrict__ values,
    const float* __restrict__ means,
    const float* __restrict__ inverse_stds,
    const float* __restrict__ weights,
    float* __restrict__ scores,
    unsigned char* __restrict__ flags,
    int rows,
    int features,
    float threshold
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float score = 0.0f;
    for (int feature = 0; feature < features; ++feature) {
        score += transform_value(
            values[feature * rows + row], means[feature], inverse_stds[feature], weights[feature]
        );
    }
    scores[row] = score;
    flags[row] = score > threshold;
}

extern "C" __global__
void score_strided(
    const float* __restrict__ values,
    const float* __restrict__ means,
    const float* __restrict__ inverse_stds,
    const float* __restrict__ weights,
    float* __restrict__ scores,
    unsigned char* __restrict__ flags,
    int rows,
    int features,
    int row_stride,
    float threshold
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float score = 0.0f;
    int base = row * row_stride;
    for (int feature = 0; feature < features; ++feature) {
        score += transform_value(
            values[base + feature], means[feature], inverse_stds[feature], weights[feature]
        );
    }
    scores[row] = score;
    flags[row] = score > threshold;
}
"""


PARTIAL_FUSION_OPERATION = r"""
int feature = i % features;
float z = (x - means[feature]) * inverse_stds[feature];
z = min(5.0f, max(-5.0f, z));
out = z * weights[feature];
"""


def touch_kernel() -> Any:
    import cupy as cp

    return cp.RawKernel(TOUCH_SOURCE, "touch", options=("-std=c++17",))


def score_module() -> Any:
    import cupy as cp

    return cp.RawModule(
        code=SCORE_SOURCE,
        options=("-std=c++17", "--use_fast_math"),
        name_expressions=("score_aos", "score_soa", "score_strided"),
    )


def partial_fusion_kernel() -> Any:
    import cupy as cp

    return cp.ElementwiseKernel(
        "float32 x, raw float32 means, raw float32 inverse_stds, "
        "raw float32 weights, int32 features",
        "float32 out",
        PARTIAL_FUSION_OPERATION,
        "partial_feature_transform",
    )


def kernel_attributes(kernel: Any, block_size: int, metadata: dict[str, Any]) -> dict[str, Any]:
    import cupy as cp

    attributes = dict(kernel.attributes)
    active_blocks = None
    theoretical_occupancy = None
    try:
        active_blocks = int(
            cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
                kernel.kernel.ptr, int(block_size), 0
            )
        )
        maximum = int(metadata.get("max_threads_per_multiprocessor", 0))
        if maximum:
            theoretical_occupancy = min(1.0, active_blocks * block_size / maximum)
    except Exception:
        pass
    return {
        "block_size": int(block_size),
        "registers_per_thread": int(attributes.get("num_regs", -1)),
        "static_shared_memory_bytes": int(attributes.get("shared_size_bytes", -1)),
        "local_memory_bytes": int(attributes.get("local_size_bytes", -1)),
        "max_threads_per_block": int(attributes.get("max_threads_per_block", -1)),
        "active_blocks_per_multiprocessor": active_blocks,
        "theoretical_occupancy": theoretical_occupancy,
    }
