"""Fail-closed Blackwell compatibility for cuSpatial's 25.04 Python stack."""

from __future__ import annotations


def initialize_single_gpu_context() -> None:
    """Correct Numba 0.60's stale active-context ordinal on this one-GPU host.

    cuSpatial 25.04 depends on a Numba CUDA Array Interface conversion. On the
    current 595 driver, that release's Numba wrapper reports the active CuPy
    context as ordinal 3 even though both CuPy and NVIDIA's CUDA bindings report
    exactly one device, ordinal 0. We validate those independent facts before
    changing Numba's thread-local mapping. This must never be used on a
    multi-GPU process.
    """
    import cupy as cp
    from cuda.bindings import driver as cuda_driver
    from numba.cuda.cudadrv.driver import driver as numba_driver

    cp.arange(1, dtype=cp.int8).sum().item()
    cupy_count = cp.cuda.runtime.getDeviceCount()
    cupy_ordinal = cp.cuda.runtime.getDevice()
    result, binding_count = cuda_driver.cuDeviceGetCount()
    context_result, binding_ordinal = cuda_driver.cuCtxGetDevice()
    if int(result) != 0 or int(context_result) != 0:
        raise RuntimeError("NVIDIA CUDA bindings could not inspect the active context")
    if (cupy_count, cupy_ordinal, binding_count, int(binding_ordinal)) != (1, 0, 1, 0):
        raise RuntimeError(
            "cuSpatial compatibility shim is restricted to one visible GPU at ordinal 0"
        )

    active = numba_driver.get_active_context()
    active.__enter__()
    try:
        if active.context_handle is None:
            raise RuntimeError("Numba did not observe CuPy's active CUDA context")
        active_type = type(active)
        active_type._tls_cache.ctx_devnum = (active.context_handle, 0)
    finally:
        # Do not call __exit__: it would remove the corrected thread-local map.
        active._is_top = False

