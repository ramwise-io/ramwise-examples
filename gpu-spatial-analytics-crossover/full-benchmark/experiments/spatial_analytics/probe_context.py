"""Inspect CUDA context ownership across the cuSpatial 25.04 Python stack."""

from __future__ import annotations

import os


def report(label: str) -> None:
    from numba.cuda.cudadrv.driver import driver

    with driver.get_active_context() as active:
        print(
            label,
            {
                "context_handle": active.context_handle,
                "device_ordinal": active.devnum,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        )


def main() -> None:
    from numba import config, cuda
    from numba.cuda.cudadrv import driver

    print("numba_use_nvidia_binding", config.CUDA_USE_NVIDIA_BINDING)
    print("driver_use_nvidia_binding", driver.USE_NV_BINDING)
    print("numba_devices_before", cuda.list_devices())
    report("before_cupy")
    import cupy as cp

    cp.arange(8).sum().item()
    print("cupy_device", cp.cuda.runtime.getDevice())
    print("cupy_device_count", cp.cuda.runtime.getDeviceCount())
    print("cupy_compute_capability", cp.cuda.Device().compute_capability)
    try:
        from cuda.bindings import driver as cuda_driver

        print("cuda_bindings_ctx_get_device", cuda_driver.cuCtxGetDevice())
        print("cuda_bindings_device_count", cuda_driver.cuDeviceGetCount())
    except Exception as exc:
        print("cuda_bindings_probe_failed", repr(exc))
    print("numba_devices_after", cuda.list_devices())
    report("after_cupy")


if __name__ == "__main__":
    main()
