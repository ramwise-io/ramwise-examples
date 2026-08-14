from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, value: Any, length: int = 12) -> str:
    digest = hashlib.sha256(canonical_json(value).encode()).hexdigest()[:length]
    return f"{prefix}-{digest}"


def source_id() -> str:
    configured = os.environ.get("GPU_LAB_SOURCE_ID")
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def environment_metadata() -> dict[str, Any]:
    import cupy as cp
    import numpy as np

    props = cp.cuda.runtime.getDeviceProperties(0)

    def prop(name: str, default: Any = None) -> Any:
        return props.get(name, props.get(name.encode(), default))

    device_name = prop("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    return {
        "source_id": source_id(),
        "host": os.environ.get("GPU_LAB_HOSTNAME", "unknown"),
        "host_os": os.environ.get("GPU_LAB_HOST_OS", "unknown"),
        "image": os.environ.get("GPU_LAB_IMAGE", "unknown"),
        "image_id": os.environ.get("GPU_LAB_IMAGE_ID", "unknown"),
        "cpu_set": os.environ.get("GPU_LAB_CPUSET", "unknown"),
        "numpy_version": np.__version__,
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_local_runtime_version": int(cp.cuda.get_local_runtime_version()),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "gpu_name": str(device_name),
        "compute_capability": f"{int(prop('major', 0))}.{int(prop('minor', 0))}",
        "multiprocessors": int(prop("multiProcessorCount", 0)),
        "max_threads_per_multiprocessor": int(
            prop("maxThreadsPerMultiProcessor", 0)
        ),
        "total_global_memory_bytes": int(prop("totalGlobalMem", 0)),
    }
