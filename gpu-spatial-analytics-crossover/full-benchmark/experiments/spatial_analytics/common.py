"""Shared timing, telemetry, and correctness helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@contextmanager
def elapsed(sync=None) -> Iterator[dict[str, float]]:
    if sync:
        sync()
    started = time.perf_counter()
    result: dict[str, float] = {}
    yield result
    if sync:
        sync()
    result["seconds"] = time.perf_counter() - started


def gpu_sync() -> None:
    import cupy as cp

    cp.cuda.get_current_stream().synchronize()


def memory_snapshot() -> dict[str, int]:
    import pynvml
    import psutil

    process = psutil.Process()
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_used = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
    finally:
        pynvml.nvmlShutdown()
    return {"rss_bytes": int(process.memory_info().rss), "gpu_used_bytes": gpu_used}


def pair_signature(point_ids: np.ndarray, feature_ids: np.ndarray) -> str:
    if len(point_ids) != len(feature_ids):
        raise ValueError("pair arrays must have equal length")
    # Sorting tens of millions of pairs would turn validation into the dominant
    # memory cost. Two splitmix64 reductions plus count and xor are
    # order-independent and make accidental collisions vanishingly unlikely.
    point_values = point_ids.astype(np.uint64, copy=False)
    feature_values = feature_ids.astype(np.uint64, copy=False)
    keys = point_values ^ (feature_values + np.uint64(0x9E3779B97F4A7C15))

    def mix(values: np.ndarray, seed: int) -> np.ndarray:
        mixed = values + np.uint64(seed)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return mixed ^ (mixed >> np.uint64(31))

    first = mix(keys, 0x243F6A8885A308D3)
    second = mix(keys, 0x13198A2E03707344)
    first_sum = int(first.sum(dtype=np.uint64))
    second_sum = int(second.sum(dtype=np.uint64))
    xor = int(np.bitwise_xor.reduce(first, initial=np.uint64(0)))
    return f"{len(keys):016x}:{first_sum:016x}:{second_sum:016x}:{xor:016x}"


def distance_signature(point_ids: np.ndarray, distances: np.ndarray) -> str:
    quantized = np.rint(distances * 1_000_000_000_000).astype(np.int64)
    return pair_signature(point_ids, quantized)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def environment() -> dict[str, Any]:
    import importlib.metadata as metadata
    import pynvml
    import psutil

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        driver = pynvml.nvmlSystemGetDriverVersion()
        total = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
    finally:
        pynvml.nvmlShutdown()
    return {
        "host": os.environ.get("GPU_LAB_HOSTNAME", platform.node()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_affinity": psutil.Process().cpu_affinity(),
        "gpu": {"name": gpu_name, "driver": driver, "total_bytes": total},
        "packages": {
            name: metadata.version(name)
            for name in ("cudf", "cuspatial", "cupy", "geopandas", "numpy", "shapely")
        },
    }
