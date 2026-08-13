"""Process-tree CPU and NVML telemetry around a Spark action."""

from __future__ import annotations

import statistics
import threading
import time
from typing import Any

import psutil


def _nvml() -> tuple[Any, Any]:
    import pynvml

    pynvml.nvmlInit()
    return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)


def _process_tree(root: psutil.Process) -> list[psutil.Process]:
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return processes


def _tree_counters(root: psutil.Process) -> tuple[float, int]:
    cpu_seconds = 0.0
    rss_bytes = 0
    for process in _process_tree(root):
        try:
            times = process.cpu_times()
            cpu_seconds += times.user + times.system
            rss_bytes += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return cpu_seconds, rss_bytes


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0}
    result: dict[str, Any] = {
        "duration_seconds": samples[-1]["elapsed_seconds"],
        "sample_count": len(samples),
    }
    for field in (
        "gpu_memory_used_bytes",
        "gpu_power_watts",
        "gpu_temperature_c",
        "gpu_utilization_percent",
        "host_cpu_percent",
        "process_tree_rss_bytes",
    ):
        values = [float(sample[field]) for sample in samples if sample.get(field) is not None]
        if values:
            result[f"{field}_mean"] = statistics.fmean(values)
            result[f"{field}_max"] = max(values)
    cores = []
    energy_joules = 0.0
    for before, after in zip(samples, samples[1:]):
        elapsed = after["elapsed_seconds"] - before["elapsed_seconds"]
        if elapsed <= 0:
            continue
        cores.append(max(0.0, (after["process_tree_cpu_seconds"] - before["process_tree_cpu_seconds"]) / elapsed))
        if before.get("gpu_power_watts") is not None and after.get("gpu_power_watts") is not None:
            energy_joules += elapsed * (before["gpu_power_watts"] + after["gpu_power_watts"]) / 2
    if cores:
        result["process_tree_cpu_cores_mean"] = statistics.fmean(cores)
        result["process_tree_cpu_cores_max"] = max(cores)
    result["gpu_energy_joules_estimate"] = energy_joules
    return result


class TelemetrySampler:
    def __init__(self, interval_ms: int = 100) -> None:
        self.interval_ms = interval_ms
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self._root = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._pynvml: Any = None
        self._handle: Any = None

    def _capture(self) -> None:
        try:
            cpu_seconds, rss_bytes = _tree_counters(self._root)
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            sample = {
                "elapsed_seconds": time.perf_counter() - self._started,
                "gpu_memory_used_bytes": int(memory.used),
                "gpu_temperature_c": int(
                    self._pynvml.nvmlDeviceGetTemperature(
                        self._handle, self._pynvml.NVML_TEMPERATURE_GPU
                    )
                ),
                "gpu_utilization_percent": int(utilization.gpu),
                "host_cpu_percent": float(psutil.cpu_percent(interval=None)),
                "process_tree_cpu_seconds": cpu_seconds,
                "process_tree_rss_bytes": rss_bytes,
            }
            try:
                sample["gpu_power_watts"] = self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000
            except self._pynvml.NVMLError:
                sample["gpu_power_watts"] = None
            self.samples.append(sample)
        except Exception as error:  # pragma: no cover - hardware failure
            self.error = str(error)
            self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_ms / 1000):
            self._capture()

    def start(self) -> None:
        if self.interval_ms <= 0:
            return
        self._pynvml, self._handle = _nvml()
        psutil.cpu_percent(interval=None)
        self._started = time.perf_counter()
        self._capture()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self.interval_ms <= 0:
            return {"interval_ms": self.interval_ms, "samples": [], "summary": {"sample_count": 0}}
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._capture()
        return {
            "error": self.error,
            "interval_ms": self.interval_ms,
            "samples": self.samples,
            "summary": summarize(self.samples),
        }

