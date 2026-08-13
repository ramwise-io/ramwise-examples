"""Continuous GPU and host CPU telemetry for short benchmark regions."""

from __future__ import annotations

import os
import statistics
import threading
import time
from typing import Any

import psutil

from experiments.parquet_decompression.common import scalar


def _nvml() -> tuple[Any, Any]:
    import pynvml

    pynvml.nvmlInit()
    return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)


def gpu_snapshot() -> dict[str, Any]:
    try:
        pynvml, handle = _nvml()
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        snapshot = {
            "clock_memory_mhz": int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)),
            "clock_sm_mhz": int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
            "driver": scalar(pynvml.nvmlSystemGetDriverVersion()),
            "memory_total_bytes": int(memory.total),
            "memory_used_bytes": int(memory.used),
            "name": scalar(pynvml.nvmlDeviceGetName(handle)),
            "performance_state": int(pynvml.nvmlDeviceGetPerformanceState(handle)),
            "temperature_c": int(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            ),
            "utilization_gpu_percent": int(utilization.gpu),
            "utilization_memory_percent": int(utilization.memory),
        }
        try:
            snapshot["power_mw"] = int(pynvml.nvmlDeviceGetPowerUsage(handle))
        except pynvml.NVMLError:
            snapshot["power_mw"] = None
        return snapshot
    except Exception as error:  # pragma: no cover - exercised only without NVML
        return {"error": str(error)}


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0}
    summary: dict[str, Any] = {
        "duration_seconds": samples[-1]["elapsed_ns"] / 1_000_000_000,
        "sample_count": len(samples),
    }
    reductions = {
        "clock_memory_mhz": (min, max),
        "clock_sm_mhz": (min, max),
        "load_average_1m": (min, max),
        "memory_used_bytes": (min, max),
        "performance_state": (min, max),
        "power_mw": (min, max),
        "process_rss_bytes": (min, max),
        "temperature_c": (min, max),
        "utilization_gpu_percent": (min, max),
        "utilization_memory_percent": (min, max),
    }
    for field, (minimum, maximum) in reductions.items():
        values = [sample[field] for sample in samples if sample.get(field) is not None]
        if values:
            summary[f"{field}_min"] = minimum(values)
            summary[f"{field}_max"] = maximum(values)
            summary[f"{field}_mean"] = statistics.fmean(values)

    interval_metrics: list[dict[str, float]] = []
    for before, after in zip(samples, samples[1:]):
        elapsed_seconds = (after["elapsed_ns"] - before["elapsed_ns"]) / 1_000_000_000
        if elapsed_seconds <= 0:
            continue
        required = (
            "affinity_cpu_busy_seconds",
            "affinity_cpu_total_seconds",
            "host_cpu_busy_seconds",
            "host_cpu_total_seconds",
            "process_cpu_seconds",
        )
        if not all(field in before and field in after for field in required):
            continue
        affinity_busy = (
            after["affinity_cpu_busy_seconds"] - before["affinity_cpu_busy_seconds"]
        )
        affinity_total = (
            after["affinity_cpu_total_seconds"] - before["affinity_cpu_total_seconds"]
        )
        host_busy = after["host_cpu_busy_seconds"] - before["host_cpu_busy_seconds"]
        host_total = after["host_cpu_total_seconds"] - before["host_cpu_total_seconds"]
        process_cpu = after["process_cpu_seconds"] - before["process_cpu_seconds"]
        if affinity_total <= 0 or host_total <= 0:
            continue
        process_cores = max(0.0, process_cpu / elapsed_seconds)
        affinity_busy_cores = max(0.0, affinity_busy / elapsed_seconds)
        interval_metrics.append(
            {
                "affinity_cpu_utilization_percent": 100 * affinity_busy / affinity_total,
                "background_affinity_cpu_cores": max(
                    0.0, affinity_busy_cores - process_cores
                ),
                "host_cpu_utilization_percent": 100 * host_busy / host_total,
                "process_cpu_cores": process_cores,
            }
        )
    for field in (
        "affinity_cpu_utilization_percent",
        "background_affinity_cpu_cores",
        "host_cpu_utilization_percent",
        "process_cpu_cores",
    ):
        values = [metric[field] for metric in interval_metrics]
        if values:
            summary[f"{field}_mean"] = statistics.fmean(values)
            summary[f"{field}_max"] = max(values)
    return summary


def _cpu_seconds(cpu_times: Any) -> tuple[float, float]:
    """Return Linux-compatible cumulative busy and total CPU seconds."""

    total = sum(cpu_times)
    total -= getattr(cpu_times, "guest", 0.0)
    total -= getattr(cpu_times, "guest_nice", 0.0)
    idle = cpu_times.idle + getattr(cpu_times, "iowait", 0.0)
    return total - idle, total


class GpuTelemetrySampler:
    """Poll NVML and host CPU counters around a measured region."""

    def __init__(self, interval_ms: int) -> None:
        if interval_ms < 0:
            raise ValueError("telemetry interval must be non-negative")
        self.interval_ms = interval_ms
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_ns = 0
        self._pynvml: Any = None
        self._handle: Any = None
        self._process = psutil.Process()
        self._cpu_affinity = sorted(self._process.cpu_affinity())

    def _capture(self) -> None:
        try:
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            cpu_times = psutil.cpu_times(percpu=True)
            host_busy, host_total = (0.0, 0.0)
            affinity_busy, affinity_total = (0.0, 0.0)
            for cpu_index, times in enumerate(cpu_times):
                busy, total = _cpu_seconds(times)
                host_busy += busy
                host_total += total
                if cpu_index in self._cpu_affinity:
                    affinity_busy += busy
                    affinity_total += total
            process_times = self._process.cpu_times()
            sample = {
                "affinity_cpu_busy_seconds": affinity_busy,
                "affinity_cpu_total_seconds": affinity_total,
                "clock_memory_mhz": int(
                    self._pynvml.nvmlDeviceGetClockInfo(
                        self._handle, self._pynvml.NVML_CLOCK_MEM
                    )
                ),
                "clock_sm_mhz": int(
                    self._pynvml.nvmlDeviceGetClockInfo(
                        self._handle, self._pynvml.NVML_CLOCK_SM
                    )
                ),
                "elapsed_ns": time.perf_counter_ns() - self._started_ns,
                "host_cpu_busy_seconds": host_busy,
                "host_cpu_total_seconds": host_total,
                "load_average_1m": os.getloadavg()[0],
                "memory_used_bytes": int(memory.used),
                "performance_state": int(
                    self._pynvml.nvmlDeviceGetPerformanceState(self._handle)
                ),
                "temperature_c": int(
                    self._pynvml.nvmlDeviceGetTemperature(
                        self._handle, self._pynvml.NVML_TEMPERATURE_GPU
                    )
                ),
                "process_cpu_seconds": process_times.user + process_times.system,
                "process_rss_bytes": self._process.memory_info().rss,
                "utilization_gpu_percent": int(utilization.gpu),
                "utilization_memory_percent": int(utilization.memory),
            }
            try:
                sample["power_mw"] = int(
                    self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
                )
            except self._pynvml.NVMLError:
                sample["power_mw"] = None
            self.samples.append(sample)
        except Exception as error:  # pragma: no cover - hardware/driver failure
            self.error = str(error)
            self._stop.set()

    def _run(self) -> None:
        interval_seconds = self.interval_ms / 1000
        while not self._stop.wait(interval_seconds):
            self._capture()

    def start(self) -> None:
        if self.interval_ms == 0:
            return
        self._pynvml, self._handle = _nvml()
        self._started_ns = time.perf_counter_ns()
        self._capture()
        self._thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self.interval_ms == 0:
            return {"interval_ms": 0, "samples": [], "summary": {"sample_count": 0}}
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_ms / 1000 * 4))
        self._capture()
        return {
            "cpu_affinity": self._cpu_affinity,
            "error": self.error,
            "interval_ms": self.interval_ms,
            "samples": self.samples,
            "summary": summarize_samples(self.samples),
        }
