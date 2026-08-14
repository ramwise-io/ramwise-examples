from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GpuSampler:
    interval_seconds: float = 0.02
    samples: list[dict[str, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            started = time.perf_counter()
            while not self._stop.is_set():
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except pynvml.NVMLError:
                    power = 0.0
                self.samples.append(
                    {
                        "seconds": time.perf_counter() - started,
                        "memory_used_bytes": float(memory.used),
                        "power_watts": power,
                        "utilization_percent": float(utilization.gpu),
                    }
                )
                self._stop.wait(self.interval_seconds)
        except Exception:
            return

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if not self.samples:
            return {"samples": 0}
        energy = 0.0
        for left, right in zip(self.samples, self.samples[1:]):
            duration = right["seconds"] - left["seconds"]
            energy += duration * (left["power_watts"] + right["power_watts"]) / 2
        return {
            "samples": len(self.samples),
            "memory_used_min_bytes": min(x["memory_used_bytes"] for x in self.samples),
            "memory_used_max_bytes": max(x["memory_used_bytes"] for x in self.samples),
            "utilization_mean_percent": sum(
                x["utilization_percent"] for x in self.samples
            )
            / len(self.samples),
            "energy_joules_estimate": energy,
        }

