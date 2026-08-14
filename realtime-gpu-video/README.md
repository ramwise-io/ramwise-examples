# Five 4K60 streams made the deadline. Six did not.

This companion contains an output-complete notebook and the sanitized derived
evidence behind the Ramwise real-time GPU video study.

The private load generator replayed deterministic H.264 clips through an
NVDEC -> CUDA luma processing -> NVENC pipeline. It paced each stream at its
declared frame rate and measured latency, deadline misses, queue growth,
throughput, codec-engine utilization, GPU utilization, VRAM, and power. The
matrix varied resolution, frame rate, processing work, codec, memory
placement, and concurrent 4K60 stream count.

## Files

- `realtime_gpu_video.ipynb`: runnable explanation, local queue model, and
  charts from the 51-run confirmation study;
- `results/case-medians.csv`: medians and replication ranges for all 17
  confirmed cases;
- `results/summary.json`: the same derived evidence plus the confirmed
  capacity boundary;
- `results/README.md`: timing, correctness, aggregation, and provenance notes.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

The notebook does not require an NVIDIA GPU. It works from the published
derived results and includes a small queue simulation readers can change. The
full hardware harness remains in the private lab repository because it also
contains machine-specific orchestration and raw per-frame telemetry.

The confirmed hardware was an NVIDIA RTX PRO 4000 Blackwell SFF Edition with
24 GB of VRAM. The Ubuntu 26.04 host ran a pinned Ubuntu 24.04 container with
Python 3.12.13, PyNvVideoCodec 2.1.0, CuPy 14.1.1, CUDA 13.2 runtime reporting,
FFmpeg 6.1.1, and driver 595.71.05.
