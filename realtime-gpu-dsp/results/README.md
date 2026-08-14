# Published evidence

These are derived medians, not raw telemetry.

| File | Matrix | Source commit | Protocol |
| --- | --- | --- | --- |
| `capacity.csv` | `realtime-dsp-full-d9bf157a1a` | `fed1a9f695f54479b7731f2df9a394a1668bb373` | 3 warmups, 7 measured trials of 16 blocks, 3 fresh processes |
| `paced-full.csv` | `realtime-dsp-full-d9bf157a1a` | `fed1a9f695f54479b7731f2df9a394a1668bb373` | 1-second arrivals, bounded 8-block queue, 3 fresh processes |
| `paced-confirm.csv` | `realtime-dsp-confirm-a3f32ed0f5` | `6b7da3294b16dee6332058cff1143848d06bce58` | 5-second arrivals around selected boundaries, 3 fresh processes |

For each row, scalar timing and fraction fields are the median of the three
replication-level values. `max_queue_depth` is the maximum across
replications. A block misses its deadline when completion occurs after the
next scheduled arrival. The producer is schedule-driven and never waits for
the consumer; arrivals are dropped when the eight-block queue is full.

Input generation and replay preparation are outside timing. The pinned path
uses preallocated, prepopulated page-locked buffers to represent an acquisition
stack that can DMA directly into pinned memory. The resident path is an upper
bound with input already on the GPU. All GPU work is synchronized, and every
fresh process passes the SciPy correctness gate before recording performance.

The host was Ubuntu 26.04 LTS. The benchmark container used Python 3.12,
NumPy 2.4.6, SciPy 1.16.3, CuPy 14.1.1, and CUDA 13.1. The GPU was an NVIDIA
RTX PRO 4000 Blackwell SFF Edition with 24 GB VRAM.
