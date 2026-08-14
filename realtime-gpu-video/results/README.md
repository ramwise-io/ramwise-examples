# Published evidence

These files were generated from the immutable summaries for confirmation
study `20260814T212604Z-confirm`.

The schedule contained 17 conditions and three process-level replications per
condition. Its 51 runs were shuffled with seed `20260814`. Each worker warmed
15 frames before the measured interval and then processed the remaining 585
frames of a ten-second 60 FPS clip, or 285 frames for 30 FPS.

For every run, a decoded frame was processed once before timing and compared
exactly with a NumPy reference. After timing, FFmpeg independently decoded the
elementary output and verified codec, dimensions, and frame count. A result
was retained only after both checks passed.

The frame deadline was one frame period: 16.667 ms at 60 FPS and 33.333 ms at
30 FPS. A case passed when aggregate throughput reached at least 99% of the
arrival rate, the deadline-miss rate was at most 1%, and p99 latency stayed
within one frame period.

`case-medians.csv` contains medians plus minimum and maximum values across the
three replications. NVML telemetry was sampled every 100 ms. Output rate is
the total elementary-bitstream size divided by the ten-second source duration.
Raw frame traces and temporary bitstreams are intentionally excluded from this
public repository.
