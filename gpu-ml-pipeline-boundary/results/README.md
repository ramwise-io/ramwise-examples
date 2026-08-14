# Published results

This directory contains the derived evidence used by the article and notebook:

- `published_estimator_results.csv`: fit and one-shot estimator medians,
  fresh-process ranges, quality, memory, and speedups;
- `published_inference_results.csv`: batch latency, throughput, speedup, and
  lifecycle break-even estimates; and
- `published_pipeline_results.csv`: end-to-end placement results, stage
  medians, explicit transfer accounting, quality, and reuse break-even.

The CSVs are generated only after every configured quality and GPU-dispatch
control passes. They publish condition-level evidence, including replication
ranges, but omit raw JSON, sampled telemetry, hostnames, private paths, and
container logs.
