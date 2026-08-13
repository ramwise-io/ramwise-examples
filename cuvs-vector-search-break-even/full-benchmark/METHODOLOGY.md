# Methodology

## Question

When does the one-time cost of building an approximate nearest-neighbour index
become smaller than the cumulative GPU time saved relative to resident exact
brute-force search?

## Data

The study uses deterministic, unit-normalized clustered Gaussian vectors.
Queries and corpus rows use independent random streams, so scale cases with the
same dimensionality receive identical queries. Squared Euclidean distance and
cosine similarity induce the same ranking for unit vectors.

The controlled matrix covers:

- 100,000, 1,000,000, and 5,000,000 vectors at 128 dimensions;
- 1,000,000 vectors at 384 and 768 dimensions;
- query batches of 1, 32, 256, and 1,024;
- recall@10 targets of 0.80, 0.90, 0.95, and 0.99; and
- exact brute force, IVF-Flat, compact IVF-PQ, balanced IVF-PQ, and CAGRA.

This synthetic distribution makes the experiment deterministic and allows all
ground truth to be calculated exactly. It is not a substitute for testing a
production embedding model and corpus.

## Selection and confirmation

Each algorithm and dataset first receives one parameter-grid tuning pass. The
fastest parameter set whose tuning recall clears the target plus a 0.02 cushion
(capped at 1.0) is selected. Only the union of those selected settings proceeds
to three fresh-process confirmation replications. A target is substantiated
only when the minimum confirmation recall still meets it. Confirmation results
are never used to choose replacement parameters.

Exact cuVS neighbours are computed once per dataset and persisted. Every ANN
result is checked against that immutable reference.

## Timers

- Upload covers host NumPy arrays to resident CuPy arrays and synchronization.
- Build covers the cuVS build call through resource synchronization.
- Search covers a single cuVS call through resource synchronization.
- Serialization is recorded separately and excluded from build amortization.
- Latency trials exclude NVML-thread startup. Telemetry is collected in a
  separate 250 ms sustained-search window.
- VRAM fields are absolute whole-device readings after build and at the search
  peak. They are not presented as process-only deltas because unrelated
  whole-device allocations can change between snapshots.
- Each confirmation process performs two warmups and five timed searches per
  selected batch/parameter pair.

## Break-even

Dataset upload is a common cost and is excluded from both resident alternatives.
For query batch size `B`:

```text
saved seconds/query = (exact batch seconds - ANN batch seconds) / B
break-even queries  = ANN build seconds / saved seconds/query
```

If ANN search is not faster, the result has no finite break-even point. Query
counts are also divided by 1, 10, 100, and 1,000 QPS to show time-to-amortize
workload scenarios. These rates do not simulate production queueing.

## Scope boundaries

The study measures warm, resident, single-GPU search. It excludes network
ingress, vector-database metadata, filters, concurrent tenants, index updates,
deletions, queueing, replication, persistence restore time, and multi-GPU
execution. VRAM, persisted index size, build time, latency, throughput, and
recall are reported separately rather than collapsed into a composite score.
