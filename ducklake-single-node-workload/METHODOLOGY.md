# Methodology

## Locked protocol

The workload and repetition plan were committed before the first measured Run. One exploratory quick Run validated measurement plumbing and was excluded from the main summaries.

| Profile | Base orders | Changes | Events | Customers | Products |
|---|---:|---:|---:|---:|---:|
| quick | 100,000 | 5,000 | 200,000 | 10,000 | 1,000 |
| small | 1,000,000 | 50,000 | 2,000,000 | 100,000 | 10,000 |
| medium | 10,000,000 | 500,000 | 20,000,000 | 1,000,000 | 100,000 |
| large | 50,000,000 | 2,500,000 | 100,000,000 | 5,000,000 | 500,000 |

Each profile received one warm-up and five measured complete Job Runs. Runs were serial and used deterministic seed `20260901`. The OS page cache was not flushed.

## Measurement boundaries

Every data notebook used `time.perf_counter_ns()` around work owned by that step. Anatini's Run timestamps supplied end-to-end duration. These remain separate because notebook launch/execution, orchestration, persistence, and static HTML retention are part of the product path but outside the internal data timers.

Reported summaries are median/minimum/maximum across the five measured Runs. No measured observation was excluded. All Runs had to pass 16 named correctness checks.

## Environment

- Windows 11 Pro build 26200
- AMD Ryzen 9 5950X, 16 cores / 32 threads
- 128 GiB physical memory
- Sabrent Rocket 4.0 2 TB NVMe
- CPython 3.14.6
- Anatini 0.9.0 native Windows runtime
- DuckDB 1.5.5
- DuckLake extension `d8a1881e`
- marimo 0.23.16
- Job profile: 6GB DuckDB memory limit, four threads

Docker, WSL, network services, and external data sources were not in the measured path.

## Important limitations

This was one high-end host, warm local storage, clean deterministic data, and one Job at a time. It does not establish behavior for ordinary laptops, cold cache, concurrent writers, failure recovery, object storage, network filesystems, managed governance, high availability, or distributed compute.

The final physical lake occupied 61.317 GiB while the current managed membership was 7.111 GiB. That ratio reflects an intentionally mutation-heavy suite with 24 total Runs, 533 snapshots, and no snapshot expiry or compaction; it is not a general storage multiplier.
