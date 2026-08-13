# Published methodology

## Hardware and software

- NVIDIA RTX PRO 4000 Blackwell SFF, 24,467 MiB
- NVIDIA driver 595.71.05; CUDA 13.1 runtime
- Intel Core Ultra 9 285HX; 24 physical CPU cores; 122 GiB RAM
- Ubuntu 26.04 host; digest-pinned Ubuntu 24.04 CUDA container
- Apache Spark 3.5.8, Scala 2.12 distribution, Java 17
- RAPIDS Accelerator 26.06.1 CUDA 13; cuDF 26.06.1
- Python 3.12, pandas 2.2.3, PyArrow 15.0.2

## Data and ETL plan

The deterministic fact table contains 22 columns: identifiers, numeric order
features, short categorical strings, text, and twelve floating-point metrics.
A 100,000-row dimension supplies customer segment, region, and risk score.
Parquet uses Zstandard and 262,144-row groups on local NVMe.

The ETL plan reads Parquet, filters amount and quantity, broadcast-joins the
dimension, derives net amount, groups by region and segment, calculates count,
sum, and averages, then sorts the forty-row output.

The scale matrix runs the native plan at 1M, 5M, 10M, 20M, and 50M fact rows.
The fallback matrix fixes 50M and inserts identity-style vectorized Python UDFs:

- after aggregation;
- after filtering;
- before filtering with 1, 4, 8, or 12 numeric inputs; and
- in two separated locations.

Every UDF input is semantically referenced while preserving the native result.

## Spark configuration and timing

Both modes load the same plugin in the same container. The controlled variable
is `spark.rapids.sql.enabled`. Important fixed settings include:

```text
master=local[16]
spark.sql.shuffle.partitions=32
spark.sql.adaptive.enabled=false
spark.sql.autoBroadcastJoinThreshold=-1
spark.sql.session.timeZone=UTC
spark.rapids.memory.pinnedPool.size=8G
spark.rapids.sql.batchSizeBytes=536870912
spark.rapids.sql.exec.ArrowEvalPythonExec=false
```

AQE is disabled so the audited physical-plan topology is stable. The dimension
is explicitly broadcast. Disabling RAPIDS replacement of ArrowEvalPythonExec
makes each UDF a deliberate visible CPU island; Python code is never described
as GPU-executed.

Each condition runs in a fresh JVM process with one warmup and two measured
actions. Three process-level replications are seeded and randomized. The timer
starts immediately before `collect()` and stops after the final CPU-backed
rows arrive. Every measured result is checked against a per-size CPU reference.

The post-execution physical plan, Spark event log, plan transition counts,
process-tree CPU/RSS, and 100 ms NVML telemetry are captured for each process.

## Accelerated Python-bridge control

RAPIDS 26.06.1 can partially accelerate the columnar transfer around
`ArrowEvalPythonExec`; it does not execute the Python UDF on the GPU. We ran a
separate 50M-row diagnostic with that replacement enabled. Seven initial cases
completed, one replication failed with a JVM SIGSEGV in `libcuda.so.1`, and a
retry completed the one-column case before the 12-column case failed with the
same fatal signature. Because the control was not stable across replications,
its timings are excluded from the public performance tables. The config and
retry-safe harness remain published so the compatibility issue is auditable
and can be retested on another driver/plugin combination.

## Public evidence and limits

The public CSVs preserve 78 process-level condition medians and ranges, or 156
measured actions. They exclude raw event logs, driver logs, private paths, and
per-sample host telemetry.

This is local Spark mode on one workstation, not a clustered Spark deployment
benchmark. Input was warm-cache local NVMe; AQE and the RAPIDS Python bridge
were deliberately disabled for plan control. Results do not generalize to
remote object stores, dynamic adaptive plans, multi-node shuffle, or every
unsupported Spark expression. NVML energy is an estimated GPU-only integral,
not wall-socket energy. Whole-device memory reflects the RMM pool and is not a
query working-set measurement.
