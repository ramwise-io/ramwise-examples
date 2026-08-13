"""Build the output-ready Spark RAPIDS fallback analysis notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent
OUTPUT = ROOT / "spark_rapids_fallback_boundary.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
notebook["cells"] = [
    markdown(
        """
# When partial Spark GPU acceleration stops helping

This output-complete notebook analyzes the same realistic PySpark ETL plan on
CPU and with the RAPIDS Accelerator. It then inserts one or two deliberate
Python CPU islands at different plan locations and varies how many columns
cross the boundary.

The central question is not whether Spark has GPU operators. It is whether
enough useful GPU work survives fallback and host/device transitions to beat
the equivalent complete CPU plan.
"""
    ),
    code(
        """
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
scale = pd.read_csv(Path("results/scale_results.csv"))
fallback = pd.read_csv(Path("results/fallback_results.csv"))
len(scale), len(fallback), int(scale["trials"].sum() + fallback["trials"].sum())
"""
    ),
    markdown(
        """
## Experimental envelope

- Spark 3.5.8, RAPIDS Accelerator 26.06.1 for CUDA 13, Java 17.
- NVIDIA RTX PRO 4000 Blackwell SFF, 24 GB; Intel Core Ultra 9 285HX.
- One local Spark process, 16 worker threads, 32 shuffle partitions.
- Deterministic wide Zstandard Parquet fact data plus a 100,000-row dimension.
- Filter, broadcast join, projection, grouped aggregates, and final sort.
- Warm OS cache; one warmup and two measured actions per process; three
  independently started process replications per condition.
- Adaptive query execution disabled so physical-plan topology remains stable.
- Every accepted result matched a per-size CPU reference.

The CPU and GPU conditions load the same plugin in the same container. Only
`spark.rapids.sql.enabled` differs. For the main fallback matrix, RAPIDS
replacement of `ArrowEvalPythonExec` is disabled deliberately so each Pandas
UDF is a visible CPU island.
"""
    ),
    code(
        """
assert scale["correct"].all() and fallback["correct"].all()
assert len(scale) == 30 and len(fallback) == 48
assert int(scale["trials"].sum() + fallback["trials"].sum()) == 156
scale.groupby(["rows", "mode"])["replication"].nunique().unstack()
"""
    ),
    markdown("## 1. Native Spark/RAPIDS crossover"),
    code(
        """
scale_agg = (
    scale.groupby(["rows", "mode"], as_index=False)
    .agg(median_seconds=("median_seconds", "median"),
         replication_min=("median_seconds", "min"),
         replication_max=("median_seconds", "max"))
)
chart = sns.lineplot(
    data=scale_agg, x="rows", y="median_seconds", hue="mode", marker="o"
)
chart.set_xscale("log")
chart.set_xlabel("Fact rows")
chart.set_ylabel("Median action seconds across replications")
chart.set_title("The native GPU plan crosses over near 20 million rows")
plt.show()
"""
    ),
    code(
        """
scale_table = scale_agg.pivot(index="rows", columns="mode", values="median_seconds")
scale_table["gpu_speedup"] = scale_table["cpu"] / scale_table["gpu"]
scale_table.round(3)
"""
    ),
    markdown(
        """
CPU won through 10 million rows. The GPU reached practical parity at 20
million (1.05×) and won by 1.14× at 50 million in the scale matrix. Those are
measured points, not an exact universal crossover curve.
"""
    ),
    markdown("## 2. Moving one CPU island through the plan"),
    code(
        """
labels = {
    "native": "Native",
    "udf_post_aggregate": "UDF after aggregate",
    "udf_after_filter_1": "UDF after filter",
    "udf_before_filter_1": "UDF before filter",
    "udf_before_filter_4": "Pre-filter UDF, 4 cols",
    "udf_before_filter_8": "Pre-filter UDF, 8 cols",
    "udf_before_filter_12": "Pre-filter UDF, 12 cols",
    "udf_two_islands": "Two CPU islands",
}
order = list(labels)
agg = (
    fallback.groupby(["topology", "mode"], as_index=False)
    .agg(median_seconds=("median_seconds", "median"),
         replication_min=("median_seconds", "min"),
         replication_max=("median_seconds", "max"),
         transitions=("transitions", "median"))
)
agg["label"] = agg["topology"].map(labels)
agg["label"] = pd.Categorical(agg["label"], [labels[x] for x in order], ordered=True)
plot = sns.catplot(
    data=agg, y="label", x="median_seconds", hue="mode",
    kind="bar", height=5.3, aspect=1.5,
)
plot.set_axis_labels("Median action seconds", "")
plot.figure.suptitle("Fallback placement changes the value of partial acceleration", y=1.02)
plt.show()
"""
    ),
    code(
        """
comparison = agg.pivot(index="topology", columns="mode", values="median_seconds")
comparison["gpu_speedup"] = comparison["cpu"] / comparison["gpu"]
comparison["gpu_transitions"] = (
    agg[agg["mode"] == "gpu"].set_index("topology")["transitions"]
)
comparison.loc[order].rename(index=labels).round(3)
"""
    ),
    markdown(
        """
The native 50-million-row GPU plan won by 1.22× in this randomized matrix.
With one CPU island after aggregation it still won by 1.19×; after the filter,
only 1.05×. Moving the same one-column UDF before the selective filter produced
practical parity (0.98×). Two separated CPU islands created five transition
nodes and made the hybrid plan 0.75× as fast as CPU—about 33% slower.
"""
    ),
    markdown("## 3. Width was not the main boundary"),
    code(
        """
width_order = ["udf_before_filter_1", "udf_before_filter_4", "udf_before_filter_8", "udf_before_filter_12"]
width = fallback[fallback["topology"].isin(width_order)].copy()
width_agg = width.groupby(["udf_width", "mode"], as_index=False)["median_seconds"].median()
sns.lineplot(data=width_agg, x="udf_width", y="median_seconds", hue="mode", marker="o")
plt.xlabel("Numeric columns entering the pre-filter Pandas UDF")
plt.ylabel("Median action seconds")
plt.title("A wider single CPU island slowed both plans")
plt.show()
"""
    ),
    markdown(
        """
A single wider UDF slowed CPU and GPU modes together. Hybrid GPU execution
still beat the equivalent CPU plan at widths 4, 8, and 12. This means the
experiment does **not** support a simple rule such as “N fallback columns make
GPU execution uneconomic.” Placement and repeated boundaries mattered more
than width alone for this ETL shape.
"""
    ),
    markdown("## 4. The physical-plan anatomy"),
    code(
        """
fallback[fallback["mode"] == "gpu"].groupby("topology")[[
    "gpu_plan_lines", "python_plan_lines", "transitions"
]].median().loc[order]
"""
    ),
    markdown(
        """
Representative topology:

```text
One CPU island
GpuFileSourceScan → GpuColumnarToRow → ArrowEvalPython (CPU)
                  → GpuRowToColumnar → GpuFilter → GpuBroadcastHashJoin
                  → GpuHashAggregate → GpuSort → final host collection

Two CPU islands
Gpu scan → CPU Python → GPU filter/join → CPU Python
         → GPU aggregate/sort → final host collection
```

The final host collection appears even in the native GPU plan, so comparisons
use transition *topology*, not the presence of any transition at all.
"""
    ),
    markdown("## 5. Resource context"),
    code(
        """
fallback.groupby(["topology", "mode"])[[
    "gpu_utilization_mean_percent", "gpu_energy_joules_estimate",
    "process_tree_cpu_cores_mean", "process_tree_rss_max_bytes"
]].median().round(2).loc[order]
"""
    ),
    markdown(
        """
GPU energy is a trapezoidal estimate from NVML power samples, not wall-socket
energy. NVML memory is whole-device allocation context: RAPIDS reserves most
of the device in an RMM pool, so it must not be interpreted as the query's live
working set. CPU and RSS telemetry includes the PySpark driver, JVM, and Python
worker descendants.
"""
    ),
    markdown(
        """
## What this evidence supports

1. **Inspect the executed plan, not the configuration.** A loaded accelerator
   and `spark.rapids.sql.enabled=true` do not prove end-to-end GPU execution.
2. **Find the native scale boundary first.** On this strong workstation CPU,
   the tested ETL needed roughly 20 million rows before GPU parity.
3. **Fallback after cardinality reduction can be cheap.** A CPU UDF over forty
   aggregate rows preserved most of the GPU advantage.
4. **Fallback before a selective operator is much more expensive.** The same
   one-column UDF before filtering erased the native advantage.
5. **Count boundaries, but also locate them.** Two separated CPU islands were
   decisively worse than CPU-only, while one wider island was not.
6. **Compare the hybrid plan with the equivalent CPU plan.** Wider UDFs slowed
   both modes; comparing every hybrid condition only with native CPU would
   confuse UDF cost with transition cost.

The practical rule is: use the qualification tools to find unsupported nodes,
then benchmark the executed mixed plan at representative scale. Push CPU-only
logic after filters and aggregations when possible, avoid repeated GPU↔CPU
islands, and keep a CPU-only escape hatch when the physical plan fragments.
"""
    ),
    markdown(
        """
## Excluded accelerated-bridge control

RAPIDS 26.06.1 enables partial acceleration of the Arrow backend around scalar
Pandas UDFs by default. A separate 50-million-row diagnostic restored that
setting. Seven initial cases completed, but one JVM terminated with a SIGSEGV
in `libcuda.so.1`; a retry completed the one-column case before the 12-column
case failed with the same signature. Those unstable timings are intentionally
excluded from the evidence above. The control config and retry-safe harness are
included under `full-benchmark/` for retesting on another driver/plugin stack.
"""
    ),
]

nbf.write(notebook, OUTPUT)
print(OUTPUT)
