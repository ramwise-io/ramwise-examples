"""Build the output-ready analysis notebook from the published result tables."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).parent
OUTPUT = ROOT / "gpu_data_engineering_boundary.ipynb"


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
# Where GPU data engineering starts to pay off

This output-complete notebook analyzes an end-to-end comparison of DuckDB CPU,
Polars CPU streaming, Polars GPU with CPU fallback disabled, and native cuDF.
It also preserves the failed native-cuDF larger-than-VRAM control.

The measured boundary begins before Parquet input and ends after a CPU-backed
Arrow result is materialized. Every accepted timing matched the canonical
reference result.
"""
    ),
    code(
        """
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
RESULTS = Path("results/published_results.csv")
FAILED = Path("results/failed_controls.csv")
results = pd.read_csv(RESULTS)
failed = pd.read_csv(FAILED)
results.groupby("study").size().rename("derived conditions").to_frame()
"""
    ),
    markdown(
        """
## Experimental boundary

- Hardware: NVIDIA RTX PRO 4000 Blackwell SFF, 24 GB VRAM; eight CPU threads.
- Software: Python 3.12.13, DuckDB 1.5.5, Polars 1.42.1, cuDF 26.08.00,
  PyArrow 23.0.1, in a pinned CUDA 13.1 container.
- Data: deterministic synthetic Parquet, 262,144-row groups, warm-cache trials.
- Evidence: 196 engine/workload conditions and 588 measured timings, three per
  condition, plus warmups and strict result checks.
- Scope: local NVMe input and CPU-backed output. These are workstation boundary
  measurements, not universal framework rankings.

The public tables contain derived medians, ranges, process RSS, utilization,
and correctness status. Raw host telemetry and infrastructure identifiers stay
in the private experiment workspace.
"""
    ),
    code(
        """
assert results["correct"].all()
assert results["trials"].min() >= 3
assert len(results) == 196
assert int(results["trials"].sum()) == 588
results.groupby(["study", "workload"])["engine"].nunique().unstack(fill_value=0)
"""
    ),
    markdown(
        """
## 1. Crossover: size changes the winner

All engines perform the same end-to-end work. Log scales keep 100,000 through
50 million rows readable together.
"""
    ),
    code(
        """
crossover = results[results["study"] == "crossover"].copy()
core_names = ["scan_groupby", "join_groupby", "string_features", "sort_topk"]
core = crossover[crossover["workload"].isin(core_names)]
grid = sns.relplot(
    data=core, x="rows", y="median_seconds", hue="engine",
    col="workload", col_wrap=2, kind="line", marker="o",
    facet_kws={"sharey": False}, height=3.4, aspect=1.35,
)
grid.set(xscale="log", yscale="log")
grid.set_axis_labels("Rows", "Median end-to-end seconds")
grid.figure.suptitle("End-to-end analytical pipeline crossover", y=1.03)
plt.show()
"""
    ),
    markdown(
        """
## 2. Width means columns actually processed

Adding unused columns is not a width benchmark because projection removes
them. These workloads calculate a feature score from 2, 6, or 12 metric
columns before aggregation.
"""
    ),
    code(
        """
width = crossover[crossover["workload"].str.startswith("feature_width_")].copy()
width["metrics"] = width["workload"].str.rsplit("_", n=1).str[-1].astype(int)
largest = width[width["rows"] == 50_000_000]
chart = sns.catplot(
    data=largest, x="metrics", y="median_seconds", hue="engine",
    kind="bar", height=4.5, aspect=1.5,
)
chart.set_axis_labels("Metric columns processed", "Median end-to-end seconds")
chart.figure.suptitle("Width sensitivity at 50 million rows", y=1.02)
plt.show()
"""
    ),
    markdown("## 3. Measured crossover boundaries"),
    code(
        """
winners = (
    crossover.sort_values("median_seconds")
    .groupby(["rows", "workload"], as_index=False)
    .first()[["rows", "workload", "engine", "median_seconds"]]
)
winners.pivot(index="workload", columns="rows", values="engine")
"""
    ),
    code(
        """
gpu_engines = {"polars-gpu", "cudf"}
boundaries = []
for workload, group in winners.groupby("workload"):
    gpu_rows = group.loc[group["engine"].isin(gpu_engines), "rows"]
    boundaries.append({
        "workload": workload,
        "first_measured_GPU_win": (
            f"{int(gpu_rows.min()):,} rows" if not gpu_rows.empty else "No GPU win through 50,000,000"
        ),
    })
pd.DataFrame(boundaries).set_index("workload")
"""
    ),
    markdown(
        """
## 4. Parquet codec is part of the workload

This controlled 20-million-row matrix changes only Snappy versus Zstandard.
Codec effects are shown per engine and query; there is no single codec winner
independent of the execution path.
"""
    ),
    code(
        """
codec = results[results["study"] == "codec"].copy()
grid = sns.catplot(
    data=codec, x="engine", y="median_seconds", hue="codec",
    col="workload", kind="bar", sharey=False, height=3.8, aspect=1.15,
)
grid.set_axis_labels("", "Median end-to-end seconds")
grid.set_xticklabels(rotation=25)
grid.figure.suptitle("Codec sensitivity at 20 million rows", y=1.04)
plt.show()
"""
    ),
    markdown(
        """
## 5. Larger than VRAM: execution strategy matters

The 200-million-row wide dataset is 34.6 GB logical and 19.7 GB compressed,
or about 1.34 times the nominal 24 GiB device capacity. Polars GPU completed
with its streaming executor and CPU fallback disabled.

Native cuDF's one-shot read is retained below as a failed control. The separate
`cudf-streaming` condition is an explicit bounded algorithm: read one file at a
time, compute exact GPU partial sums/counts, then merge the tiny aggregates. It
is not evidence that arbitrary cuDF programs become out-of-core automatically.
"""
    ),
    code(
        """
ooc = results[results["study"] == "out_of_core"].copy()
ooc["median_seconds"] = ooc["median_seconds"].round(3)
ooc[["engine", "median_seconds", "relative_to_fastest", "process_rss_peak_bytes"]].sort_values("median_seconds")
"""
    ),
    code(
        """
failed[["engine", "outcome", "reason"]]
"""
    ),
    markdown(
        """
## 6. Memory and utilization

Peak RSS is the benchmark process resident set; GPU utilization is sampled
through NVML. Whole-device memory is intentionally omitted from public engine
comparisons because unrelated resident services and allocator reuse prevent
honest per-process attribution.
"""
    ),
    code(
        """
memory = results.assign(rss_peak_gib=results["process_rss_peak_bytes"] / 1024**3)
memory.groupby(["study", "engine"])[["rss_peak_gib", "gpu_utilization_mean_percent"]].max().round(2)
"""
    ),
    markdown(
        """
## What this evidence supports

1. **Projection is the first optimization.** Width mattered only when queries
   actually consumed the extra columns.
2. **GPU payoff is a curve, not a label.** In this matrix the first GPU win was
   5 million rows for the join, 10 million for the 12-column feature workload,
   and 20 million for the scan and 2/6-column feature workloads.
3. **Strings and filtered top-k stayed CPU territory through 50 million rows.**
   DuckDB won both at the largest measured size.
4. **A GPU-capable plan can exceed VRAM when it streams.** Polars GPU was fastest
   on the 34.6 GB logical dataset, but native cuDF's one-shot read failed and
   the manual cuDF partition loop was about nine times slower.
5. **Codec and execution engine interact.** Snappy versus Zstandard changed
   timings and occasionally the CPU winner; choose with the actual query and
   storage constraints, not a universal codec rule.
6. **End-to-end boundaries are conservative by design.** If the next stage
   remains GPU-resident, avoiding the final CPU materialization may move the
   break-even point earlier; remote storage, cold cache, and tiny-file overhead
   may move it elsewhere.

The practical rule is: reduce columns early, keep a CPU reference, benchmark
the complete query at representative sizes, and use GPU execution once enough
parallel work survives I/O and setup overhead. For data beyond VRAM, require a
verified streaming or partitioning plan—"spilling enabled" is not itself a
guarantee.
"""
    ),
]

nbf.write(notebook, OUTPUT)
print(OUTPUT)
