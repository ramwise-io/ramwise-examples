"""Build the output-ready analysis notebook from published spatial results."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent
OUTPUT = ROOT / "gpu_spatial_analytics_crossover.ipynb"


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
# Where does GPU spatial analytics cross over?

This output-complete notebook compares an eight-thread Shapely/STRtree CPU
baseline with cuSpatial 25.04 on one RTX PRO 4000 Blackwell SFF. It asks how
point count, feature count, polygon complexity, selectivity, index build, and
result placement move the boundary.
"""
    ),
    code(
        """
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
RESULTS = Path("results/published_results.csv")
EXCLUDED = Path("results/excluded_cases.csv")
results = pd.read_csv(RESULTS)
excluded = pd.read_csv(EXCLUDED)
results.shape, excluded.shape
"""
    ),
    markdown(
        """
## Evidence checks

Every performance row is a median of three fresh-process replications. CPU and
GPU match counts agree, point/polygon pair signatures agree, and nearest-line
rows are published only where every pointwise distance passed the recorded
planar tolerance. Failed correctness controls remain visible in the diagnostic
table but are not converted into speedups.
"""
    ),
    code(
        """
assert results["replications"].eq(3).all()
assert results["gpu_end_to_end_speedup"].gt(0).all()
assert results["case_id"].is_unique
assert excluded["publication_status"].eq("excluded_from_performance_claims").all()
results.groupby("operation").size(), excluded.groupby("operation").size()
"""
    ),
    markdown("## Point scale changes both fixed cost and useful parallel work"),
    code(
        """
scale = results[results["sweep"].isin(["window_scale", "direct_scale", "join_scale"])].copy()
figure, axis = plt.subplots(figsize=(9, 5))
for operation, group in scale.groupby("operation"):
    ordered = group.sort_values("points")
    axis.plot(ordered["points"], ordered["gpu_end_to_end_speedup"], marker="o", label=operation)
axis.set_xscale("log")
axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
axis.set(xlabel="Points", ylabel="CPU time / GPU time", title="End-to-end spatial crossover")
axis.legend()
figure.tight_layout()
plt.show()
"""
    ),
    markdown("## Polygon count, vertices, and selectivity move different stages"),
    code(
        """
axes = results[results["sweep"].isin(["join_features", "join_vertices", "join_selectivity"])].copy()
figure, panels = plt.subplots(1, 3, figsize=(13, 3.8))
for panel, (sweep, group) in zip(panels, axes.groupby("sweep")):
    ordered = group.sort_values("axis_value")
    panel.plot(ordered["axis_value"], ordered["gpu_end_to_end_speedup"], marker="o")
    panel.axhline(1.0, color="black", linewidth=1, linestyle="--")
    panel.set(title=sweep, xlabel=ordered["axis"].iloc[0], ylabel="CPU time / GPU time")
figure.tight_layout()
plt.show()
"""
    ),
    markdown("## Index reuse is a lifecycle, not one universal number"),
    code(
        """
reuse = results[results["operation"] == "indexed_join"].copy()
reuse[[
    "sweep", "axis_value", "reused_index_break_even_queries_median",
    "materialized_reuse_break_even_queries_median", "gpu_resident_query_speedup",
]].sort_values(["sweep", "axis_value"]).head(20)
"""
    ),
    markdown("## Correctness boundaries are part of the result"),
    code(
        """
excluded[[
    "sweep", "axis_value", "successful_replications", "failed_replications",
    "returncodes", "failure_summary", "publication_status",
]].sort_values(["sweep", "axis_value"])
"""
    ),
    code(
        """
import json

quadtree = pd.DataFrame(json.loads(Path("results/quadtree_diagnostic.json").read_text()))
nearest_100k = pd.DataFrame(json.loads(Path("results/nearest_100k_diagnostic.json").read_text()))
nearest_1m = pd.DataFrame(json.loads(Path("results/nearest_1m_diagnostic.json").read_text()))
quadtree, nearest_100k[["multiplier", "maximum_absolute_error", "mismatches"]], nearest_1m[["multiplier", "maximum_absolute_error", "mismatches"]]
"""
    ),
    markdown(
        """
## Scope

The data is deterministic synthetic planar geometry in `[0, 1)^2`. The study
does not cover geographic CRS transforms, invalid real-world polygons, file
parsing, spherical distance, dynamic index updates, or multi-GPU execution.
The notebook analyzes derived confirmation evidence; rerunning it does not
rerun the GPU benchmark.
"""
    ),
]

nbf.write(notebook, OUTPUT)
print(OUTPUT)
