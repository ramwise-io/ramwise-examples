"""Render responsive article charts from the sanitized confirmation medians."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
RESULTS = ROOT / "results" / "published_results.csv"
OUTPUT = ROOT / "article_charts"
OUTPUT.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
results = pd.read_csv(RESULTS)

scale = results[results["sweep"].isin(["window_scale", "direct_scale", "join_scale"])].copy()
figure, axis = plt.subplots(figsize=(10, 5.8))
for operation, group in scale.groupby("operation"):
    ordered = group.sort_values("points")
    axis.plot(ordered["points"], ordered["gpu_end_to_end_speedup"], marker="o", linewidth=2.2, label=operation)
axis.set_xscale("log")
axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
axis.set(xlabel="Points", ylabel="CPU time / GPU time", title="The crossover depends on the spatial operation")
axis.legend(frameon=False)
figure.tight_layout()
figure.savefig(OUTPUT / "spatial-scale-crossover.png", dpi=180, bbox_inches="tight")
plt.close(figure)

stages = results[results["sweep"] == "join_scale"].copy()
figure, axis = plt.subplots(figsize=(10, 5.8))
stages = stages.sort_values("points")
axis.plot(stages["points"], stages["gpu_resident_query_speedup"], marker="o", linewidth=2.2, label="Resident query")
axis.plot(stages["points"], stages["gpu_end_to_end_speedup"], marker="s", linewidth=2.2, label="End to end")
axis.set_xscale("log")
axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
axis.set(xlabel="Points", ylabel="CPU time / GPU time", title="Ingestion and index build move the spatial boundary")
axis.legend(frameon=False)
figure.tight_layout()
figure.savefig(OUTPUT / "spatial-lifecycle-crossover.png", dpi=180, bbox_inches="tight")
plt.close(figure)
