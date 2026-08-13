"""Generate article figures directly from the sanitized published evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd


ROOT = Path(__file__).parent
RESULTS = ROOT / "results" / "published_results.csv"
OUTPUT = ROOT / "article_charts"

ENGINE_ORDER = ["polars-cpu", "duckdb", "polars-gpu", "cudf"]
COLORS = ["#6b7280", "#2563eb", "#16a34a", "#f59e0b"]
LABELS = {
    "polars-cpu": "Polars CPU",
    "duckdb": "DuckDB",
    "polars-gpu": "Polars GPU",
    "cudf": "cuDF",
}
WORKLOADS = {
    "join_groupby": "Join + group-by",
    "feature_width_12": "Feature width 12",
    "scan_groupby": "Scan + group-by",
    "feature_width_2": "Feature width 2",
    "feature_width_6": "Feature width 6",
    "sort_topk": "Filtered top-k",
    "string_features": "String features",
}


def main() -> None:
    results = pd.read_csv(RESULTS)
    crossover = results[results["study"] == "crossover"]
    winners = (
        crossover.sort_values("median_seconds")
        .groupby(["workload", "rows"], as_index=False)
        .first()
    )
    pivot = winners.pivot(index="workload", columns="rows", values="engine").loc[list(WORKLOADS)]
    numeric = pivot.map(ENGINE_ORDER.index)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.82, bottom=0.22)
    ax.imshow(numeric, cmap=ListedColormap(COLORS), aspect="auto", vmin=-0.5, vmax=3.5)
    ax.set_xticks(range(len(pivot.columns)), [f"{value / 1e6:g}M" for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [WORKLOADS[value] for value in pivot.index])
    ax.set_xlabel("Rows")
    fig.suptitle(
        "The fastest engine changed with both query shape and dataset size",
        x=0.13, ha="left", fontsize=16, weight="bold",
    )
    fig.text(
        0.13, 0.865,
        "Warm-cache Parquet-to-Arrow median; three measured trials per condition",
        ha="left", fontsize=9.5, color="#4b5563",
    )

    for row in range(numeric.shape[0]):
        for column in range(numeric.shape[1]):
            engine = pivot.iloc[row, column]
            ax.text(column, row, LABELS[engine], ha="center", va="center", color="white", fontsize=8.5, weight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS]
    ax.legend(handles, [LABELS[name] for name in ENGINE_ORDER], ncol=4, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))

    OUTPUT.mkdir(exist_ok=True)
    destination = OUTPUT / "gpu-query-winner-map.png"
    fig.savefig(destination, dpi=180, facecolor="white", bbox_inches="tight")
    print(destination)


if __name__ == "__main__":
    main()
