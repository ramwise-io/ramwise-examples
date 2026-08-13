"""Generate article figures from the sanitized cuVS evidence."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent
RESULTS = ROOT / "results" / "published_results.csv"
OUTPUT = ROOT / "article_charts"
PALETTE = {
    "brute_force": "#374151",
    "ivf_flat": "#2563eb",
    "ivf_pq_compact": "#f59e0b",
    "ivf_pq_balanced": "#d97706",
    "cagra": "#16a34a",
}


def main() -> None:
    results = pd.read_csv(RESULTS)
    OUTPUT.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    scale = results[(results["dimensions"] == 128) & (results["batch_size"] == 32)].copy()
    scale = scale.sort_values("latency_ms_per_batch_median").groupby(["rows", "algorithm"], as_index=False).first()
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    sns.lineplot(data=scale, x="rows", y="latency_ms_per_query_median", hue="algorithm", marker="o", palette=PALETTE, ax=ax)
    ax.set(xscale="log", yscale="log", xlabel="Corpus vectors", ylabel="Median milliseconds per query")
    ax.set_title("The exact-versus-ANN boundary moved with corpus size", loc="left", weight="bold")
    ax.legend(title="", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "cuvs-search-scale.png", dpi=180, facecolor="white")
    plt.close(fig)

    finite = results[(results["break_even_status"] == "finite") & (results["dimensions"] == 128) & (results["batch_size"] == 32)].copy()
    finite["highest confirmed recall"] = finite["confirmed_target_recalls"].str.split(";").str[-1]
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    sns.scatterplot(data=finite, x="rows", y="break_even_queries", hue="algorithm", style="highest confirmed recall", s=125, palette=PALETTE, ax=ax)
    ax.set(xscale="log", yscale="log", xlabel="Corpus vectors", ylabel="Queries required to amortize build")
    ax.set_title("Faster search still had to repay index construction", loc="left", weight="bold")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT / "cuvs-build-break-even.png", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
