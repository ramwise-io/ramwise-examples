from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, value: Any, length: int = 12) -> str:
    digest = hashlib.sha256(canonical_json(value).encode()).hexdigest()[:length]
    return f"{prefix}-{digest}"


def source_id() -> str:
    configured = os.environ.get("GPU_LAB_SOURCE_ID")
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def warm_file_cache(paths: list[Path], chunk_bytes: int = 8 * 1024 * 1024) -> None:
    for path in paths:
        with path.open("rb") as handle:
            while handle.read(chunk_bytes):
                pass


def primary_quality_metric(algorithm: str) -> tuple[str, float]:
    """Return the comparison metric and maximum tolerated GPU/CPU gap."""
    return {
        "pca": ("explained_variance_ratio_sum", 0.03),
        "kmeans": ("adjusted_rand_score", 0.12),
        "logistic_regression": ("roc_auc", 0.02),
        "random_forest": ("roc_auc", 0.03),
        "umap": ("trustworthiness", 0.05),
        "hdbscan": ("adjusted_rand_score", 0.15),
        "xgboost": ("roc_auc", 0.02),
    }[algorithm]


def quality_matches(algorithm: str, cpu: dict[str, Any], candidate: dict[str, Any]) -> bool:
    metric, tolerance = primary_quality_metric(algorithm)
    return float(candidate[metric]) >= float(cpu[metric]) - tolerance
