"""Generate deterministic dense data for the GPU ML estimator matrix."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gpu_ml_pipeline import EXPERIMENT_NAME, SCHEMA_VERSION
from experiments.gpu_ml_pipeline.common import atomic_json, read_json, source_id, stable_id


def dataset_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "rows": args.rows,
        "features": args.features,
        "test_fraction": args.test_fraction,
        "clusters": args.clusters,
        "informative_features": min(args.informative_features, args.features),
        "cluster_scale": args.cluster_scale,
        "seed": args.seed,
        "dtype": "float32",
    }


def validate_config(config: dict[str, Any]) -> None:
    if int(config["rows"]) < 100:
        raise ValueError("rows must be at least 100")
    if int(config["features"]) < 2:
        raise ValueError("features must be at least 2")
    if not 0.05 <= float(config["test_fraction"]) <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5")
    if int(config["clusters"]) < 2:
        raise ValueError("clusters must be at least 2")


def write_split(
    output: Path,
    name: str,
    rows: int,
    features: int,
    informative: int,
    centers: np.ndarray,
    seed: int,
    chunk_rows: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    x_path = output / f"X_{name}.npy"
    y_path = output / f"y_{name}.npy"
    cluster_path = output / f"clusters_{name}.npy"
    x = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(rows, features))
    y = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int8, shape=(rows,))
    cluster_labels = np.lib.format.open_memmap(
        cluster_path, mode="w+", dtype=np.int16, shape=(rows,)
    )
    positive = 0
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        count = stop - start
        labels = rng.integers(0, len(centers), size=count, dtype=np.int16)
        values = rng.normal(0.0, 1.0, size=(count, features)).astype(np.float32)
        values[:, :informative] += centers[labels]
        logits = (
            1.25 * values[:, 0]
            - 0.85 * values[:, 1]
            + 0.55 * values[:, 2 % informative]
            + rng.normal(0.0, 0.75, size=count)
        )
        targets = (logits > 0).astype(np.int8)
        x[start:stop] = values
        y[start:stop] = targets
        cluster_labels[start:stop] = labels
        positive += int(targets.sum())
    x.flush()
    y.flush()
    cluster_labels.flush()
    del x, y, cluster_labels
    return {
        "rows": rows,
        "positive_rows": positive,
        "files": {
            "X": {"path": x_path.name, "bytes": x_path.stat().st_size},
            "y": {"path": y_path.name, "bytes": y_path.stat().st_size},
            "clusters": {"path": cluster_path.name, "bytes": cluster_path.stat().st_size},
        },
    }


def existing_is_valid(output: Path, config: dict[str, Any]) -> bool:
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = read_json(manifest_path)
        if manifest["config"] != config:
            return False
        for split in manifest["splits"].values():
            for item in split["files"].values():
                path = output / item["path"]
                if not path.exists() or path.stat().st_size != int(item["bytes"]):
                    return False
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument("--informative-features", type=int, default=8)
    parser.add_argument("--cluster-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument(
        "--output-root", type=Path, default=Path("/data/generated/gpu-ml-pipeline")
    )
    args = parser.parse_args()
    config = dataset_config(args)
    validate_config(config)
    identity = {"experiment": EXPERIMENT_NAME, "generator_version": 1, "config": config}
    dataset_id = stable_id(f"r{args.rows}-f{args.features}", identity)
    output = args.output_root / dataset_id
    if existing_is_valid(output, config):
        print(json.dumps({"dataset_id": dataset_id, "manifest": str(output / "manifest.json"), "reused": True}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    informative = int(config["informative_features"])
    center_rng = np.random.default_rng(args.seed)
    centers = center_rng.normal(
        0.0, args.cluster_scale, size=(args.clusters, informative)
    ).astype(np.float32)
    test_rows = max(1, round(args.rows * args.test_fraction))
    train_rows = args.rows - test_rows
    splits = {
        "train": write_split(
            output, "train", train_rows, args.features, informative, centers,
            args.seed + 1, args.chunk_rows,
        ),
        "test": write_split(
            output, "test", test_rows, args.features, informative, centers,
            args.seed + 2, args.chunk_rows,
        ),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "dataset_id": dataset_id,
        "config": config,
        "splits": splits,
        "git_commit": source_id(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"dataset_id": dataset_id, "manifest": str(output / "manifest.json"), "reused": False}, indent=2))


if __name__ == "__main__":
    main()

