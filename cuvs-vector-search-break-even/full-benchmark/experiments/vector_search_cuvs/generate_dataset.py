"""Generate a deterministic clustered embedding corpus and query set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.vector_search_cuvs.common import atomic_json, stable_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--dimensions", type=int, required=True)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--clusters", type=int, default=4096)
    parser.add_argument("--noise", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--output-root", type=Path, default=Path("/data/generated/vector-search-cuvs"))
    return parser.parse_args()


def digest_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def normalize(values: np.ndarray) -> np.ndarray:
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def main() -> None:
    args = parse_args()
    if min(args.rows, args.dimensions, args.queries, args.clusters) < 1:
        raise ValueError("rows, dimensions, queries, and clusters must be positive")
    config = {
        "generator_version": 2,
        "rows": args.rows,
        "dimensions": args.dimensions,
        "queries": args.queries,
        "clusters": min(args.clusters, args.rows),
        "noise": args.noise,
        "seed": args.seed,
        "dtype": "float32",
        "distribution": "normalized-clustered-gaussian",
    }
    dataset_id = stable_id("vectors", config)
    output = args.output_root / dataset_id
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        print(json.dumps({"dataset_id": dataset_id, "manifest": str(manifest_path)}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    center_rng = np.random.default_rng(args.seed)
    corpus_rng = np.random.default_rng(args.seed + 1)
    query_rng = np.random.default_rng(args.seed + 2)
    centers = normalize(center_rng.standard_normal((config["clusters"], args.dimensions), dtype=np.float32))
    corpus_path = output / "corpus.npy"
    corpus = np.lib.format.open_memmap(corpus_path, mode="w+", dtype=np.float32, shape=(args.rows, args.dimensions))
    for start in range(0, args.rows, args.chunk_rows):
        stop = min(start + args.chunk_rows, args.rows)
        labels = corpus_rng.integers(0, config["clusters"], size=stop - start)
        chunk = centers[labels].copy()
        chunk += corpus_rng.standard_normal(chunk.shape, dtype=np.float32) * args.noise
        corpus[start:stop] = normalize(chunk)
    corpus.flush()
    del corpus
    query_labels = query_rng.integers(0, config["clusters"], size=args.queries)
    queries = centers[query_labels].copy()
    queries += query_rng.standard_normal(queries.shape, dtype=np.float32) * args.noise
    queries = normalize(queries)
    query_path = output / "queries.npy"
    np.save(query_path, queries)
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "config": config,
        "corpus": {"path": corpus_path.name, "bytes": corpus_path.stat().st_size, "sha256": digest_file(corpus_path)},
        "queries": {"path": query_path.name, "bytes": query_path.stat().st_size, "sha256": digest_file(query_path)},
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({"dataset_id": dataset_id, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
