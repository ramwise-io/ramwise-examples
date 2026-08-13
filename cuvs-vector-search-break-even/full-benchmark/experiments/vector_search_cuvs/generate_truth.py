"""Compute and persist exact cuVS neighbors once for an experiment dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np
from cuvs.common import Resources
from cuvs.neighbors import brute_force

from experiments.vector_search_cuvs.common import atomic_json, read_json
from experiments.vector_search_cuvs.generate_dataset import digest_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        print(json.dumps({"truth_manifest": str(args.output)}, indent=2))
        return
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    corpus = cp.asarray(np.load(manifest_path.parent / manifest["corpus"]["path"], mmap_mode="r"))
    queries = cp.asarray(np.load(manifest_path.parent / manifest["queries"]["path"], mmap_mode="r"))
    resources = Resources()
    started = time.perf_counter()
    index = brute_force.build(corpus, metric="sqeuclidean", resources=resources)
    distances, neighbors = brute_force.search(index, queries, args.k, resources=resources)
    resources.sync()
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    neighbors_path = args.output.parent / "neighbors.npy"
    distances_path = args.output.parent / "distances.npy"
    np.save(neighbors_path, cp.asarray(neighbors).get())
    np.save(distances_path, cp.asarray(distances).get())
    truth = {
        "schema_version": 1,
        "dataset_id": manifest["dataset_id"],
        "k": args.k,
        "queries": int(len(queries)),
        "compute_seconds": elapsed,
        "method": "cuvs.brute_force sqeuclidean",
        "neighbors": {"path": neighbors_path.name, "sha256": digest_file(neighbors_path)},
        "distances": {"path": distances_path.name, "sha256": digest_file(distances_path)},
    }
    atomic_json(args.output, truth)
    print(json.dumps({"truth_manifest": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
