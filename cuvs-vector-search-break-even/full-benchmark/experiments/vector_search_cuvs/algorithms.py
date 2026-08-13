from __future__ import annotations

import math
from typing import Any

import numpy as np
from cuvs.neighbors import brute_force, cagra, ivf_flat, ivf_pq

ALGORITHMS = ("brute_force", "ivf_flat", "ivf_pq_compact", "ivf_pq_balanced", "cagra")


def n_lists(rows: int) -> int:
    target = max(64, min(4096, int(math.sqrt(rows))))
    return 1 << int(round(math.log2(target)))


def build(algorithm: str, dataset: Any, rows: int, dimensions: int, resources: Any = None):
    lists = n_lists(rows)
    if algorithm == "brute_force":
        return brute_force.build(dataset, metric="sqeuclidean", resources=resources), {"metric": "sqeuclidean"}
    if algorithm == "ivf_flat":
        params = ivf_flat.IndexParams(n_lists=lists, metric="sqeuclidean", kmeans_trainset_fraction=min(0.5, 2_000_000 / rows))
        return ivf_flat.build(params, dataset, resources=resources), {"n_lists": lists}
    if algorithm in {"ivf_pq_compact", "ivf_pq_balanced"}:
        pq_dim = max(16, dimensions // 4) if algorithm == "ivf_pq_compact" else max(32, dimensions // 2)
        params = ivf_pq.IndexParams(n_lists=lists, metric="sqeuclidean", pq_bits=8, pq_dim=pq_dim, kmeans_trainset_fraction=min(0.5, 2_000_000 / rows), conservative_memory_allocation=True)
        return ivf_pq.build(params, dataset, resources=resources), {"n_lists": lists, "pq_bits": 8, "pq_dim": pq_dim}
    if algorithm == "cagra":
        params = cagra.IndexParams(metric="sqeuclidean", intermediate_graph_degree=128, graph_degree=64, build_algo="ivf_pq")
        return cagra.build(params, dataset, resources=resources), {"graph_degree": 64, "intermediate_graph_degree": 128, "build_algo": "ivf_pq"}
    raise ValueError(algorithm)


def search_grid(algorithm: str, rows: int) -> list[dict[str, Any]]:
    if algorithm == "brute_force":
        return [{}]
    if algorithm in {"ivf_flat", "ivf_pq_compact", "ivf_pq_balanced"}:
        maximum = n_lists(rows)
        return [{"n_probes": value} for value in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096) if value <= maximum]
    return [
        {"itopk_size": topk, "search_width": width}
        for topk in (32, 64, 128, 256, 512)
        for width in (1, 2, 4, 8, 16)
    ]


def search(algorithm: str, index: Any, queries: Any, k: int, params: dict[str, Any], resources: Any):
    if algorithm == "brute_force":
        return brute_force.search(index, queries, k, resources=resources)
    if algorithm == "ivf_flat":
        return ivf_flat.search(ivf_flat.SearchParams(**params), index, queries, k, resources=resources)
    if algorithm in {"ivf_pq_compact", "ivf_pq_balanced"}:
        return ivf_pq.search(ivf_pq.SearchParams(**params), index, queries, k, resources=resources)
    if algorithm == "cagra":
        return cagra.search(cagra.SearchParams(**params), index, queries, k, resources=resources)
    raise ValueError(algorithm)


def save(algorithm: str, filename: str, index: Any) -> None:
    module = {"brute_force": brute_force, "ivf_flat": ivf_flat, "ivf_pq_compact": ivf_pq, "ivf_pq_balanced": ivf_pq, "cagra": cagra}[algorithm]
    module.save(filename, index, include_dataset=True)
