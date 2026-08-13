from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.spatial_analytics.geometry import (
    linestring_coordinates,
    make_points,
    polygon_rings,
    quadtree_max_size,
    quadtree_scale,
)
from experiments.spatial_analytics.common import pair_signature
from experiments.spatial_analytics.matrix_runner import case_id, expand, matrix_id
from experiments.spatial_analytics.summarize import break_even_queries, confirmed_rows, excluded_rows


def test_points_are_deterministic_and_bounded() -> None:
    first = make_points(32)
    second = make_points(32)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert np.all((first[0] >= 0.0) & (first[0] < 1.0))


def test_polygon_rings_are_closed_and_control_vertex_count() -> None:
    rings = polygon_rings(7, 19)
    assert len(rings) == 7
    assert all(ring.shape == (20, 2) for ring in rings)
    assert all(np.array_equal(ring[0], ring[-1]) for ring in rings)


def test_linestrings_control_feature_and_vertex_count() -> None:
    lines = linestring_coordinates(11, 13)
    assert len(lines) == 11
    assert all(line.shape == (13, 2) for line in lines)


def test_quadtree_scale_is_positive_and_depth_sensitive() -> None:
    assert 0 < quadtree_scale(15) < quadtree_scale(12) < 1


def test_quadtree_scale_rejects_unsupported_depths() -> None:
    with pytest.raises(ValueError):
        quadtree_scale(16)


def test_quadtree_leaf_capacity_scales_with_point_count() -> None:
    assert quadtree_max_size(100_000) == 64
    assert quadtree_max_size(1_000_000) == 64
    assert quadtree_max_size(10_000_000) == 256
    assert quadtree_max_size(50_000_000) == 1024
    assert quadtree_max_size(100_000_000) == 2048


def test_pair_signature_ignores_pair_order() -> None:
    assert pair_signature(np.array([2, 1]), np.array([4, 3])) == pair_signature(
        np.array([1, 2]), np.array([3, 4])
    )


def test_matrix_expansion_changes_one_axis() -> None:
    config = {
        "defaults": {"points": 10, "features": 2},
        "sweeps": [
            {"name": "x", "operation": "indexed_join", "axis": "points", "values": [10, 20]}
        ],
    }
    cases = expand(config)
    assert [case["points"] for case in cases] == [10, 20]
    assert cases[0]["features"] == cases[1]["features"] == 2
    assert case_id(cases[0]) != case_id(cases[1])


def test_matrix_identity_binds_source_commit() -> None:
    config = {"name": "smoke", "sweeps": []}
    assert matrix_id(config, "a") != matrix_id(config, "b")


def test_reused_index_break_even_counts_fixed_cost_difference() -> None:
    cpu = {"ingest_seconds": 1.0, "index_build_seconds": 1.0, "query_seconds": 4.0}
    gpu = {"ingest_seconds": 3.0, "index_build_seconds": 3.0, "query_seconds": 2.0}
    assert break_even_queries(cpu, gpu) == 2
    gpu["query_seconds"] = 5.0
    assert break_even_queries(cpu, gpu) is None


def test_materialized_break_even_includes_return_cost() -> None:
    cpu = {"ingest_seconds": 0.0, "index_build_seconds": 0.0, "query_seconds": 4.0}
    gpu = {
        "ingest_seconds": 3.0,
        "index_build_seconds": 0.0,
        "query_seconds": 1.0,
        "materialize_seconds": 1.0,
    }
    assert break_even_queries(cpu, gpu) == 1
    assert break_even_queries(cpu, gpu, include_materialization=True) == 2


def test_public_summary_excludes_any_case_with_a_failed_replication(tmp_path) -> None:
    root = tmp_path / "matrix"
    case_dir = root / "join_scale-100000-test"
    case_dir.mkdir(parents=True)
    case = {
        "sweep": "join_scale",
        "operation": "indexed_join",
        "axis": "points",
        "axis_value": 100000,
        "points": 100000,
        "features": 16,
        "vertices": 17,
        "cpu_workers": 8,
        "warmups": 1,
        "trials": 3,
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "matrix_id": "test-matrix",
                "source_commit": "abc123",
                "config": {"replications": 3},
            }
        )
    )
    result = {
        "schema_version": 1,
        "case": case,
        "cpu": {"query_seconds": 2.0, "matches": 1, "memory": {"rss_bytes": 1, "gpu_used_bytes": 0}},
        "gpu": {"query_seconds": 1.0, "matches": 1, "memory": {"rss_bytes": 1, "gpu_used_bytes": 1}},
        "correctness": {"matches_equal": True},
    }
    for replication in (1, 2):
        (case_dir / f"replication-{replication}.json").write_text(json.dumps(result))
    (case_dir / "replication-3.failed.json").write_text(
        json.dumps({"case": case, "replication": 3, "returncode": -11, "stderr": ""})
    )

    assert confirmed_rows(root) == []
    excluded = excluded_rows(root)
    assert len(excluded) == 1
    assert excluded[0]["successful_replications"] == 2
    assert excluded[0]["failed_replications"] == 1
    assert excluded[0]["publication_status"] == "excluded_from_performance_claims"
