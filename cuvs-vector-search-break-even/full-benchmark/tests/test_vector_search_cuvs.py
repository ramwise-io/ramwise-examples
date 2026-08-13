from __future__ import annotations

import numpy as np

from experiments.vector_search_cuvs.algorithms import n_lists, search_grid
from experiments.vector_search_cuvs.common import recall_at_k, stable_id
from experiments.vector_search_cuvs.generate_dataset import normalize
from experiments.vector_search_cuvs.matrix_runner import matrix_identity, unique_confirmation_cases
from experiments.vector_search_cuvs.summarize import break_even_queries
from experiments.vector_search_cuvs.select_params import select
from experiments.vector_search_cuvs.publish_results import split


def test_recall_at_k_uses_set_overlap_per_query() -> None:
    expected = np.array([[1, 2, 3], [4, 5, 6]])
    actual = np.array([[3, 2, 9], [4, 8, 7]])
    assert recall_at_k(expected, actual, 3) == 0.5


def test_n_lists_is_bounded_power_of_two() -> None:
    for rows in (100_000, 1_000_000, 5_000_000, 100_000_000):
        value = n_lists(rows)
        assert 64 <= value <= 4096
        assert value & (value - 1) == 0


def test_search_grids_cover_exact_and_ann() -> None:
    assert search_grid("brute_force", 100_000) == [{}]
    assert {x["n_probes"] for x in search_grid("ivf_flat", 1_000_000)} >= {1, 32, 256, 1024}
    assert len(search_grid("cagra", 1_000_000)) == 25


def test_stable_id_ignores_mapping_order() -> None:
    assert stable_id("x", {"a": 1, "b": 2}) == stable_id("x", {"b": 2, "a": 1})


def test_normalize_produces_unit_rows() -> None:
    values = normalize(np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32))
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0)


def test_confirmation_cases_deduplicate_targets_with_same_params() -> None:
    selected = [
        {"batch_size": 32, "params": {"n_probes": 8}, "target_recall": 0.8},
        {"batch_size": 32, "params": {"n_probes": 8}, "target_recall": 0.9},
        {"batch_size": 32, "params": {"n_probes": 16}, "target_recall": 0.95},
    ]
    assert unique_confirmation_cases("ivf_flat", [32], selected) == [
        {"batch_size": 32, "params": {"n_probes": 8}},
        {"batch_size": 32, "params": {"n_probes": 16}},
    ]


def test_break_even_uses_per_query_savings_at_batch_size() -> None:
    assert break_even_queries(2.0, 0.010, 0.002, 32) == 8000
    assert break_even_queries(2.0, 0.002, 0.010, 32) is None


def test_matrix_identity_binds_source_commit() -> None:
    config = {"name": "full", "datasets": [{"name": "x"}]}
    assert matrix_identity(config, "a") == matrix_identity(config, "a")
    assert matrix_identity(config, "a") != matrix_identity(config, "b")


def test_selection_uses_recall_cushion_without_relabeling_target() -> None:
    result = {
        "benchmark": {"algorithm": "cagra"},
        "dataset": {"dataset_id": "x", "config": {"dimensions": 128, "rows": 1000}},
        "searches": [
            {"batch_size": 32, "params": {"search_width": 1}, "recall_at_k": 0.90, "latency_ms_per_batch": 1.0},
            {"batch_size": 32, "params": {"search_width": 2}, "recall_at_k": 0.92, "latency_ms_per_batch": 2.0},
        ],
    }
    selected = select(result, [0.90], margin=0.02)
    assert selected[0]["params"] == {"search_width": 2}
    assert selected[0]["target_recall"] == 0.90
    assert selected[0]["tuning_threshold"] == 0.92


def test_public_split_keeps_exact_and_only_confirmed_ann() -> None:
    rows = [
        {"algorithm": "brute_force", "confirmed_target_recalls": "1.00"},
        {"algorithm": "cagra", "confirmed_target_recalls": "0.90;0.95"},
        {"algorithm": "cagra", "confirmed_target_recalls": ""},
    ]
    published, controls = split(rows)
    assert len(published) == 2
    assert controls == [rows[-1]]
