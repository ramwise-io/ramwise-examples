from __future__ import annotations

from argparse import Namespace

import pyarrow as pa

from experiments.analytical_pipeline.common import compare_values, dataset_id
from experiments.analytical_pipeline.generate_dataset import (
    build_dimension_table,
    build_fact_table,
    validate_args,
)
from experiments.analytical_pipeline.matrix_runner import enumerate_cases, matrix_identity
from experiments.analytical_pipeline.queries import feature_columns, result_values
from experiments.analytical_pipeline.summarize import summarize_results


def test_dataset_id_is_stable() -> None:
    config = {
        "codec": "zstd",
        "profile": "wide",
        "row_group_rows": 262_144,
        "rows": 1_000_000,
        "rows_per_file": 1_000_000,
        "seed": 73,
    }
    assert dataset_id(config) == dataset_id(dict(reversed(list(config.items()))))


def test_fact_and_dimension_keys_overlap() -> None:
    fact = build_fact_table(0, 10_000, profile="wide", seed=73)
    dimension = build_dimension_table(seed=73)
    assert fact.num_rows == 10_000
    assert dimension.num_rows == 100_000
    assert max(fact.column("customer_id").to_pylist()) < dimension.num_rows
    assert "metric_11" in fact.column_names


def test_base_profile_omits_extra_metrics() -> None:
    table = build_fact_table(0, 100, profile="base", seed=73)
    assert not any(name.startswith("metric_") for name in table.column_names)


def test_generator_rejects_non_positive_sizes() -> None:
    args = Namespace(rows=0, rows_per_file=1, row_group_rows=1)
    try:
        validate_args(args)
    except ValueError as error:
        assert "--rows" in str(error)
    else:
        raise AssertionError("expected validation failure")


def test_result_values_preserves_order() -> None:
    table = pa.table({"value": [1.0, 2.0], "key": ["a", "b"]})
    assert result_values(table) == {
        "columns": ["key", "value"],
        "rows": [{"key": "a", "value": 1.0}, {"key": "b", "value": 2.0}],
    }


def test_compare_values_allows_reduction_noise() -> None:
    compare_values({"rows": [{"sum": 1000.0}]}, {"rows": [{"sum": 1000.000001}]})


def test_feature_width_workloads_select_real_metric_columns() -> None:
    assert feature_columns("feature_width_2") == ["metric_00", "metric_01"]
    assert len(feature_columns("feature_width_12")) == 12
    assert feature_columns("scan_groupby") == []


def test_out_of_core_matrix_names_manual_cudf_streaming() -> None:
    import json
    from pathlib import Path

    config = json.loads(
        (Path(__file__).parents[1] / "configs/out_of_core.json").read_text()
    )
    assert "cudf-streaming" in config["benchmark"]["engines"]
    assert "cudf" not in config["benchmark"]["engines"]


def test_summary_preserves_engine_medians_and_relative_time() -> None:
    result = {
        "run_id": "run",
        "dataset": {
            "config": {"profile": "wide", "codec": "zstd", "rows": 1000},
            "parquet_bytes": 500,
            "estimated_logical_bytes": 1000,
        },
        "trials": [],
    }
    for engine, values in (("cpu", [1.0, 1.2, 1.1]), ("gpu", [0.4, 0.5, 0.6])):
        for value in values:
            result["trials"].append(
                {
                    "correct": True,
                    "elapsed_seconds": value,
                    "engine": engine,
                    "gpu_telemetry": {"summary": {"sample_count": 1}},
                    "phase": "measured",
                    "workload": "scan_groupby",
                }
            )
    rows = summarize_results([result])
    cpu = next(row for row in rows if row["engine"] == "cpu")
    gpu = next(row for row in rows if row["engine"] == "gpu")
    assert cpu["median_seconds"] == 1.1
    assert gpu["median_seconds"] == 0.5
    assert cpu["relative_to_fastest"] == 2.2
    assert cpu["device_memory_used_max_bytes"] == 0
    assert cpu["process_rss_peak_bytes"] == 0


def test_matrix_enumeration_and_identity_are_stable() -> None:
    config = {
        "name": "pilot",
        "profiles": ["base", "wide"],
        "codecs": ["zstd"],
        "rows": [100, 1000],
        "benchmark": {"seed": 90},
    }
    cases = enumerate_cases(config)
    assert len(cases) == 4
    assert len({case["case_id"] for case in cases}) == 4
    assert matrix_identity(config, "commit") == matrix_identity(
        dict(reversed(list(config.items()))), "commit"
    )
