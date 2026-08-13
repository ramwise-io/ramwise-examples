from __future__ import annotations

from experiments.spark_rapids_fallback.common import compare_values, stable_id
from experiments.spark_rapids_fallback.matrix_runner import (
    benchmark_command,
    cases,
    matrix_identity,
)
from experiments.spark_rapids_fallback.summarize import summarize_results
from experiments.spark_rapids_fallback.workload import plan_summary, udf_width


def test_stable_id_ignores_mapping_order() -> None:
    assert stable_id("x", {"a": 1, "b": 2}) == stable_id("x", {"b": 2, "a": 1})


def test_compare_values_allows_reduction_noise() -> None:
    compare_values({"x": [1000.0]}, {"x": [1000.000001]})


def test_udf_width_parses_topologies() -> None:
    assert udf_width("native") == 0
    assert udf_width("udf_before_filter_12") == 12
    assert udf_width("udf_two_islands") == 1


def test_plan_summary_separates_python_and_transitions() -> None:
    summary = plan_summary(
        "GpuFileSourceScanExec\nGpuColumnarToRow\nArrowEvalPython\nGpuRowToColumnar\nGpuHashAggregate"
    )
    assert summary["has_gpu"]
    assert summary["has_python"]
    assert summary["transitions"] == 2


def test_summary_keeps_replications_separate() -> None:
    result = {
        "run_id": "r",
        "dataset": {
            "config": {"rows": 100},
            "parquet_bytes": 50,
            "estimated_logical_bytes": 100,
        },
        "benchmark": {"topology": "native", "mode": "gpu", "replication": 2},
        "trials": [
            {
                "phase": "measured",
                "correct": True,
                "elapsed_seconds": 1.0,
                "plan": {"gpu_plan_lines": 2, "python_plan_lines": 0, "transitions": 0},
                "telemetry": {"summary": {"gpu_utilization_percent_mean": 50}},
            }
        ],
    }
    row = summarize_results([result])[0]
    assert row["replication"] == 2
    assert row["median_seconds"] == 1.0


def test_matrix_cases_are_complete_and_deterministic() -> None:
    config = {
        "name": "x",
        "rows": [100],
        "topologies": ["native", "udf_before_filter_1"],
        "modes": ["cpu", "gpu"],
        "python_bridges": ["forced-cpu"],
        "replications": 2,
        "benchmark": {"seed": 9},
    }
    assert len(cases(config)) == 8
    assert cases(config) == cases(config)
    assert matrix_identity(config, "commit") == matrix_identity(
        dict(reversed(list(config.items()))), "commit"
    )


def test_retry_attempt_is_forwarded_to_benchmark() -> None:
    config = {
        "benchmark": {
            "threads": 16,
            "shuffle_partitions": 32,
            "telemetry_interval_ms": 100,
        }
    }
    command = benchmark_command(
        config,
        manifest="manifest.json",
        reference=None,
        output_root="raw",
        mode="gpu",
        topology="native",
        python_bridge="forced-cpu",
        replication=0,
        seed=1,
        warmups=1,
        trials=2,
        retry_attempt=2,
    )
    assert command[-2:] == ["--retry-attempt", "2"]
