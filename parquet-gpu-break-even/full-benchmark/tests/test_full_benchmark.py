from __future__ import annotations

from argparse import Namespace
import json

import pyarrow.parquet as parquet
import polars
import pytest

from experiments.parquet_decompression.benchmark import polars_signature
from experiments.parquet_decompression.aggregate import summarize
from experiments.parquet_decompression.common import compare_signatures, dataset_id
from experiments.parquet_decompression.generate_dataset import build_table, validate_args
from experiments.parquet_decompression.matrix_runner import (
    enumerate_cases,
    matrix_identity,
    parse_cpu_set,
    source_git_commit,
)
from experiments.parquet_decompression.matrix_report import build_summary
from experiments.parquet_decompression.telemetry import summarize_samples


def test_dataset_id_is_stable() -> None:
    config = {
        "codec": "zstd",
        "profile": "mixed",
        "row_group_rows": 128,
        "rows": 1000,
        "rows_per_file": 500,
        "seed": 47,
    }
    assert dataset_id(config) == dataset_id(dict(reversed(list(config.items()))))


def test_signature_comparison_allows_small_float_reduction_noise() -> None:
    expected = {"rows": 3, "stats": {"value": {"sum": 1000.0}}}
    actual = {"rows": 3, "stats": {"value": {"sum": 1000.0000001}}}
    compare_signatures(expected, actual)


def test_build_table_round_trip(tmp_path) -> None:
    table = build_table(0, 1000, profile="mixed", seed=47)
    output = tmp_path / "sample.parquet"
    parquet.write_table(table, output, compression="zstd", row_group_size=128)
    restored = parquet.read_table(output)
    assert restored.equals(table)
    assert restored.num_rows == 1000
    assert "category" in restored.column_names


def test_generator_rejects_non_positive_sizes() -> None:
    args = Namespace(rows=0, rows_per_file=1, row_group_rows=1)
    try:
        validate_args(args)
    except ValueError as error:
        assert "--rows" in str(error)
    else:
        raise AssertionError("expected validation failure")


def test_polars_signature_widens_integer_sum() -> None:
    frame = polars.DataFrame(
        {"group_id": polars.Series([2_000_000_000, 2_000_000_000], dtype=polars.Int32)}
    )
    assert polars_signature(frame)["stats"]["group_id"]["sum"] == 4_000_000_000


def test_aggregate_preserves_run_median_range() -> None:
    def result(run_id: str, values: list[float]) -> dict:
        return {
            "run_id": run_id,
            "trials": [
                {"engine": "cudf", "phase": "measured", "throughput_mib_s": value}
                for value in values
            ],
        }

    row = summarize([result("one", [100.0, 110.0, 120.0]), result("two", [200.0, 210.0, 220.0])])[0]
    assert row["min_run_median_mib_s"] == 110.0
    assert row["max_run_median_mib_s"] == 210.0
    assert row["median_of_run_medians_mib_s"] == 160.0


def test_matrix_enumerates_smoke_and_full_case_counts() -> None:
    base = {
        "profiles": ["numeric", "mixed", "wide"],
        "codecs": ["none", "snappy", "zstd"],
        "projections": ["all", "core"],
        "benchmark": {"seed": 1000},
    }
    smoke = {**base, "replications": 1}
    full = {**base, "replications": 3}
    confirm = {
        **base,
        "profiles": ["mixed", "wide"],
        "codecs": ["snappy", "zstd"],
        "replications": 3,
    }
    rowgroup = {
        **base,
        "profiles": ["wide"],
        "codecs": ["zstd"],
        "row_group_rows": [65_536, 262_144, 1_048_576, 2_500_000],
        "generation": {"row_group_rows": 262_144},
        "replications": 3,
    }
    assert len(enumerate_cases(smoke)) == 18
    assert len(enumerate_cases(full)) == 54
    assert len(enumerate_cases(confirm)) == 24
    assert len(enumerate_cases(rowgroup)) == 24
    assert len({case["case_id"] for case in enumerate_cases(full)}) == 54
    assert {case["row_group_rows"] for case in enumerate_cases(rowgroup)} == {
        65_536,
        262_144,
        1_048_576,
        2_500_000,
    }


def test_cpu_set_parser_expands_ranges_and_rejects_empty_input() -> None:
    assert parse_cpu_set("0-3,7,9-10") == [0, 1, 2, 3, 7, 9, 10]
    with pytest.raises(ValueError, match="must not be empty"):
        parse_cpu_set("")


def test_matrix_identity_is_stable() -> None:
    config = {"name": "matrix", "profiles": ["mixed"], "schema_version": 1}
    reversed_config = dict(reversed(list(config.items())))
    assert matrix_identity(config, "commit-a") == matrix_identity(
        reversed_config, "commit-a"
    )
    assert matrix_identity(config, "commit-a") != matrix_identity(config, "commit-b")


def test_public_source_id_overrides_repository_commit(monkeypatch) -> None:
    monkeypatch.setenv("PARQUET_BENCH_SOURCE_ID", "public-bundle-commit")
    assert source_git_commit() == "public-bundle-commit"


def test_telemetry_summary_tracks_peaks() -> None:
    samples = [
        {
            "affinity_cpu_busy_seconds": 20.0,
            "affinity_cpu_total_seconds": 80.0,
            "clock_memory_mhz": 400,
            "clock_sm_mhz": 200,
            "elapsed_ns": 0,
            "host_cpu_busy_seconds": 40.0,
            "host_cpu_total_seconds": 240.0,
            "memory_used_bytes": 100,
            "performance_state": 8,
            "power_mw": 7000,
            "process_cpu_seconds": 10.0,
            "temperature_c": 35,
            "utilization_gpu_percent": 0,
            "utilization_memory_percent": 0,
        },
        {
            "affinity_cpu_busy_seconds": 20.08,
            "affinity_cpu_total_seconds": 80.16,
            "clock_memory_mhz": 9000,
            "clock_sm_mhz": 1800,
            "elapsed_ns": 20_000_000,
            "host_cpu_busy_seconds": 40.08,
            "host_cpu_total_seconds": 240.48,
            "memory_used_bytes": 200,
            "performance_state": 0,
            "power_mw": 60000,
            "process_cpu_seconds": 10.06,
            "temperature_c": 45,
            "utilization_gpu_percent": 95,
            "utilization_memory_percent": 80,
        },
    ]
    summary = summarize_samples(samples)
    assert summary["sample_count"] == 2
    assert summary["clock_sm_mhz_max"] == 1800
    assert summary["power_mw_max"] == 60000
    assert summary["performance_state_min"] == 0
    assert summary["affinity_cpu_utilization_percent_mean"] == pytest.approx(50.0)
    assert summary["process_cpu_cores_mean"] == pytest.approx(3.0)
    assert summary["background_affinity_cpu_cores_mean"] == pytest.approx(1.0)


def test_matrix_report_groups_replications(tmp_path) -> None:
    paths = []
    for run_id, values, process_cores in (
        ("one", [100.0, 120.0], 2.0),
        ("two", [200.0, 220.0], 4.0),
    ):
        path = tmp_path / f"{run_id}.json"
        path.write_text(
            json.dumps(
                {
                    "dataset": {"dataset_id": "dataset", "parquet_bytes": 1024**2},
                    "run_id": run_id,
                    "trials": [
                        {
                            "elapsed_seconds": 1.0,
                            "engine": "cudf",
                            "gpu_telemetry": {
                                "summary": {"process_cpu_cores_mean": process_cores}
                            },
                            "phase": "measured",
                            "throughput_mib_s": value,
                        }
                        for value in values
                    ],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    state = {
        "cases": {
            f"case-{index}": {
                "codec": "zstd",
                "profile": "mixed",
                "projection": "all",
                "result": str(path),
                "row_group_rows": 262_144,
                "status": "complete",
            }
            for index, path in enumerate(paths)
        }
    }
    row = build_summary(state)[0]
    assert row["replications"] == 2
    assert row["median_mib_s"] == 160.0
    assert row["run_median_min_mib_s"] == 110.0
    assert row["run_median_max_mib_s"] == 210.0
    assert row["process_cpu_cores_mean"] == 3.0
    assert row["row_group_rows"] == 262_144
