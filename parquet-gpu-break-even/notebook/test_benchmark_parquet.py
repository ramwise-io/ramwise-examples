from __future__ import annotations

import pyarrow.parquet as pq

from benchmark_parquet import make_table, run_benchmark, signature


def test_table_is_deterministic_and_wide() -> None:
    first = make_table(100, wide=True)
    second = make_table(100, wide=True)
    assert first.equals(second)
    assert first.num_columns == 20


def test_signature_tracks_shape_and_numeric_values() -> None:
    result = signature(make_table(10, wide=False))
    assert result["rows"] == 10
    assert result["columns"] == ["row_id", "value_i64", "value_f64", "category"]
    assert result["numeric_sums"]["row_id"] == 45


def test_cpu_benchmark_round_trip(monkeypatch) -> None:
    original = pq.read_table
    calls = []

    def recording_reader(*args, **kwargs):
        calls.append(kwargs.get("columns"))
        return original(*args, **kwargs)

    monkeypatch.setattr(pq, "read_table", recording_reader)
    result = run_benchmark(
        rows=1_000,
        row_group_rows=128,
        projection="core",
        trials=2,
        wide=True,
    )
    assert result["engines"]["pyarrow"]["median_seconds"] > 0
    assert len(result["engines"]["pyarrow"]["trials_seconds"]) == 2
    assert all(columns == ["row_id", "value_i64", "value_f64", "category"] for columns in calls)
