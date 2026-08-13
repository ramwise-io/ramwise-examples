"""Small, correctness-checked Parquet CPU/GPU comparison for the companion post."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def make_table(rows: int, *, wide: bool, seed: int = 47) -> pa.Table:
    row_id = np.arange(rows, dtype=np.int64)
    columns: dict[str, Any] = {
        "row_id": row_id,
        "value_i64": (row_id * 17 + seed * 101) % 1_000_003,
        "value_f64": np.sin((row_id + seed) * 0.001) * 1000.0,
        "category": pa.array(
            np.asarray(["alpha", "bravo", "charlie", "delta"], dtype=object)[
                row_id % 4
            ]
        ),
    }
    if wide:
        for index in range(16):
            columns[f"metric_{index:02d}"] = np.cos(
                (row_id + seed) * (index + 1) * 0.0001
            )
    return pa.table(columns)


def signature(table: pa.Table) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": table.num_rows, "columns": table.column_names}
    sums: dict[str, float | int] = {}
    for field, column in zip(table.schema, table.columns, strict=True):
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            sums[field.name] = pc.sum(column).as_py()
    result["numeric_sums"] = sums
    return result


def timed_trials(
    reader: Callable[[], Any],
    normalize: Callable[[Any], pa.Table],
    synchronize: Callable[[], None],
    expected: dict[str, Any],
    trials: int,
) -> list[float]:
    warm = reader()  # import/session/cache warmup
    synchronize()
    del warm
    elapsed: list[float] = []
    for _ in range(trials):
        started = time.perf_counter()
        frame = reader()
        synchronize()
        seconds = time.perf_counter() - started
        table = normalize(frame)
        actual = signature(table)
        if actual["rows"] != expected["rows"] or actual["columns"] != expected["columns"]:
            raise AssertionError("reader returned the wrong shape")
        for name, value in expected["numeric_sums"].items():
            if not np.isclose(actual["numeric_sums"][name], value, rtol=1e-9):
                raise AssertionError(f"reader returned an incorrect sum for {name}")
        elapsed.append(seconds)
    return elapsed


def run_benchmark(
    *, rows: int, row_group_rows: int, projection: str, trials: int, wide: bool
) -> dict[str, Any]:
    table = make_table(rows, wide=wide)
    columns = table.column_names if projection == "all" else [
        "row_id",
        "value_i64",
        "value_f64",
        "category",
    ]
    expected = signature(table.select(columns))
    with tempfile.TemporaryDirectory(prefix="ramwise-parquet-") as directory:
        path = Path(directory) / "sample.parquet"
        pq.write_table(table, path, compression="zstd", row_group_size=row_group_rows)
        del table

        readers: dict[
            str, tuple[Callable[[], Any], Callable[[Any], pa.Table], Callable[[], None]]
        ] = {
            "pyarrow": (
                lambda: pq.read_table(path, columns=columns, use_threads=True),
                lambda frame: frame,
                lambda: None,
            )
        }
        try:
            import cudf
            import cupy

            readers["cudf"] = (
                lambda: cudf.read_parquet(path, columns=columns),
                lambda frame: frame.to_arrow(),
                cupy.cuda.runtime.deviceSynchronize,
            )
        except ImportError:
            pass

        engines: dict[str, Any] = {}
        for name, (reader, normalize, synchronize) in readers.items():
            values = timed_trials(reader, normalize, synchronize, expected, trials)
            engines[name] = {
                "median_seconds": statistics.median(values),
                "trials_seconds": values,
            }

        return {
            "engines": engines,
            "file_mib": path.stat().st_size / 1024**2,
            "projection": projection,
            "row_group_rows": row_group_rows,
            "rows": rows,
            "wide": wide,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--row-group-rows", type=int, default=262_144)
    parser.add_argument("--projection", choices=["all", "core"], default="core")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--narrow", action="store_true", help="omit the 16 extra metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows < 1 or args.row_group_rows < 1 or args.trials < 1:
        raise ValueError("rows, row-group-rows, and trials must be positive")
    result = run_benchmark(
        rows=args.rows,
        row_group_rows=args.row_group_rows,
        projection=args.projection,
        trials=args.trials,
        wide=not args.narrow,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
