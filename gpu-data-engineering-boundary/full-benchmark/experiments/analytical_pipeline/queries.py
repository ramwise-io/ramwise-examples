"""Equivalent analytical workloads for DuckDB, Polars CPU/GPU, and native cuDF."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa

WORKLOADS = (
    "scan_groupby",
    "join_groupby",
    "string_features",
    "sort_topk",
    "feature_width_2",
    "feature_width_6",
    "feature_width_12",
)


def feature_columns(workload: str) -> list[str]:
    if not workload.startswith("feature_width_"):
        return []
    width = int(workload.rsplit("_", 1)[1])
    if width not in {2, 6, 12}:
        raise ValueError(f"Unsupported feature width: {width}")
    return [f"metric_{index:02d}" for index in range(width)]


def _sql_paths(paths: list[str]) -> str:
    return "[" + ",".join("'" + path.replace("'", "''") + "'" for path in paths) + "]"


def dataset_paths(manifest_path: Path, manifest: dict[str, Any]) -> tuple[list[str], str]:
    root = manifest_path.parent
    fact_paths = [str(root / item["path"]) for item in manifest["fact_files"]]
    dimension_path = str(root / manifest["dimension"]["path"])
    return fact_paths, dimension_path


def run_duckdb(
    fact_paths: list[str], dimension_path: str, workload: str, *, threads: int
) -> pa.Table:
    import duckdb

    fact = f"read_parquet({_sql_paths(fact_paths)})"
    dimension = f"read_parquet('{dimension_path.replace("'", "''")}')"
    queries = {
        "scan_groupby": f"""
            SELECT channel, status, count(*)::BIGINT AS rows,
                   sum(amount * (1.0 - discount)) AS net_amount_sum,
                   avg(quantity)::DOUBLE AS quantity_mean
            FROM {fact}
            WHERE amount >= 250.0 AND quantity >= 4
            GROUP BY channel, status
            ORDER BY channel, status
        """,
        "join_groupby": f"""
            SELECT d.region, d.segment, count(*)::BIGINT AS rows,
                   sum(f.amount) AS amount_sum,
                   avg(d.risk_score)::DOUBLE AS risk_mean
            FROM {fact} f
            INNER JOIN {dimension} d USING (customer_id)
            WHERE f.amount >= 250.0 AND f.quantity >= 4
            GROUP BY d.region, d.segment
            ORDER BY d.region, d.segment
        """,
        "string_features": f"""
            SELECT lower(status) AS normalized_status, count(*)::BIGINT AS rows,
                   sum(amount) AS amount_sum
            FROM {fact}
            WHERE contains(description, 'promo')
            GROUP BY normalized_status
            ORDER BY normalized_status
        """,
        "sort_topk": f"""
            SELECT row_id, amount, quantity, status
            FROM {fact}
            WHERE quantity >= 6
            ORDER BY amount DESC, row_id ASC
            LIMIT 1000
        """,
    }
    metrics = feature_columns(workload)
    if metrics:
        score = " + ".join(metrics)
        queries[workload] = f"""
            SELECT channel, status, count(*)::BIGINT AS rows,
                   avg(({score}) / {len(metrics)})::DOUBLE AS feature_score_mean,
                   sum(amount) AS amount_sum
            FROM {fact}
            WHERE quantity >= 4
            GROUP BY channel, status
            ORDER BY channel, status
        """
    connection = duckdb.connect()
    connection.execute(f"SET threads={threads}")
    return connection.execute(queries[workload]).fetch_arrow_table()


def polars_plan(fact_paths: list[str], dimension_path: str, workload: str):
    import polars as pl

    fact = pl.scan_parquet(fact_paths)
    selected = fact.filter((pl.col("amount") >= 250.0) & (pl.col("quantity") >= 4))
    if workload == "scan_groupby":
        return (
            selected.with_columns((pl.col("amount") * (1.0 - pl.col("discount"))).alias("net_amount"))
            .group_by(["channel", "status"])
            .agg(
                pl.len().alias("rows"),
                pl.col("net_amount").sum().alias("net_amount_sum"),
                pl.col("quantity").mean().alias("quantity_mean"),
            )
            .sort(["channel", "status"])
        )
    if workload == "join_groupby":
        dimension = pl.scan_parquet(dimension_path)
        return (
            selected.join(dimension, on="customer_id", how="inner")
            .group_by(["region", "segment"])
            .agg(
                pl.len().alias("rows"),
                pl.col("amount").sum().alias("amount_sum"),
                pl.col("risk_score").mean().alias("risk_mean"),
            )
            .sort(["region", "segment"])
        )
    if workload == "string_features":
        return (
            fact.filter(pl.col("description").str.contains("promo", literal=True))
            .with_columns(pl.col("status").str.to_lowercase().alias("normalized_status"))
            .group_by("normalized_status")
            .agg(pl.len().alias("rows"), pl.col("amount").sum().alias("amount_sum"))
            .sort("normalized_status")
        )
    if workload == "sort_topk":
        return (
            fact.filter(pl.col("quantity") >= 6)
            .select(["row_id", "amount", "quantity", "status"])
            .sort(["amount", "row_id"], descending=[True, False])
            .head(1000)
        )
    metrics = feature_columns(workload)
    if metrics:
        score = pl.col(metrics[0])
        for column in metrics[1:]:
            score = score + pl.col(column)
        return (
            fact.filter(pl.col("quantity") >= 4)
            .with_columns((score / len(metrics)).alias("feature_score"))
            .group_by(["channel", "status"])
            .agg(
                pl.len().alias("rows"),
                pl.col("feature_score").mean().alias("feature_score_mean"),
                pl.col("amount").sum().alias("amount_sum"),
            )
            .sort(["channel", "status"])
        )
    raise ValueError(f"Unknown workload: {workload}")


def run_polars_cpu(fact_paths: list[str], dimension_path: str, workload: str) -> pa.Table:
    return polars_plan(fact_paths, dimension_path, workload).collect(engine="streaming").to_arrow()


def run_polars_gpu(fact_paths: list[str], dimension_path: str, workload: str) -> pa.Table:
    import polars as pl

    engine = pl.GPUEngine(raise_on_fail=True)
    return polars_plan(fact_paths, dimension_path, workload).collect(engine=engine).to_arrow()


def run_cudf(fact_paths: list[str], dimension_path: str, workload: str) -> pa.Table:
    import cudf

    if workload == "scan_groupby":
        frame = cudf.read_parquet(
            fact_paths, columns=["channel", "status", "row_id", "amount", "quantity", "discount"]
        )
        frame = frame[(frame.amount >= 250.0) & (frame.quantity >= 4)]
        frame["net_amount"] = frame.amount * (1.0 - frame.discount)
        result = frame.groupby(["channel", "status"]).agg(
            {"row_id": "count", "net_amount": "sum", "quantity": "mean"}
        ).reset_index()
        result = result.rename(
            columns={"row_id": "rows", "net_amount": "net_amount_sum", "quantity": "quantity_mean"}
        ).sort_values(["channel", "status"])
    elif workload == "join_groupby":
        frame = cudf.read_parquet(
            fact_paths, columns=["customer_id", "row_id", "amount", "quantity"]
        )
        frame = frame[(frame.amount >= 250.0) & (frame.quantity >= 4)]
        dimension = cudf.read_parquet(dimension_path)
        joined = frame.merge(dimension, on="customer_id", how="inner")
        result = joined.groupby(["region", "segment"]).agg(
            {"row_id": "count", "amount": "sum", "risk_score": "mean"}
        ).reset_index()
        result = result.rename(
            columns={"row_id": "rows", "amount": "amount_sum", "risk_score": "risk_mean"}
        ).sort_values(["region", "segment"])
    elif workload == "string_features":
        frame = cudf.read_parquet(fact_paths, columns=["status", "description", "row_id", "amount"])
        frame = frame[frame.description.str.contains("promo")]
        frame["normalized_status"] = frame.status.str.lower()
        result = frame.groupby("normalized_status").agg(
            {"row_id": "count", "amount": "sum"}
        ).reset_index()
        result = result.rename(columns={"row_id": "rows", "amount": "amount_sum"}).sort_values(
            "normalized_status"
        )
    elif workload == "sort_topk":
        frame = cudf.read_parquet(fact_paths, columns=["row_id", "amount", "quantity", "status"])
        result = frame[frame.quantity >= 6].sort_values(
            ["amount", "row_id"], ascending=[False, True]
        ).head(1000)
    elif metrics := feature_columns(workload):
        frame = cudf.read_parquet(
            fact_paths,
            columns=["channel", "status", "row_id", "amount", "quantity", *metrics],
        )
        frame = frame[frame.quantity >= 4]
        score = frame[metrics[0]]
        for column in metrics[1:]:
            score = score + frame[column]
        frame["feature_score"] = score / len(metrics)
        result = frame.groupby(["channel", "status"]).agg(
            {"row_id": "count", "feature_score": "mean", "amount": "sum"}
        ).reset_index()
        result = result.rename(
            columns={
                "row_id": "rows",
                "feature_score": "feature_score_mean",
                "amount": "amount_sum",
            }
        ).sort_values(["channel", "status"])
    else:
        raise ValueError(f"Unknown workload: {workload}")
    return result.reset_index(drop=True).to_arrow()


def run_cudf_streaming(fact_paths: list[str], workload: str) -> pa.Table:
    """Execute width aggregation in bounded native-cuDF file partitions."""

    import cudf

    metrics = feature_columns(workload)
    if not metrics:
        raise ValueError("cudf-streaming currently supports feature-width workloads only")
    partials = []
    for path in fact_paths:
        frame = cudf.read_parquet(
            path,
            columns=["channel", "status", "row_id", "amount", "quantity", *metrics],
        )
        frame = frame[frame.quantity >= 4]
        score = frame[metrics[0]]
        for column in metrics[1:]:
            score = score + frame[column]
        frame["feature_score"] = score / len(metrics)
        partial = frame.groupby(["channel", "status"]).agg(
            {"row_id": "count", "feature_score": "sum", "amount": "sum"}
        ).reset_index()
        partials.append(
            partial.rename(
                columns={
                    "row_id": "rows",
                    "feature_score": "feature_score_sum",
                    "amount": "amount_sum",
                }
            )
        )
        del frame, partial, score

    combined = cudf.concat(partials, ignore_index=True)
    result = combined.groupby(["channel", "status"]).agg(
        {"rows": "sum", "feature_score_sum": "sum", "amount_sum": "sum"}
    ).reset_index()
    result["feature_score_mean"] = result.feature_score_sum / result.rows
    result = result.drop(columns=["feature_score_sum"]).sort_values(["channel", "status"])
    return result.reset_index(drop=True).to_arrow()


def run_engine(
    engine: str,
    fact_paths: list[str],
    dimension_path: str,
    workload: str,
    *,
    threads: int,
) -> pa.Table:
    if workload not in WORKLOADS:
        raise ValueError(f"Unknown workload: {workload}")
    if engine == "duckdb":
        return run_duckdb(fact_paths, dimension_path, workload, threads=threads)
    if engine == "polars-cpu":
        return run_polars_cpu(fact_paths, dimension_path, workload)
    if engine == "polars-gpu":
        return run_polars_gpu(fact_paths, dimension_path, workload)
    if engine == "cudf":
        return run_cudf(fact_paths, dimension_path, workload)
    if engine == "cudf-streaming":
        return run_cudf_streaming(fact_paths, workload)
    raise ValueError(f"Unknown engine: {engine}")


def result_values(table: pa.Table) -> dict[str, Any]:
    column_names = sorted(table.column_names)
    canonical = table.select(column_names)
    return {"columns": column_names, "rows": canonical.to_pylist()}
