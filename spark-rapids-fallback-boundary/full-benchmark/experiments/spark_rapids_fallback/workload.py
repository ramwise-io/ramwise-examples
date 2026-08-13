"""Build equivalent Spark ETL plans with deliberately placed Python CPU islands."""

from __future__ import annotations

from typing import Any

TOPOLOGIES = (
    "native",
    "udf_post_aggregate",
    "udf_after_filter_1",
    "udf_before_filter_1",
    "udf_before_filter_4",
    "udf_before_filter_8",
    "udf_before_filter_12",
    "udf_two_islands",
)


def udf_width(topology: str) -> int:
    if topology.startswith("udf_before_filter_"):
        return int(topology.rsplit("_", 1)[1])
    if topology in {"udf_after_filter_1", "udf_two_islands"}:
        return 1
    return 0


def _identity_udf(width: int):
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import DoubleType

    def combine(*columns):
        output = columns[0].astype("float64", copy=False)
        for column in columns[1:]:
            output = output + column.astype("float64", copy=False) * 0.0
        return output

    if width == 1:
        def identity(c0):
            return combine(c0)
    elif width == 4:
        def identity(c0, c1, c2, c3):
            return combine(c0, c1, c2, c3)
    elif width == 8:
        def identity(c0, c1, c2, c3, c4, c5, c6, c7):
            return combine(c0, c1, c2, c3, c4, c5, c6, c7)
    elif width == 12:
        def identity(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11):
            return combine(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)
    else:
        raise ValueError(f"Unsupported UDF width: {width}")
    identity.__name__ = f"identity_width_{width}"
    return pandas_udf(identity, returnType=DoubleType())


def udf_input_columns(width: int):
    """Return exactly *width* numeric inputs, with amount always first."""

    from pyspark.sql import functions as F

    if width < 1 or width > 12:
        raise ValueError("UDF width must be between 1 and 12")
    metrics = [f"metric_{index:02d}" for index in range(11)]
    return [F.col("amount"), *[F.col(name) for name in metrics[: width - 1]]]


def build_query(spark: Any, fact_path: str, dimension_path: str, topology: str):
    from pyspark.sql import functions as F

    if topology not in TOPOLOGIES:
        raise ValueError(f"Unknown topology: {topology}")

    fact = spark.read.parquet(fact_path)

    if topology.startswith("udf_before_filter_") or topology == "udf_two_islands":
        width = udf_width(topology)
        identity = _identity_udf(width)
        inputs = udf_input_columns(width)
        fact = fact.withColumn("filter_amount", identity(*inputs))
    else:
        fact = fact.withColumn("filter_amount", F.col("amount"))

    filtered = fact.filter((F.col("filter_amount") >= 250.0) & (F.col("quantity") >= 4))

    if topology == "udf_after_filter_1":
        filtered = filtered.withColumn("amount_for_etl", _identity_udf(1)(F.col("amount")))
    else:
        filtered = filtered.withColumn("amount_for_etl", F.col("amount"))

    dimension = spark.read.parquet(dimension_path)
    joined = filtered.join(F.broadcast(dimension), "customer_id", "inner")

    if topology == "udf_two_islands":
        joined = joined.withColumn("risk_for_etl", _identity_udf(1)(F.col("risk_score")))
    else:
        joined = joined.withColumn("risk_for_etl", F.col("risk_score"))

    result = (
        joined.withColumn(
            "net_amount", F.col("amount_for_etl") * (F.lit(1.0) - F.col("discount"))
        )
        .groupBy("region", "segment")
        .agg(
            F.count(F.lit(1)).alias("rows"),
            F.sum("net_amount").alias("net_amount_sum"),
            F.avg("quantity").alias("quantity_mean"),
            F.avg("risk_for_etl").alias("risk_mean"),
        )
    )

    if topology == "udf_post_aggregate":
        result = result.withColumn("net_amount_sum", _identity_udf(1)(F.col("net_amount_sum")))

    return result.orderBy("region", "segment")


def canonical_rows(rows: list[Any]) -> dict[str, Any]:
    return {
        "rows": [
            {
                "net_amount_sum": float(row["net_amount_sum"]),
                "quantity_mean": float(row["quantity_mean"]),
                "region": row["region"],
                "risk_mean": float(row["risk_mean"]),
                "rows": int(row["rows"]),
                "segment": row["segment"],
            }
            for row in rows
        ]
    }


def plan_summary(plan: str) -> dict[str, Any]:
    lines = [line.strip() for line in plan.splitlines() if line.strip()]
    gpu_lines = [line for line in lines if "Gpu" in line]
    python_lines = [line for line in lines if "Python" in line]
    transition_names = (
        "GpuColumnarToRow",
        "GpuRowToColumnar",
        "ColumnarToRow",
        "RowToColumnar",
    )
    transitions = [line for line in lines if any(name in line for name in transition_names)]
    return {
        "gpu_plan_lines": len(gpu_lines),
        "has_gpu": bool(gpu_lines),
        "has_python": bool(python_lines),
        "operator_lines": lines,
        "python_plan_lines": len(python_lines),
        "transition_lines": transitions,
        "transitions": len(transitions),
    }
