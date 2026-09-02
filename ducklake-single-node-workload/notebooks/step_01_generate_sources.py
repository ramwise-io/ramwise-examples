import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="DuckLake Workload Lab · 1. Generate sources")


@app.cell
def _():
    from anatini import lake, notebook_output, run_artifacts
    return lake, notebook_output, run_artifacts


@app.cell
def _(notebook_output):
    notebook_output.markdown(
        """# 1. Generate deterministic source files

Create the same fictional commerce workload at each declared scale. The files are
ordinary project data: Parquet for orders/customers, CSV for products and changes,
and newline-delimited JSON for behavioral events."""
    )
    return


@app.cell
def _():
    params = {}
    return (params,)


@app.cell
def _(lake, params, run_artifacts):
    import json
    import os
    import threading
    import time
    from pathlib import Path

    import psutil

    _profiles = {
        "quick": {"orders": 100_000, "changes": 5_000, "events": 200_000, "customers": 10_000, "products": 1_000},
        "small": {"orders": 1_000_000, "changes": 50_000, "events": 2_000_000, "customers": 100_000, "products": 10_000},
        "medium": {"orders": 10_000_000, "changes": 500_000, "events": 20_000_000, "customers": 1_000_000, "products": 100_000},
        "large": {"orders": 50_000_000, "changes": 2_500_000, "events": 100_000_000, "customers": 5_000_000, "products": 500_000},
    }
    _profile = str(params.get("profile", "quick")).lower()
    if _profile not in _profiles:
        raise ValueError(f"profile must be one of {sorted(_profiles)}")
    _spec = _profiles[_profile]
    _repetition = int(params.get("repetition", 0))
    _phase = str(params.get("phase", "exploratory"))
    _run_id = os.getenv("ANATINI_RUN_ID", "interactive")
    _source_root = lake.project_path("files/benchmark/current")
    _source_root.mkdir(parents=True, exist_ok=True)
    _paths = {
        "orders": _source_root / "orders.parquet",
        "customers": _source_root / "customers.parquet",
        "products": _source_root / "products.csv",
        "changes": _source_root / "order_changes.csv",
        "events": _source_root / "events.jsonl",
    }
    for _path in _paths.values():
        if _path.exists():
            _path.unlink()

    def _literal(path: Path) -> str:
        return "'" + str(path).replace("'", "''") + "'"

    _process = psutil.Process()
    _peak = [_process.memory_info().rss]
    _stop = threading.Event()

    def _sample_memory():
        while not _stop.wait(0.05):
            _peak[0] = max(_peak[0], _process.memory_info().rss)

    _sampler = threading.Thread(target=_sample_memory, daemon=True)
    _sampler.start()
    _started = time.perf_counter_ns()
    con = lake.connect(profile="job")
    _snapshot_before = int(con.execute('SELECT id FROM "ducklake_workload_lab".current_snapshot()').fetchone()[0])
    try:
        con.execute(
            f"""COPY (
                SELECT i AS order_id,
                       (i * 17) % {_spec['customers']} AS customer_id,
                       (i * 31) % {_spec['products']} AS product_id,
                       TIMESTAMP '2026-01-01 00:00:00' + (i % 7776000) * INTERVAL 1 SECOND AS order_ts,
                       CASE i % 5 WHEN 0 THEN 'web' WHEN 1 THEN 'store' WHEN 2 THEN 'partner' WHEN 3 THEN 'mobile' ELSE 'marketplace' END AS channel,
                       CASE WHEN i % 50000 = 0 THEN 0 ELSE 1 + (i % 5) END AS quantity,
                       round(5 + ((i * 37) % 25000) / 100.0, 2) AS unit_price,
                       round((i % 7) * 0.01, 2) AS discount_rate,
                       CASE i % 4 WHEN 0 THEN 'east' WHEN 1 THEN 'west' WHEN 2 THEN 'north' ELSE 'south' END AS region
                FROM range({_spec['orders']}) rows(i)
            ) TO {_literal(_paths['orders'])} (FORMAT PARQUET, COMPRESSION SNAPPY)"""
        )
        con.execute(
            f"""COPY (
                SELECT i AS customer_id,
                       printf('Customer %08d', i) AS customer_name,
                       CASE i % 4 WHEN 0 THEN 'consumer' WHEN 1 THEN 'small_business' WHEN 2 THEN 'mid_market' ELSE 'enterprise' END AS segment,
                       CASE i % 4 WHEN 0 THEN 'east' WHEN 1 THEN 'west' WHEN 2 THEN 'north' ELSE 'south' END AS region
                FROM range({_spec['customers']}) rows(i)
            ) TO {_literal(_paths['customers'])} (FORMAT PARQUET, COMPRESSION SNAPPY)"""
        )
        con.execute(
            f"""COPY (
                SELECT i AS product_id,
                       printf('SKU-%07d', i) AS sku,
                       CASE i % 6 WHEN 0 THEN 'home' WHEN 1 THEN 'office' WHEN 2 THEN 'outdoor' WHEN 3 THEN 'electronics' WHEN 4 THEN 'apparel' ELSE 'grocery' END AS category,
                       round(4 + ((i * 29) % 30000) / 100.0, 2) AS list_price
                FROM range({_spec['products']}) rows(i)
            ) TO {_literal(_paths['products'])} (FORMAT CSV, HEADER)"""
        )
        con.execute(
            f"""COPY (
                SELECT CASE WHEN i % 10 = 0 THEN 'delete' WHEN i % 10 < 4 THEN 'update' ELSE 'insert' END AS operation,
                       CASE WHEN i % 10 < 4 THEN (i * 7919) % {_spec['orders']} ELSE {_spec['orders']} + i END AS order_id,
                       (i * 23) % {_spec['customers']} AS customer_id,
                       (i * 43) % {_spec['products']} AS product_id,
                       TIMESTAMP '2026-04-01 00:00:00' + (i % 604800) * INTERVAL 1 SECOND AS order_ts,
                       CASE i % 3 WHEN 0 THEN 'web' WHEN 1 THEN 'mobile' ELSE 'partner' END AS channel,
                       1 + (i % 5) AS quantity,
                       round(8 + ((i * 41) % 20000) / 100.0, 2) AS unit_price,
                       round((i % 5) * 0.01, 2) AS discount_rate,
                       CASE i % 4 WHEN 0 THEN 'east' WHEN 1 THEN 'west' WHEN 2 THEN 'north' ELSE 'south' END AS region
                FROM range({_spec['changes']}) rows(i)
            ) TO {_literal(_paths['changes'])} (FORMAT CSV, HEADER)"""
        )
        con.execute(
            f"""COPY (
                SELECT i AS event_id,
                       CASE WHEN i % 10000 = 0 THEN NULL ELSE (i * 19) % {_spec['customers']} END AS customer_id,
                       TIMESTAMP '2026-04-01 00:00:00' + (i % 604800) * INTERVAL 1 SECOND AS event_ts,
                       CASE i % 5 WHEN 0 THEN 'page_view' WHEN 1 THEN 'search' WHEN 2 THEN 'add_to_cart' WHEN 3 THEN 'checkout' ELSE 'purchase' END AS event_type,
                       {{'session_id': printf('S%010d', floor(i / 7)::BIGINT), 'device': CASE i % 3 WHEN 0 THEN 'mobile' WHEN 1 THEN 'desktop' ELSE 'tablet' END, 'campaign': CASE WHEN i % 4 = 0 THEN 'spring' ELSE NULL END}} AS properties
                FROM range({_spec['events']}) rows(i)
            ) TO {_literal(_paths['events'])} (FORMAT JSON, ARRAY false)"""
        )
    finally:
        _elapsed_ms = (time.perf_counter_ns() - _started) / 1_000_000
        _stop.set()
        _sampler.join(timeout=1)
        _peak[0] = max(_peak[0], _process.memory_info().rss)

    _snapshot_after = int(con.execute('SELECT id FROM "ducklake_workload_lab".current_snapshot()').fetchone()[0])
    _source_bytes = sum(path.stat().st_size for path in _paths.values())
    _measurement = {
        "run_id": _run_id,
        "profile": _profile,
        "repetition": _repetition,
        "phase": _phase,
        "step_order": 1,
        "workload": "generate_sources",
        "rows_in": 0,
        "rows_out": sum(_spec.values()),
        "input_bytes": 0,
        "output_bytes": _source_bytes,
        "elapsed_ms": round(_elapsed_ms, 3),
        "rows_per_second": round(sum(_spec.values()) / (_elapsed_ms / 1000), 2),
        "peak_rss_bytes": int(_peak[0]),
        "snapshot_before": _snapshot_before,
        "snapshot_after": _snapshot_after,
        "correctness": True,
        "details": {"seed": 20260901, **_spec},
    }
    run_artifacts.path("measurements/01_generate_sources.json").write_text(json.dumps(_measurement, indent=2), encoding="utf-8")
    result = {**_measurement, "source_paths": {key: f"files/benchmark/current/{path.name}" for key, path in _paths.items()}}
    preview = [{"source": key, "rows": _spec.get(key, _spec.get("changes") if key == "changes" else None), "bytes": path.stat().st_size} for key, path in _paths.items()]
    return con, preview, result


@app.cell
def _(notebook_output, preview, result):
    notebook_output.table([result], title="Step result")
    notebook_output.table(preview, title="Generated files")
    return


@app.cell
def _(con, result):
    print(f"[benchmark] generated sources profile={result['profile']} rows={result['rows_out']:,} elapsed_ms={result['elapsed_ms']}", flush=True)
    con.close()
    return


if __name__ == "__main__":
    app.run()
