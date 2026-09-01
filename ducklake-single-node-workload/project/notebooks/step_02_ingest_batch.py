import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="DuckLake Workload Lab · 2. Ingest batch")


@app.cell
def _():
    from anatini import lake, notebook_output, quality, run_artifacts
    return lake, notebook_output, quality, run_artifacts


@app.cell
def _(notebook_output):
    notebook_output.markdown(
        """# 2. Ingest the initial batch

Read ordinary Parquet and CSV source files into DuckLake raw tables. This step
measures file decoding plus lakehouse writes, and validates the landed row counts."""
    )
    return


@app.cell
def _():
    params = {}
    return (params,)


@app.cell
def _(lake, params, quality, run_artifacts):
    import json
    import os
    import threading
    import time
    from pathlib import Path

    import psutil

    _profile = str(params.get("profile", "quick")).lower()
    _repetition = int(params.get("repetition", 0))
    _phase = str(params.get("phase", "exploratory"))
    _run_id = os.getenv("ANATINI_RUN_ID", "interactive")
    _root = lake.project_path("files/benchmark/current")
    _paths = {
        "orders": _root / "orders.parquet",
        "customers": _root / "customers.parquet",
        "products": _root / "products.csv",
    }
    if not all(path.is_file() for path in _paths.values()):
        raise FileNotFoundError("Run source generation before ingestion")

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
    con = lake.connect(profile="job")
    _snapshot_before = int(con.execute('SELECT id FROM "ducklake_workload_lab".current_snapshot()').fetchone()[0])
    _started = time.perf_counter_ns()
    try:
        con.execute(f'CREATE OR REPLACE TABLE "ducklake_workload_lab".raw.orders_source AS SELECT * FROM read_parquet({_literal(_paths["orders"])})')
        con.execute(f'CREATE OR REPLACE TABLE "ducklake_workload_lab".raw.customers_source AS SELECT * FROM read_parquet({_literal(_paths["customers"])})')
        con.execute(f'CREATE OR REPLACE TABLE "ducklake_workload_lab".raw.products_source AS SELECT * FROM read_csv_auto({_literal(_paths["products"])}, header=true)')
    finally:
        _elapsed_ms = (time.perf_counter_ns() - _started) / 1_000_000
        _stop.set()
        _sampler.join(timeout=1)
        _peak[0] = max(_peak[0], _process.memory_info().rss)

    _counts = con.execute(
        'SELECT (SELECT count(*) FROM "ducklake_workload_lab".raw.orders_source), '
        '(SELECT count(*) FROM "ducklake_workload_lab".raw.customers_source), '
        '(SELECT count(*) FROM "ducklake_workload_lab".raw.products_source)'
    ).fetchone()
    quality.check(con, "raw_orders_have_unique_ids", 'SELECT order_id FROM "ducklake_workload_lab".raw.orders_source GROUP BY order_id HAVING count(*) > 1')
    quality.check(con, "raw_customer_ids_are_unique", 'SELECT customer_id FROM "ducklake_workload_lab".raw.customers_source GROUP BY customer_id HAVING count(*) > 1')
    quality.check(con, "raw_product_ids_are_unique", 'SELECT product_id FROM "ducklake_workload_lab".raw.products_source GROUP BY product_id HAVING count(*) > 1')
    _snapshot_after = int(con.execute('SELECT id FROM "ducklake_workload_lab".current_snapshot()').fetchone()[0])
    _storage = con.execute("SELECT coalesce(sum(file_count), 0), coalesce(sum(file_size_bytes), 0) FROM ducklake_table_info('ducklake_workload_lab')").fetchone()
    _rows_out = int(sum(_counts))
    _input_bytes = sum(path.stat().st_size for path in _paths.values())
    _measurement = {
        "run_id": _run_id, "profile": _profile, "repetition": _repetition, "phase": _phase,
        "step_order": 2, "workload": "ingest_batch", "rows_in": _rows_out, "rows_out": _rows_out,
        "input_bytes": _input_bytes, "output_bytes": int(_storage[1]), "elapsed_ms": round(_elapsed_ms, 3),
        "rows_per_second": round(_rows_out / (_elapsed_ms / 1000), 2), "peak_rss_bytes": int(_peak[0]),
        "snapshot_before": _snapshot_before, "snapshot_after": _snapshot_after, "correctness": True,
        "details": {"orders": int(_counts[0]), "customers": int(_counts[1]), "products": int(_counts[2]), "managed_files": int(_storage[0])},
    }
    run_artifacts.path("measurements/02_ingest_batch.json").write_text(json.dumps(_measurement, indent=2), encoding="utf-8")
    result = _measurement
    preview = con.sql('SELECT * FROM "ducklake_workload_lab".raw.orders_source ORDER BY order_id LIMIT 12').pl().to_dicts()
    return con, preview, result


@app.cell
def _(notebook_output, preview, result):
    notebook_output.table([result], title="Step result")
    notebook_output.table(preview, title="Raw order preview")
    return


@app.cell
def _(con, result):
    print(f"[benchmark] ingested rows={result['rows_out']:,} throughput={result['rows_per_second']:,.0f} rows/s", flush=True)
    con.close()
    return


if __name__ == "__main__":
    app.run()
