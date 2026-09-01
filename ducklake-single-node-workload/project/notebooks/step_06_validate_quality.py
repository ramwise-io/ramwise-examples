import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="DuckLake Workload Lab · 6. Validate quality")


@app.cell
def _():
    from anatini import lake, notebook_output, quality, run_artifacts
    return lake, notebook_output, quality, run_artifacts


@app.cell
def _(notebook_output):
    notebook_output.markdown(
        """# 6. Validate quality and quarantine

Run explicit contracts over the modeled orders and events. Invalid source rows
remain visible in quarantine tables; nothing is silently dropped."""
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

    import psutil

    _profile = str(params.get("profile", "quick")).lower()
    _repetition = int(params.get("repetition", 0))
    _phase = str(params.get("phase", "exploratory"))
    _run_id = os.getenv("ANATINI_RUN_ID", "interactive")
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
        quality.check(con, "fact_order_ids_are_unique", 'SELECT order_id FROM "ducklake_workload_lab".analytics.fact_orders GROUP BY order_id HAVING count(*) > 1')
        quality.check(con, "fact_customers_resolve", 'SELECT f.customer_id FROM "ducklake_workload_lab".analytics.fact_orders f LEFT JOIN "ducklake_workload_lab".analytics.dim_customer d USING (customer_id) WHERE d.customer_id IS NULL LIMIT 100')
        quality.check(con, "fact_products_resolve", 'SELECT f.product_id FROM "ducklake_workload_lab".analytics.fact_orders f LEFT JOIN "ducklake_workload_lab".analytics.dim_product d USING (product_id) WHERE d.product_id IS NULL LIMIT 100')
        quality.check(con, "daily_revenue_reconciles", 'SELECT 1 WHERE (SELECT round(sum(net_amount), 2) FROM "ducklake_workload_lab".analytics.fact_orders) <> (SELECT round(sum(revenue), 2) FROM "ducklake_workload_lab".analytics.daily_channel_sales)')
        quality.check(con, "quarantine_reasons_are_known", "SELECT quarantine_reason FROM \"ducklake_workload_lab\".staging.orders_quarantine WHERE quarantine_reason NOT IN ('quantity_not_positive', 'negative_unit_price') UNION ALL SELECT quarantine_reason FROM \"ducklake_workload_lab\".staging.events_quarantine WHERE quarantine_reason <> 'missing_customer_id'")
        _summary = con.execute(
            """SELECT 'orders_accepted' AS metric, count(*)::DOUBLE AS value FROM "ducklake_workload_lab".staging.orders
               UNION ALL SELECT 'orders_quarantined', count(*)::DOUBLE FROM "ducklake_workload_lab".staging.orders_quarantine
               UNION ALL SELECT 'events_accepted', count(*)::DOUBLE FROM "ducklake_workload_lab".staging.events
               UNION ALL SELECT 'events_quarantined', count(*)::DOUBLE FROM "ducklake_workload_lab".staging.events_quarantine
               UNION ALL SELECT 'fact_revenue', round(sum(net_amount), 2)::DOUBLE FROM "ducklake_workload_lab".analytics.fact_orders
               UNION ALL SELECT 'sessions', count(*)::DOUBLE FROM "ducklake_workload_lab".analytics.sessions"""
        ).fetchall()
        con.execute(
            """CREATE OR REPLACE TABLE "ducklake_workload_lab".analytics.quality_summary AS
               SELECT * FROM (VALUES
                   ('orders_accepted', (SELECT count(*)::DOUBLE FROM "ducklake_workload_lab".staging.orders)),
                   ('orders_quarantined', (SELECT count(*)::DOUBLE FROM "ducklake_workload_lab".staging.orders_quarantine)),
                   ('events_accepted', (SELECT count(*)::DOUBLE FROM "ducklake_workload_lab".staging.events)),
                   ('events_quarantined', (SELECT count(*)::DOUBLE FROM "ducklake_workload_lab".staging.events_quarantine)),
                   ('fact_revenue', (SELECT round(sum(net_amount), 2)::DOUBLE FROM "ducklake_workload_lab".analytics.fact_orders)),
                   ('sessions', (SELECT count(*)::DOUBLE FROM "ducklake_workload_lab".analytics.sessions))
               ) metrics(metric, value)"""
        )
    finally:
        _elapsed_ms = (time.perf_counter_ns() - _started) / 1_000_000
        _stop.set()
        _sampler.join(timeout=1)
        _peak[0] = max(_peak[0], _process.memory_info().rss)

    _metric_rows = len(_summary)
    _snapshot_after = int(con.execute('SELECT id FROM "ducklake_workload_lab".current_snapshot()').fetchone()[0])
    _storage = con.execute("SELECT coalesce(sum(file_count), 0), coalesce(sum(file_size_bytes), 0) FROM ducklake_table_info('ducklake_workload_lab')").fetchone()
    _measurement = {
        "run_id": _run_id, "profile": _profile, "repetition": _repetition, "phase": _phase,
        "step_order": 6, "workload": "validate_quality", "rows_in": int(sum(row[1] for row in _summary[:4])), "rows_out": _metric_rows,
        "input_bytes": 0, "output_bytes": int(_storage[1]), "elapsed_ms": round(_elapsed_ms, 3),
        "rows_per_second": None, "peak_rss_bytes": int(_peak[0]),
        "snapshot_before": _snapshot_before, "snapshot_after": _snapshot_after, "correctness": True,
        "details": {"metrics": {row[0]: row[1] for row in _summary}, "managed_files": int(_storage[0])},
    }
    run_artifacts.path("measurements/06_validate_quality.json").write_text(json.dumps(_measurement, indent=2), encoding="utf-8")
    result = _measurement
    preview = [{"metric": row[0], "value": row[1]} for row in _summary]
    return con, preview, result


@app.cell
def _(notebook_output, preview, result):
    notebook_output.table([result], title="Step result")
    notebook_output.table(preview, title="Quality summary")
    return


@app.cell
def _(con, result):
    print(f"[benchmark] quality checks passed metrics={result['rows_out']} elapsed_ms={result['elapsed_ms']}", flush=True)
    con.close()
    return


if __name__ == "__main__":
    app.run()
