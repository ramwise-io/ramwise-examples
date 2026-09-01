import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="DuckLake Workload Lab · 3. Prepare and model")


@app.cell
def _():
    from anatini import lake, notebook_output, quality, run_artifacts
    return lake, notebook_output, quality, run_artifacts


@app.cell
def _(notebook_output):
    notebook_output.markdown(
        """# 3. Prepare and model

Separate invalid orders, calculate a typed business amount, join customer and
product dimensions, and publish daily/channel metrics with a running revenue window."""
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
    _rows_in = int(con.execute('SELECT count(*) FROM "ducklake_workload_lab".raw.orders_source').fetchone()[0])
    _started = time.perf_counter_ns()
    try:
        con.execute(
            """CREATE OR REPLACE TABLE "ducklake_workload_lab".staging.orders_quarantine AS
               SELECT *, CASE WHEN quantity <= 0 THEN 'quantity_not_positive'
                              WHEN unit_price < 0 THEN 'negative_unit_price'
                              ELSE 'unknown' END AS quarantine_reason
               FROM "ducklake_workload_lab".raw.orders_source
               WHERE quantity <= 0 OR unit_price < 0"""
        )
        con.execute(
            """CREATE OR REPLACE TABLE "ducklake_workload_lab".staging.orders AS
               SELECT order_id::BIGINT AS order_id, customer_id::BIGINT AS customer_id,
                      product_id::BIGINT AS product_id, order_ts::TIMESTAMP AS order_ts,
                      channel::VARCHAR AS channel, quantity::INTEGER AS quantity,
                      unit_price::DECIMAL(18,2) AS unit_price,
                      discount_rate::DECIMAL(9,4) AS discount_rate, region::VARCHAR AS region,
                      round(quantity * unit_price * (1 - discount_rate), 2)::DECIMAL(18,2) AS net_amount
               FROM "ducklake_workload_lab".raw.orders_source
               WHERE quantity > 0 AND unit_price >= 0"""
        )
        con.execute('CREATE OR REPLACE TABLE "ducklake_workload_lab".analytics.dim_customer AS SELECT * FROM "ducklake_workload_lab".raw.customers_source')
        con.execute('CREATE OR REPLACE TABLE "ducklake_workload_lab".analytics.dim_product AS SELECT * FROM "ducklake_workload_lab".raw.products_source')
        con.execute(
            """CREATE OR REPLACE TABLE "ducklake_workload_lab".analytics.fact_orders AS
               SELECT o.*, c.segment, p.category
               FROM "ducklake_workload_lab".staging.orders o
               JOIN "ducklake_workload_lab".analytics.dim_customer c USING (customer_id)
               JOIN "ducklake_workload_lab".analytics.dim_product p USING (product_id)"""
        )
        con.execute(
            """CREATE OR REPLACE TABLE "ducklake_workload_lab".analytics.daily_channel_sales AS
               WITH daily AS (
                   SELECT order_ts::DATE AS order_date, channel, region, count(*) AS orders,
                          count(DISTINCT customer_id) AS customers, round(sum(net_amount), 2) AS revenue
                   FROM "ducklake_workload_lab".analytics.fact_orders
                   GROUP BY ALL
               )
               SELECT *, round(sum(revenue) OVER (PARTITION BY channel, region ORDER BY order_date), 2) AS running_revenue
               FROM daily ORDER BY order_date, channel, region"""
        )
    finally:
        _elapsed_ms = (time.perf_counter_ns() - _started) / 1_000_000
        _stop.set()
        _sampler.join(timeout=1)
        _peak[0] = max(_peak[0], _process.memory_info().rss)

    _accepted, _quarantined, _fact_rows, _daily_rows = con.execute(
        'SELECT (SELECT count(*) FROM "ducklake_workload_lab".staging.orders), '
        '(SELECT count(*) FROM "ducklake_workload_lab".staging.orders_quarantine), '
        '(SELECT count(*) FROM "ducklake_workload_lab".analytics.fact_orders), '
        '(SELECT count(*) FROM "ducklake_workload_lab".analytics.daily_channel_sales)'
    ).fetchone()
    quality.check(con, "prepared_orders_are_unique", 'SELECT order_id FROM "ducklake_workload_lab".staging.orders GROUP BY order_id HAVING count(*) > 1')
    quality.check(con, "accepted_plus_quarantine_reconciles", f"SELECT {_rows_in} AS raw_rows WHERE {_accepted} + {_quarantined} <> {_rows_in}")
    quality.check(con, "fact_matches_accepted_orders", f"SELECT {_accepted} AS accepted WHERE {_fact_rows} <> {_accepted}")
    _snapshot_after = int(con.execute('SELECT id FROM "ducklake_workload_lab".current_snapshot()').fetchone()[0])
    _storage = con.execute("SELECT coalesce(sum(file_count), 0), coalesce(sum(file_size_bytes), 0) FROM ducklake_table_info('ducklake_workload_lab')").fetchone()
    _measurement = {
        "run_id": _run_id, "profile": _profile, "repetition": _repetition, "phase": _phase,
        "step_order": 3, "workload": "prepare_and_model", "rows_in": _rows_in, "rows_out": int(_fact_rows),
        "input_bytes": 0, "output_bytes": int(_storage[1]), "elapsed_ms": round(_elapsed_ms, 3),
        "rows_per_second": round(_rows_in / (_elapsed_ms / 1000), 2), "peak_rss_bytes": int(_peak[0]),
        "snapshot_before": _snapshot_before, "snapshot_after": _snapshot_after, "correctness": True,
        "details": {"accepted": int(_accepted), "quarantined": int(_quarantined), "daily_rows": int(_daily_rows), "managed_files": int(_storage[0])},
    }
    run_artifacts.path("measurements/03_prepare_model.json").write_text(json.dumps(_measurement, indent=2), encoding="utf-8")
    result = _measurement
    preview = con.sql('SELECT * FROM "ducklake_workload_lab".analytics.daily_channel_sales ORDER BY revenue DESC LIMIT 15').pl().to_dicts()
    return con, preview, result


@app.cell
def _(notebook_output, preview, result):
    notebook_output.table([result], title="Step result")
    notebook_output.table(preview, title="Daily channel sales")
    return


@app.cell
def _(con, result):
    print(f"[benchmark] modeled accepted={result['details']['accepted']:,} quarantined={result['details']['quarantined']:,} elapsed_ms={result['elapsed_ms']}", flush=True)
    con.close()
    return


if __name__ == "__main__":
    app.run()
