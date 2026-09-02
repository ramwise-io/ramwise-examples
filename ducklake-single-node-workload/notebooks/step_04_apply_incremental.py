import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="DuckLake Workload Lab · 4. Apply incremental changes")


@app.cell
def _():
    from anatini import lake, notebook_output, quality, run_artifacts
    return lake, notebook_output, quality, run_artifacts


@app.cell
def _(notebook_output):
    notebook_output.markdown(
        """# 4. Apply incremental changes

Land a CSV change batch, then apply inserts, updates, and deletes idempotently.
The snapshot immediately before mutation is retained for the historical-read step."""
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
    _changes_path = lake.project_path("files/benchmark/current/order_changes.csv")
    if not _changes_path.is_file():
        raise FileNotFoundError("order_changes.csv is missing")

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
    _before_rows = int(con.execute('SELECT count(*) FROM "ducklake_workload_lab".staging.orders').fetchone()[0])
    _started = time.perf_counter_ns()
    try:
        con.execute(f'CREATE OR REPLACE TABLE "ducklake_workload_lab".raw.order_changes AS SELECT * FROM read_csv_auto({_literal(_changes_path)}, header=true)')
        _op_counts = {row[0]: int(row[1]) for row in con.execute('SELECT operation, count(*) FROM "ducklake_workload_lab".raw.order_changes GROUP BY operation').fetchall()}
        con.execute(
            """DELETE FROM "ducklake_workload_lab".staging.orders target
               USING "ducklake_workload_lab".raw.order_changes changes
               WHERE changes.operation = 'delete' AND target.order_id = changes.order_id"""
        )
        con.execute(
            """UPDATE "ducklake_workload_lab".staging.orders target SET
                   customer_id = changes.customer_id, product_id = changes.product_id,
                   order_ts = changes.order_ts, channel = changes.channel,
                   quantity = changes.quantity, unit_price = changes.unit_price,
                   discount_rate = changes.discount_rate, region = changes.region,
                   net_amount = round(changes.quantity * changes.unit_price * (1 - changes.discount_rate), 2)
               FROM "ducklake_workload_lab".raw.order_changes changes
               WHERE changes.operation = 'update' AND target.order_id = changes.order_id"""
        )
        con.execute(
            """INSERT INTO "ducklake_workload_lab".staging.orders
               SELECT changes.order_id, changes.customer_id, changes.product_id, changes.order_ts,
                      changes.channel, changes.quantity, changes.unit_price, changes.discount_rate,
                      changes.region, round(changes.quantity * changes.unit_price * (1 - changes.discount_rate), 2)
               FROM "ducklake_workload_lab".raw.order_changes changes
               WHERE changes.operation = 'insert'
                 AND NOT EXISTS (SELECT 1 FROM "ducklake_workload_lab".staging.orders target WHERE target.order_id = changes.order_id)"""
        )
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
                   FROM "ducklake_workload_lab".analytics.fact_orders GROUP BY ALL
               )
               SELECT *, round(sum(revenue) OVER (PARTITION BY channel, region ORDER BY order_date), 2) AS running_revenue
               FROM daily ORDER BY order_date, channel, region"""
        )
    finally:
        _elapsed_ms = (time.perf_counter_ns() - _started) / 1_000_000
        _stop.set()
        _sampler.join(timeout=1)
        _peak[0] = max(_peak[0], _process.memory_info().rss)

    _after_rows = int(con.execute('SELECT count(*) FROM "ducklake_workload_lab".staging.orders').fetchone()[0])
    _fact_rows = int(con.execute('SELECT count(*) FROM "ducklake_workload_lab".analytics.fact_orders').fetchone()[0])
    quality.check(con, "incremental_orders_remain_unique", 'SELECT order_id FROM "ducklake_workload_lab".staging.orders GROUP BY order_id HAVING count(*) > 1')
    quality.check(con, "incremental_fact_matches_prepared", f"SELECT {_after_rows} AS prepared WHERE {_fact_rows} <> {_after_rows}")
    _snapshot_after = int(con.execute('SELECT id FROM "ducklake_workload_lab".current_snapshot()').fetchone()[0])
    _storage = con.execute("SELECT coalesce(sum(file_count), 0), coalesce(sum(file_size_bytes), 0) FROM ducklake_table_info('ducklake_workload_lab')").fetchone()
    _change_rows = sum(_op_counts.values())
    _measurement = {
        "run_id": _run_id, "profile": _profile, "repetition": _repetition, "phase": _phase,
        "step_order": 4, "workload": "apply_incremental", "rows_in": int(_change_rows), "rows_out": _after_rows,
        "input_bytes": _changes_path.stat().st_size, "output_bytes": int(_storage[1]), "elapsed_ms": round(_elapsed_ms, 3),
        "rows_per_second": round(_change_rows / (_elapsed_ms / 1000), 2), "peak_rss_bytes": int(_peak[0]),
        "snapshot_before": _snapshot_before, "snapshot_after": _snapshot_after, "correctness": True,
        "details": {"before_rows": _before_rows, "after_rows": _after_rows, "operations": _op_counts, "managed_files": int(_storage[0])},
    }
    run_artifacts.path("measurements/04_apply_incremental.json").write_text(json.dumps(_measurement, indent=2), encoding="utf-8")
    result = _measurement
    preview = con.sql('SELECT operation, count(*) AS rows FROM "ducklake_workload_lab".raw.order_changes GROUP BY operation ORDER BY operation').pl().to_dicts()
    return con, preview, result


@app.cell
def _(notebook_output, preview, result):
    notebook_output.table([result], title="Step result")
    notebook_output.table(preview, title="Change batch")
    return


@app.cell
def _(con, result):
    print(f"[benchmark] incremental operations={result['rows_in']:,} before={result['details']['before_rows']:,} after={result['details']['after_rows']:,}", flush=True)
    con.close()
    return


if __name__ == "__main__":
    app.run()
