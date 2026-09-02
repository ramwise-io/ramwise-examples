import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="DuckLake Workload Lab · 5. Process events")


@app.cell
def _():
    from anatini import lake, notebook_output, quality, run_artifacts
    return lake, notebook_output, quality, run_artifacts


@app.cell
def _(notebook_output):
    notebook_output.markdown(
        """# 5. Process semi-structured events

Parse newline-delimited JSON with a nested properties object, quarantine events
without an identity, and produce session-level funnel facts."""
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
    _events_path = lake.project_path("files/benchmark/current/events.jsonl")
    if not _events_path.is_file():
        raise FileNotFoundError("events.jsonl is missing")

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
        con.execute(f'CREATE OR REPLACE TABLE "ducklake_workload_lab".raw.events AS SELECT * FROM read_json_auto({_literal(_events_path)}, format=\'newline_delimited\')')
        con.execute(
            """CREATE OR REPLACE TABLE "ducklake_workload_lab".staging.events_quarantine AS
               SELECT *, 'missing_customer_id' AS quarantine_reason
               FROM "ducklake_workload_lab".raw.events WHERE customer_id IS NULL"""
        )
        con.execute(
            """CREATE OR REPLACE TABLE "ducklake_workload_lab".staging.events AS
               SELECT event_id::BIGINT AS event_id, customer_id::BIGINT AS customer_id,
                      event_ts::TIMESTAMP AS event_ts, event_type::VARCHAR AS event_type,
                      properties.session_id::VARCHAR AS session_id,
                      properties.device::VARCHAR AS device,
                      properties.campaign::VARCHAR AS campaign
               FROM "ducklake_workload_lab".raw.events WHERE customer_id IS NOT NULL"""
        )
        con.execute(
            """CREATE OR REPLACE TABLE "ducklake_workload_lab".analytics.sessions AS
               SELECT session_id, any_value(customer_id) AS customer_id,
                      min(event_ts) AS session_start, max(event_ts) AS session_end,
                      date_diff('second', min(event_ts), max(event_ts)) AS duration_seconds,
                      count(*) AS event_count,
                      count_if(event_type = 'page_view') AS page_views,
                      count_if(event_type = 'add_to_cart') AS cart_events,
                      count_if(event_type = 'checkout') AS checkout_events,
                      count_if(event_type = 'purchase') AS purchase_events,
                      any_value(device) AS device, any_value(campaign) AS campaign
               FROM "ducklake_workload_lab".staging.events GROUP BY session_id"""
        )
    finally:
        _elapsed_ms = (time.perf_counter_ns() - _started) / 1_000_000
        _stop.set()
        _sampler.join(timeout=1)
        _peak[0] = max(_peak[0], _process.memory_info().rss)

    _raw_rows, _accepted, _quarantined, _sessions = con.execute(
        'SELECT (SELECT count(*) FROM "ducklake_workload_lab".raw.events), '
        '(SELECT count(*) FROM "ducklake_workload_lab".staging.events), '
        '(SELECT count(*) FROM "ducklake_workload_lab".staging.events_quarantine), '
        '(SELECT count(*) FROM "ducklake_workload_lab".analytics.sessions)'
    ).fetchone()
    quality.check(con, "event_ids_are_unique", 'SELECT event_id FROM "ducklake_workload_lab".raw.events GROUP BY event_id HAVING count(*) > 1')
    quality.check(con, "events_reconcile_with_quarantine", f"SELECT {_raw_rows} AS raw_rows WHERE {_accepted} + {_quarantined} <> {_raw_rows}")
    quality.check(con, "session_ids_are_unique", 'SELECT session_id FROM "ducklake_workload_lab".analytics.sessions GROUP BY session_id HAVING count(*) > 1')
    _snapshot_after = int(con.execute('SELECT id FROM "ducklake_workload_lab".current_snapshot()').fetchone()[0])
    _storage = con.execute("SELECT coalesce(sum(file_count), 0), coalesce(sum(file_size_bytes), 0) FROM ducklake_table_info('ducklake_workload_lab')").fetchone()
    _measurement = {
        "run_id": _run_id, "profile": _profile, "repetition": _repetition, "phase": _phase,
        "step_order": 5, "workload": "process_events", "rows_in": int(_raw_rows), "rows_out": int(_accepted),
        "input_bytes": _events_path.stat().st_size, "output_bytes": int(_storage[1]), "elapsed_ms": round(_elapsed_ms, 3),
        "rows_per_second": round(_raw_rows / (_elapsed_ms / 1000), 2), "peak_rss_bytes": int(_peak[0]),
        "snapshot_before": _snapshot_before, "snapshot_after": _snapshot_after, "correctness": True,
        "details": {"accepted": int(_accepted), "quarantined": int(_quarantined), "sessions": int(_sessions), "managed_files": int(_storage[0])},
    }
    run_artifacts.path("measurements/05_process_events.json").write_text(json.dumps(_measurement, indent=2), encoding="utf-8")
    result = _measurement
    preview = con.sql('SELECT * FROM "ducklake_workload_lab".analytics.sessions ORDER BY event_count DESC, session_id LIMIT 15').pl().to_dicts()
    return con, preview, result


@app.cell
def _(notebook_output, preview, result):
    notebook_output.table([result], title="Step result")
    notebook_output.table(preview, title="Session preview")
    return


@app.cell
def _(con, result):
    print(f"[benchmark] events={result['rows_in']:,} accepted={result['details']['accepted']:,} quarantined={result['details']['quarantined']:,}", flush=True)
    con.close()
    return


if __name__ == "__main__":
    app.run()
