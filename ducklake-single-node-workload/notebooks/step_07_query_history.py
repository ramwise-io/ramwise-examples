import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="DuckLake Workload Lab · 7. Query history")


@app.cell
def _():
    from anatini import lake, notebook_output, run_artifacts
    return lake, notebook_output, run_artifacts


@app.cell
def _(notebook_output):
    notebook_output.markdown(
        """# 7. Query historical state

Use the snapshot captured immediately before the incremental batch to reproduce
the earlier order state, then compare it with the current table."""
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

    import psutil

    _profile = str(params.get("profile", "quick")).lower()
    _repetition = int(params.get("repetition", 0))
    _phase = str(params.get("phase", "exploratory"))
    _run_id = os.getenv("ANATINI_RUN_ID", "interactive")
    _incremental = json.loads(run_artifacts.path("measurements/04_apply_incremental.json").read_text(encoding="utf-8"))
    _historical_snapshot = int(_incremental["snapshot_before"])
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
        _historical_rows, _historical_revenue = con.execute(
            f'SELECT count(*), round(sum(net_amount), 2) FROM "ducklake_workload_lab".staging.orders AT (VERSION => {_historical_snapshot})'
        ).fetchone()
        _current_rows, _current_revenue = con.execute(
            'SELECT count(*), round(sum(net_amount), 2) FROM "ducklake_workload_lab".staging.orders'
        ).fetchone()
    finally:
        _elapsed_ms = (time.perf_counter_ns() - _started) / 1_000_000
        _stop.set()
        _sampler.join(timeout=1)
        _peak[0] = max(_peak[0], _process.memory_info().rss)

    _snapshot_after = int(con.execute('SELECT id FROM "ducklake_workload_lab".current_snapshot()').fetchone()[0])
    _storage = con.execute("SELECT coalesce(sum(file_count), 0), coalesce(sum(file_size_bytes), 0) FROM ducklake_table_info('ducklake_workload_lab')").fetchone()
    _measurement = {
        "run_id": _run_id, "profile": _profile, "repetition": _repetition, "phase": _phase,
        "step_order": 7, "workload": "query_history", "rows_in": int(_historical_rows + _current_rows), "rows_out": 2,
        "input_bytes": 0, "output_bytes": int(_storage[1]), "elapsed_ms": round(_elapsed_ms, 3),
        "rows_per_second": round((_historical_rows + _current_rows) / (_elapsed_ms / 1000), 2), "peak_rss_bytes": int(_peak[0]),
        "snapshot_before": _snapshot_before, "snapshot_after": _snapshot_after, "correctness": _snapshot_before == _snapshot_after,
        "details": {"historical_snapshot": _historical_snapshot, "historical_rows": int(_historical_rows), "current_rows": int(_current_rows), "historical_revenue": float(_historical_revenue), "current_revenue": float(_current_revenue), "managed_files": int(_storage[0])},
    }
    run_artifacts.path("measurements/07_query_history.json").write_text(json.dumps(_measurement, indent=2), encoding="utf-8")
    result = _measurement
    preview = [
        {"state": "before incremental", "snapshot": _historical_snapshot, "rows": int(_historical_rows), "revenue": float(_historical_revenue)},
        {"state": "current", "snapshot": _snapshot_after, "rows": int(_current_rows), "revenue": float(_current_revenue)},
    ]
    return con, preview, result


@app.cell
def _(notebook_output, preview, result):
    notebook_output.table([result], title="Step result")
    notebook_output.table(preview, title="Historical comparison")
    return


@app.cell
def _(con, result):
    print(f"[benchmark] historical snapshot={result['details']['historical_snapshot']} rows_before={result['details']['historical_rows']:,} rows_current={result['details']['current_rows']:,}", flush=True)
    con.close()
    return


if __name__ == "__main__":
    app.run()
