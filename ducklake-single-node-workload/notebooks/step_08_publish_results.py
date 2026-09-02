import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="DuckLake Workload Lab · 8. Publish results")


@app.cell
def _():
    from anatini import lake, notebook_output, run_artifacts
    return lake, notebook_output, run_artifacts


@app.cell
def _(notebook_output):
    notebook_output.markdown(
        """# 8. Publish the run evidence

Combine the step measurements into one browsable DuckLake table and portable Run
artifacts. The SVG below is generated from the same retained measurement rows."""
    )
    return


@app.cell
def _():
    params = {}
    return (params,)


@app.cell
def _(lake, params, run_artifacts):
    import csv
    import html
    import json
    import os
    import time

    import marimo as mo
    import pyarrow as pa

    _profile = str(params.get("profile", "quick")).lower()
    _repetition = int(params.get("repetition", 0))
    _phase = str(params.get("phase", "exploratory"))
    _run_id = os.getenv("ANATINI_RUN_ID", "interactive")
    _measurement_paths = sorted(run_artifacts.path("measurements").glob("[0-7][0-9]_*.json"))
    if len(_measurement_paths) != 7:
        raise RuntimeError(f"Expected seven workload measurements, found {len(_measurement_paths)}")
    _rows = [json.loads(path.read_text(encoding="utf-8")) for path in _measurement_paths]
    for _row in _rows:
        _row["details_json"] = json.dumps(_row.pop("details"), separators=(",", ":"), sort_keys=True)

    _csv_path = run_artifacts.path("benchmark-measurements.csv")
    with _csv_path.open("w", newline="", encoding="utf-8") as _handle:
        _writer = csv.DictWriter(_handle, fieldnames=list(_rows[0]))
        _writer.writeheader()
        _writer.writerows(_rows)

    _total_step_ms = sum(float(row["elapsed_ms"]) for row in _rows)
    _summary = {
        "run_id": _run_id,
        "profile": _profile,
        "repetition": _repetition,
        "phase": _phase,
        "workload_steps": len(_rows),
        "total_measured_step_ms": round(_total_step_ms, 3),
        "all_correct": all(bool(row["correctness"]) for row in _rows),
        "max_peak_rss_bytes": max(int(row["peak_rss_bytes"]) for row in _rows),
        "final_managed_bytes": int(_rows[-1]["output_bytes"]),
        "snapshot_start": int(_rows[0]["snapshot_before"]),
        "snapshot_end": int(_rows[-1]["snapshot_after"]),
    }
    run_artifacts.path("benchmark-summary.json").write_text(json.dumps(_summary, indent=2), encoding="utf-8")

    _chart_width = 900
    _chart_height = 90 + len(_rows) * 54
    _max_elapsed = max(float(row["elapsed_ms"]) for row in _rows) or 1
    _bars = []
    for _index, _row in enumerate(_rows):
        _y = 58 + _index * 54
        _width = max(2, 560 * float(_row["elapsed_ms"]) / _max_elapsed)
        _label = html.escape(str(_row["workload"]).replace("_", " "))
        _bars.append(f'<text x="20" y="{_y + 18}" font-size="14" fill="#24332d">{_label}</text>')
        _bars.append(f'<rect x="210" y="{_y}" width="{_width:.1f}" height="24" rx="4" fill="#2d6a55"/>')
        _bars.append(f'<text x="{220 + _width:.1f}" y="{_y + 17}" font-size="13" fill="#24332d">{float(_row["elapsed_ms"]):,.1f} ms</text>')
    _svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_chart_width}" height="{_chart_height}" viewBox="0 0 {_chart_width} {_chart_height}" role="img" aria-label="Elapsed time for each workload step">'
        '<rect width="100%" height="100%" fill="#f7faf8"/>'
        f'<text x="20" y="30" font-size="20" font-weight="700" fill="#153e32">{html.escape(_profile.title())} profile · elapsed time by step</text>'
        + "".join(_bars) + '</svg>'
    )
    run_artifacts.path("elapsed-by-step.svg").write_text(_svg, encoding="utf-8")

    con = lake.connect(profile="job")
    _started = time.perf_counter_ns()
    con.execute('CREATE SCHEMA IF NOT EXISTS "ducklake_workload_lab".benchmark')
    con.execute(
        """CREATE TABLE IF NOT EXISTS "ducklake_workload_lab".benchmark.measurements (
               run_id VARCHAR, profile VARCHAR, repetition INTEGER, phase VARCHAR,
               step_order INTEGER, workload VARCHAR, rows_in BIGINT, rows_out BIGINT,
               input_bytes BIGINT, output_bytes BIGINT, elapsed_ms DOUBLE,
               rows_per_second DOUBLE, peak_rss_bytes BIGINT, snapshot_before BIGINT,
               snapshot_after BIGINT, correctness BOOLEAN, details_json JSON,
               recorded_at TIMESTAMP DEFAULT current_timestamp
           )"""
    )
    con.execute('DELETE FROM "ducklake_workload_lab".benchmark.measurements WHERE run_id = ?', [_run_id])
    _frame = pa.Table.from_pylist(_rows)
    con.register("_run_measurements", _frame)
    con.execute(
        """INSERT INTO "ducklake_workload_lab".benchmark.measurements
           (run_id, profile, repetition, phase, step_order, workload, rows_in, rows_out,
            input_bytes, output_bytes, elapsed_ms, rows_per_second, peak_rss_bytes,
            snapshot_before, snapshot_after, correctness, details_json)
           SELECT run_id, profile, repetition, phase, step_order, workload, rows_in, rows_out,
                  input_bytes, output_bytes, elapsed_ms, rows_per_second, peak_rss_bytes,
                  snapshot_before, snapshot_after, correctness, details_json::JSON
           FROM _run_measurements ORDER BY step_order"""
    )
    con.execute(
        """CREATE OR REPLACE VIEW "ducklake_workload_lab".analytics.benchmark_run_summary AS
           SELECT profile, repetition, phase, run_id, count(*) AS workload_steps,
                  round(sum(elapsed_ms), 3) AS total_measured_step_ms,
                  max(peak_rss_bytes) AS max_peak_rss_bytes,
                  max(output_bytes) AS final_managed_bytes,
                  bool_and(correctness) AS all_correct,
                  min(snapshot_before) AS snapshot_start,
                  max(snapshot_after) AS snapshot_end
           FROM "ducklake_workload_lab".benchmark.measurements
           GROUP BY ALL"""
    )
    _publish_ms = (time.perf_counter_ns() - _started) / 1_000_000
    result = {**_summary, "publish_ms": round(_publish_ms, 3), "artifacts": ["benchmark-measurements.csv", "benchmark-summary.json", "elapsed-by-step.svg"]}
    _preview_result = con.execute('SELECT step_order, workload, rows_in, rows_out, elapsed_ms, rows_per_second, peak_rss_bytes, correctness FROM "ducklake_workload_lab".benchmark.measurements WHERE run_id = ? ORDER BY step_order', [_run_id])
    _preview_columns = [column[0] for column in _preview_result.description]
    preview = [dict(zip(_preview_columns, row, strict=True)) for row in _preview_result.fetchall()]
    chart = mo.Html(_svg)
    return chart, con, preview, result


@app.cell
def _(chart, notebook_output, preview, result):
    notebook_output.table([result], title="Run summary")
    notebook_output.display(chart, title="Elapsed time by workload")
    notebook_output.table(preview, title="Retained measurements")
    return


@app.cell
def _(con, result):
    print(f"[benchmark] published profile={result['profile']} steps={result['workload_steps']} total_step_ms={result['total_measured_step_ms']}", flush=True)
    con.close()
    return


if __name__ == "__main__":
    app.run()
