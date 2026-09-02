import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="DuckLake Workload Lab · Complete workload")


@app.cell
def _():
    from anatini import notebook_output, workflow
    return notebook_output, workflow


@app.cell
def _(notebook_output):
    notebook_output.markdown(
        """# DuckLake single-node workload lab

One schedulable notebook executes the complete workload: source files, batch
ingestion, preparation and modeling, incremental changes, JSON events, quality,
historical reads, and publication. Every child notebook remains clickable in the
retained Run graph."""
    )
    return


@app.cell
def _():
    params = {}
    return (params,)


@app.cell
def _(notebook_output, params):
    _parameter_rows = [{"parameter": key, "value": repr(value)} for key, value in sorted(params.items())]
    notebook_output.markdown("## Workload parameters")
    if _parameter_rows:
        notebook_output.table(_parameter_rows)
    else:
        notebook_output.markdown("No parameters supplied; the quick exploratory profile is used.")
    return


@app.cell
def _(params, workflow):
    from anatini_projects.ducklake_workload_lab.notebooks.step_01_generate_sources import app as generate_sources
    from anatini_projects.ducklake_workload_lab.notebooks.step_02_ingest_batch import app as ingest_batch
    from anatini_projects.ducklake_workload_lab.notebooks.step_03_prepare_model import app as prepare_model
    from anatini_projects.ducklake_workload_lab.notebooks.step_04_apply_incremental import app as apply_incremental
    from anatini_projects.ducklake_workload_lab.notebooks.step_05_process_events import app as process_events
    from anatini_projects.ducklake_workload_lab.notebooks.step_06_validate_quality import app as validate_quality
    from anatini_projects.ducklake_workload_lab.notebooks.step_07_query_history import app as query_history
    from anatini_projects.ducklake_workload_lab.notebooks.step_08_publish_results import app as publish_results

    _steps = [
        ("generate_sources", generate_sources),
        ("ingest_batch", ingest_batch),
        ("prepare_and_model", prepare_model),
        ("apply_incremental", apply_incremental),
        ("process_events", process_events),
        ("validate_quality", validate_quality),
        ("query_history", query_history),
        ("publish_results", publish_results),
    ]
    completed = []
    for _position, (_name, _app) in enumerate(_steps, start=1):
        print(f"[pipeline] Step {_position}/{len(_steps)}: {_name}", flush=True)
        _outputs, _definitions = workflow.run_step(_name, _app, params)
        completed.append({"step": _name, "result": _definitions.get("result")})
    result = {
        "status": "success",
        "profile": str(params.get("profile", "quick")),
        "repetition": int(params.get("repetition", 0)),
        "phase": str(params.get("phase", "exploratory")),
        "steps": completed,
        "summary": completed[-1]["result"],
    }
    print(f"[pipeline] Completed profile={result['profile']} repetition={result['repetition']}", flush=True)
    return completed, result


@app.cell
def _(completed, notebook_output, result):
    notebook_output.table([result["summary"]], title="Complete workload result")
    notebook_output.table([{"step": row["step"], "elapsed_ms": row["result"].get("elapsed_ms"), "correctness": row["result"].get("correctness", row["result"].get("all_correct"))} for row in completed], title="Pipeline steps")
    return


if __name__ == "__main__":
    app.run()
