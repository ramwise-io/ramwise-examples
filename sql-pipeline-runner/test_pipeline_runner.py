"""Tests for pipeline_runner.py. Run with `python test_pipeline_runner.py` or `pytest`."""
from __future__ import annotations

from pipeline_runner import (
    DATA_BAD,
    EXAMPLE,
    OK,
    SCRIPT_BROKEN,
    STRUCTURE_BAD,
    compile,
    run,
)

CLEAN = [{"id": 1, "amount": 10.0}, {"id": 2, "amount": 20.0}, {"id": 3, "amount": -5.0}]
DIRTY = CLEAN + [{"id": None, "amount": 15.0}]


def test_clean_source_passes_and_exports():
    rep = run(EXAMPLE, CLEAN)
    assert rep.exit_code == OK
    assert rep.status == "success"
    ids = sorted(r["id"] for r in rep.output)
    assert ids == [1, 2]                       # id 3 filtered out by amount > 0
    assert all(a["status"] == "passed" for a in rep.assertions)


def test_dirty_source_halts_before_export():
    rep = run(EXAMPLE, DIRTY)
    assert rep.exit_code == DATA_BAD           # "data is bad" (10)
    assert rep.status == "assertion_failed"
    assert rep.output == []                     # nothing was emitted
    failed = [a for a in rep.assertions if a["status"] == "failed"]
    assert failed and failed[0]["name"] == "no_null_id"
    assert failed[0]["evidence"]                # kept the offending row as evidence


def test_broken_sql_step_exits_20():
    script = """\
--@pipeline broken
--@step oops
CREATE TABLE out AS SELECT * FROM table_that_does_not_exist
"""
    rep = run(script, CLEAN)
    assert rep.exit_code == SCRIPT_BROKEN       # "script is broken" (20), a different owner
    assert rep.status == "sql_error"


def test_missing_pipeline_directive_is_structure_error():
    rep = run("--@step x\nSELECT 1", CLEAN)
    assert rep.exit_code == STRUCTURE_BAD


def test_step_must_be_one_statement():
    rep = run("--@pipeline p\n--@step two\nSELECT 1; SELECT 2", CLEAN)
    assert rep.exit_code == STRUCTURE_BAD


def test_compile_is_pure():
    assert compile(EXAMPLE) == compile(EXAMPLE)   # same text -> same plan, deterministic


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
