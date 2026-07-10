"""A tiny compile -> run -> assert -> emit SQL pipeline runner, on SQLite.

Companion code for:
https://ramwise.dev/blog/keep-the-runner-dumb/

The idea from the post, distilled: a data pipeline can be one SQL file whose
steps run top to bottom, with data-quality ASSERTIONS as first-class steps that
halt the run *before* bad data is written. The runner stays deliberately dumb --
it does not parse your SQL, hold state, or schedule anything. It compiles the
directives to a plan, runs the SQL on an in-memory database (so a failed run
corrupts nothing and "re-run from the top" is the whole resume story), gates on
the assertions, emits a structured report, and exits with a code that says WHO
to wake up: the data arrived bad (10) vs the script is broken (20).

The real tool, pktl, runs on DuckDB; this teaching version uses stdlib sqlite3
so it runs with nothing installed. The pattern is engine-agnostic.

Pure standard library (sqlite3, re). No dependencies.

Run:  python pipeline_runner.py
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

# Exit codes are a contract: they tell the caller who owns the failure.
OK, DATA_BAD, SCRIPT_BROKEN, STRUCTURE_BAD = 0, 10, 20, 30

_DIRECTIVE = re.compile(r"^--@(\w+)\s*(.*)$")
_ASSERT_MODES = ("EXPECT_NO_ROWS", "EXPECT_TRUE")


@dataclass
class Block:
    kind: str          # "step" | "assert"
    name: str
    sql: str
    mode: str = ""     # for asserts: EXPECT_NO_ROWS | EXPECT_TRUE


@dataclass
class Plan:
    pipeline: str
    blocks: list


@dataclass
class Report:
    pipeline: str
    status: str
    exit_code: int
    steps: list = field(default_factory=list)
    assertions: list = field(default_factory=list)
    output: list = field(default_factory=list)
    error: str = ""


def compile(script: str) -> Plan:
    """Parse directive-annotated SQL into a plan. Pure: same text -> same plan.

    Directives: --@pipeline <name>, --@step <name>, --@assert <name> <MODE>.
    The SQL between directives is opaque; we never parse it.
    """
    pipeline: str | None = None
    blocks: list[Block] = []
    cur: tuple[str, str, str] | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal cur, body
        if cur is None:
            body = []
            return
        sql = "\n".join(body).strip().rstrip(";").strip()
        kind, name, mode = cur
        if not sql:
            raise SyntaxError(f"{kind} '{name}' has no SQL")
        if ";" in sql:
            raise SyntaxError(f"{kind} '{name}' must be exactly one statement")
        blocks.append(Block(kind, name, sql, mode))
        cur, body = None, []

    for line in script.splitlines():
        m = _DIRECTIVE.match(line.strip())
        if not m:
            body.append(line)
            continue
        directive, rest = m.group(1), m.group(2).strip()
        if directive == "pipeline":
            flush()
            pipeline = rest
        elif directive == "step":
            flush()
            cur = ("step", rest, "")
        elif directive == "assert":
            flush()
            parts = rest.split()
            name = parts[0]
            mode = parts[1] if len(parts) > 1 else "EXPECT_NO_ROWS"
            if mode not in _ASSERT_MODES:
                raise SyntaxError(f"unknown assert mode {mode!r}")
            cur = ("assert", name, mode)
        else:
            raise SyntaxError(f"unknown directive --@{directive}")
    flush()

    if pipeline is None:
        raise SyntaxError("missing --@pipeline")
    return Plan(pipeline, blocks)


def run(script: str, source_rows) -> Report:
    """Compile, load the source, run blocks in order, gate on asserts, emit a report.

    `source_rows` is the input ("--@param source INPUT", simplified) -- a list
    of {"id", "amount"} dicts loaded into a table `raw`. The pipeline is
    expected to build a table `out`; on success its rows become the artifact.
    """
    try:
        plan = compile(script)
    except SyntaxError as e:
        return Report("?", "structure_error", STRUCTURE_BAD, error=str(e))

    rep = Report(plan.pipeline, "success", OK)
    db = sqlite3.connect(":memory:")   # ephemeral: nothing survives to corrupt the next run
    db.row_factory = sqlite3.Row
    try:
        db.execute("CREATE TABLE raw (id INTEGER, amount REAL)")
        db.executemany("INSERT INTO raw (id, amount) VALUES (:id, :amount)", source_rows)

        for b in plan.blocks:
            if b.kind == "step":
                try:
                    db.execute(b.sql)
                except sqlite3.Error as e:
                    rep.status, rep.exit_code = "sql_error", SCRIPT_BROKEN
                    rep.error = f"step '{b.name}': {e}"
                    rep.steps.append({"name": b.name, "status": "error"})
                    return rep
                rep.steps.append({"name": b.name, "status": "ok"})
            else:  # assert
                try:
                    rows = db.execute(b.sql).fetchall()
                except sqlite3.Error as e:
                    rep.status, rep.exit_code = "sql_error", SCRIPT_BROKEN
                    rep.error = f"assert '{b.name}': {e}"
                    return rep
                if b.mode == "EXPECT_NO_ROWS":
                    passed = len(rows) == 0
                else:  # EXPECT_TRUE
                    passed = bool(rows) and bool(rows[0][0])
                rep.assertions.append({
                    "name": b.name,
                    "mode": b.mode,
                    "status": "passed" if passed else "failed",
                    "evidence": [dict(r) for r in rows[:5]] if not passed else [],
                })
                if not passed:
                    # The data is bad. Halt here -- before anything downstream writes it.
                    rep.status, rep.exit_code = "assertion_failed", DATA_BAD
                    return rep

        # Success: emit the `out` table (if the pipeline built one) as the artifact.
        try:
            rep.output = [dict(r) for r in db.execute("SELECT * FROM out").fetchall()]
        except sqlite3.Error:
            rep.output = []
        return rep
    finally:
        db.close()


EXAMPLE = """\
--@pipeline load_orders

--@step clean
CREATE TABLE out AS SELECT id, amount FROM raw WHERE amount > 0

--@assert no_null_id EXPECT_NO_ROWS
SELECT * FROM out WHERE id IS NULL

--@assert has_rows EXPECT_TRUE
SELECT count(*) > 0 FROM out
"""


def _show(title, rep: Report) -> None:
    print(title)
    print(f"  status={rep.status}  exit={rep.exit_code}")
    for a in rep.assertions:
        print(f"  assert {a['name']}: {a['status']}"
              + (f"  evidence={a['evidence']}" if a["evidence"] else ""))
    print(f"  output rows: {rep.output}")
    print()


if __name__ == "__main__":
    clean = [{"id": 1, "amount": 10.0}, {"id": 2, "amount": 20.0}, {"id": 3, "amount": -5.0}]
    dirty = clean + [{"id": None, "amount": 15.0}]   # a null id sneaks in

    _show("clean source -> passes, exports:", run(EXAMPLE, clean))
    _show("dirty source -> assert halts BEFORE export (exit 10):", run(EXAMPLE, dirty))
    print("The null-id row trips `no_null_id`, so the run stops with exit 10")
    print("('data is bad') and never emits output. A broken SQL step would exit")
    print("20 ('script is broken') -- a different problem with a different owner.")
