"""Semantic-layer text-to-SQL with a verification boundary.

Companion code for:
https://ramwise.dev/blog/i-taught-an-llm-to-query-data-in-english/

Two ideas, both runnable:

  1. The SEMANTIC LAYER is the unlock. A model can't reason about cryptic schema
     names (`ctry`, `amt`, `sts`). A metadata layer describing what each table
     and column MEANS is what lets it draft a sensible query. `build_prompt`
     shows exactly what context we hand the model.

  2. The VERIFICATION BOUNDARY is what makes it safe. The model DRAFTS; a
     deterministic gate (`verify`) checks the query is read-only and
     schema-valid BEFORE it is ever run and before anyone believes the result.
     The model proposes; it never gets to conclude.

The data is fully synthetic (a toy e-commerce store). The LLM step is pluggable
and stubbed offline, so this runs with no API key and no network. The gate is
the real, deterministic part -- and it is what catches a hallucinated query.

Pure standard library (sqlite3, json, re).
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
SEMANTIC_LAYER = json.loads((HERE / "semantic_layer.json").read_text())


def setup_db() -> sqlite3.Connection:
    """A tiny synthetic store. Cryptic column names on purpose."""
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE customers (cust_id INTEGER PRIMARY KEY, ctry TEXT, seg TEXT);
        CREATE TABLE orders (ord_id INTEGER PRIMARY KEY, cust_id INTEGER,
                             amt REAL, sts TEXT, ord_dt TEXT);
        """
    )
    db.executemany(
        "INSERT INTO customers VALUES (?,?,?)",
        [(1, "US", "consumer"), (2, "IN", "business"), (3, "US", "business")],
    )
    db.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?)",
        [
            (10, 1, 100.0, "paid", "2026-01-05"),
            (11, 2, 250.0, "paid", "2026-01-06"),
            (12, 1, 40.0, "pending", "2026-01-07"),
            (13, 3, 300.0, "paid", "2026-01-08"),
            (14, 2, 90.0, "refunded", "2026-01-09"),
        ],
    )
    db.commit()
    return db


def schema_map() -> dict[str, set[str]]:
    return {t: set(spec["columns"]) for t, spec in SEMANTIC_LAYER["tables"].items()}


def build_prompt(question: str) -> str:
    """The context we hand the model: the schema PLUS what it means."""
    lines = ["You write SQLite SELECT queries. Schema and meaning:"]
    for table, spec in SEMANTIC_LAYER["tables"].items():
        lines.append(f"\nTable {table} -- {spec['meaning']}")
        for col, meaning in spec["columns"].items():
            lines.append(f"  {table}.{col}: {meaning}")
    lines.append("\nJoins: " + "; ".join(SEMANTIC_LAYER["joins"]))
    lines.append(f"\nQuestion: {question}\nReturn one SELECT query only.")
    return "\n".join(lines)


# --- the pluggable LLM step -------------------------------------------------
# Swap this for a real model call. It receives the prompt from build_prompt and
# returns a candidate SQL string. Stubbed offline here so the demo runs anywhere.
def draft_sql(question: str, prompt: str) -> str:
    q = question.lower()
    if "hallucinate" in q:
        # a plausible-looking draft that invents a column: orders.revenue does
        # not exist (the real column is orders.amt). The gate must catch this.
        return (
            "SELECT customers.ctry, SUM(orders.revenue) "
            "FROM orders JOIN customers ON orders.cust_id = customers.cust_id "
            "GROUP BY customers.ctry"
        )
    if "revenue" in q and "country" in q:
        # a correct draft
        return (
            "SELECT customers.ctry AS country, SUM(orders.amt) AS revenue "
            "FROM orders JOIN customers ON orders.cust_id = customers.cust_id "
            "WHERE orders.sts = 'paid' GROUP BY customers.ctry ORDER BY revenue DESC"
        )
    raise ValueError(f"stub has no canned answer for: {question!r}")


# --- the verification boundary (deterministic; the real star) ---------------
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|"
    r"truncate|vacuum|reindex)\b",
    re.I,
)
_QUALIFIED = re.compile(r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b", re.I)
_TABLES_FROM = re.compile(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", re.I)


def verify(sql: str, schema: dict[str, set[str]]) -> tuple[bool, str]:
    """Return (ok, reason). A query must pass this before it may run."""
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        return False, "only a single statement is allowed"
    if not re.match(r"(?is)^\s*select\b", stripped):
        return False, "only read-only SELECT queries are allowed"
    if _FORBIDDEN.search(stripped):
        return False, "contains a write / DDL keyword"
    for table in _TABLES_FROM.findall(stripped):
        if table not in schema:
            return False, f"unknown table: {table}"
    for table, col in _QUALIFIED.findall(stripped):
        if table not in schema:
            return False, f"unknown table: {table}"
        if col not in schema[table]:
            return False, f"hallucinated column: {table}.{col} does not exist"
    return True, "ok"


def ask(db: sqlite3.Connection, question: str, *, approve=lambda sql: True):
    """Full loop: draft -> verify -> (approve) -> run. Raises if the gate fails."""
    prompt = build_prompt(question)
    sql = draft_sql(question, prompt)
    ok, reason = verify(sql, schema_map())
    if not ok:
        raise PermissionError(f"verification failed: {reason}\n  query: {sql}")
    if not approve(sql):
        raise PermissionError("query not approved by the human in the loop")
    return sql, db.execute(sql).fetchall()


if __name__ == "__main__":
    db = setup_db()

    print("== a valid question ==")
    sql, rows = ask(db, "total revenue by country for paid orders")
    print("drafted & verified SQL:\n ", sql)
    print("result:", rows)

    print("\n== a hallucinated draft ==")
    try:
        ask(db, "revenue by country but hallucinate a column")
    except PermissionError as e:
        print("gate blocked it:", str(e).splitlines()[0])
    print("\nThe model drafts. The gate decides what is allowed to run.")
