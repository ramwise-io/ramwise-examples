# semantic-sql

A runnable sketch of natural-language-to-SQL with a **semantic layer** and a
**verification boundary** — the two ideas behind
**[I Taught an LLM to Query Data in English](https://ramwise.dev/blog/i-taught-an-llm-to-query-data-in-english/)**.

Pure standard library (`sqlite3`, `json`, `re`). The data is fully **synthetic**
(a toy e-commerce store). The LLM step is pluggable and **stubbed offline**, so
this runs with no API key and no network. The gate is the real part.

## The two ideas

1. **The semantic layer is the unlock.** The model can't reason about cryptic
   columns (`ctry`, `amt`, `sts`). [`semantic_layer.json`](semantic_layer.json)
   describes what each table and column *means*; `build_prompt` shows exactly
   what context the model receives.

2. **The verification boundary is what makes it safe.** The model *drafts*; the
   deterministic `verify` gate checks the query is read-only and schema-valid
   **before it runs and before anyone believes the result.** The model proposes;
   it never gets to conclude.

## Run it

```bash
python semantic_sql.py        # a valid question, then a hallucinated draft
python test_semantic_sql.py   # tests (also works under: pytest)
```
```
== a valid question ==
result: [('US', 400.0), ('IN', 250.0)]

== a hallucinated draft ==
gate blocked it: hallucinated column: orders.revenue does not exist
```

The valid query runs and returns the right revenue. The hallucinated one —
`SUM(orders.revenue)`, a column that doesn't exist — is caught by the gate and
never touches the database.

## Plugging in a real model

Replace `draft_sql(question, prompt)` with a call to any LLM. Everything around
it — the semantic layer as context, the read-only + schema-valid gate, the
human approval hook — stays exactly the same. On data that matters, that
scaffolding is the point, not the model.

The gate here is intentionally simple (read-only enforcement + validation of the
tables and qualified `table.column` references it can see). A production version
would parse the SQL properly; the lesson is the *boundary*, not the parser.
