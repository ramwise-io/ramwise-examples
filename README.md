# ramwise-examples

Runnable companion code for the field notes at **[ramwise.dev](https://ramwise.dev)**.

Each folder is a small, self-contained, tested example that makes one post's
core idea concrete. These are **teaching implementations** — freshly written to
illustrate a concept, not production libraries. They lean on the Python standard
library wherever possible, so most run with no dependencies at all.

| Example | The idea | Post |
|---|---|---|
| [`tiny-search`](tiny-search/) | An inverted index is the right *shape* for the data; scoring touches only the docs a query term points to — plus a benchmark that humbles the demo | [I Built a Search Engine by Hand](https://ramwise.dev/blog/i-built-a-search-engine-by-hand/) |
| [`incremental-load-patterns`](incremental-load-patterns/) | The naive timestamp-watermark **tie bug**, live, and the composite-watermark fix | [Incremental Load Is Not One Thing](https://ramwise.dev/blog/incremental-load-patterns/) |
| [`semantic-sql`](semantic-sql/) | A semantic layer as the unlock, and a verification gate that blocks a hallucinated query before it runs | [I Taught an LLM to Query Data in English](https://ramwise.dev/blog/i-taught-an-llm-to-query-data-in-english/) |
| [`date-dimension`](date-dimension/) | Persist the deterministic core; compute the today-relative fields at the edge | [Your Date Dimension Is Not Static](https://ramwise.dev/blog/date-dimension-not-static/) |
| [`recipe-ratios`](recipe-ratios/) | A cookie's effective fat-to-flour ratio is a validation range — reject the physically-impossible recipe before the oven | [You Can Check a Cookie](https://ramwise.dev/blog/you-can-check-a-cookie/) |
| [`sql-pipeline-runner`](sql-pipeline-runner/) | `compile → run → assert → emit`: assertions gate the run before bad data lands, and exit codes say who to page | [Keep the Runner Dumb](https://ramwise.dev/blog/keep-the-runner-dumb/) |
| [`evidence-weight`](evidence-weight/) | Belief-strength and agreement as separate axes, independence discounting, and a conflict router that refuses "latest wins" | [Weight Isn't Agreement](https://ramwise.dev/blog/weight-isnt-agreement/) |
| [`fastpitch-per-phoneme`](fastpitch-per-phoneme/) | Slow speech per phoneme (hold vowels, keep stops crisp) with FastPitch's per-token `pace` — a GPU/Colab notebook, not zero-dep | [Generate Slow, Don't Slow the Generation](https://ramwise.dev/blog/generate-slow-dont-slow-the-generation/) |
| [`preflight-check`](preflight-check/) | Fingerprint every file in a bulk-load set against a `name,type` baseline; name the golden/broken partition before the load runs — reports structure, never guesses a rename (needs `duckdb`) | [The Files That Break Your Bulk Load](https://ramwise.dev/blog/the-files-that-break-your-bulk-load/) |

## Running

Each folder has a `README`, a runnable module, and tests:

```bash
cd tiny-search
python tiny_search.py          # a demo
python test_tiny_search.py     # tests — every folder's tests also run under pytest
```

Run every test suite:

```bash
python -m pytest    # or: for d in */; do (cd "$d" && python test_*.py); done
```

The one exception is [`fastpitch-per-phoneme`](fastpitch-per-phoneme/), a
GPU/Colab teaching notebook rather than a zero-dependency module.

## License

MIT — see [LICENSE](LICENSE).
