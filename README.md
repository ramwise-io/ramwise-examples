# ramwise-examples

Runnable companion code for the field notes at **[ramwise.dev](https://ramwise.dev)**.

Each folder is a self-contained companion that makes one post's core idea
concrete. Most are compact teaching implementations; the GPU studies also
include output-complete analysis notebooks and sanitized, reproducible
benchmark harnesses. These are examples, not production libraries.

| Example | The idea | Post |
|---|---|---|
| [`tiny-search`](tiny-search/) | An inverted index is the right *shape* for the data; scoring touches only the docs a query term points to — plus a benchmark that humbles the demo | [I Built a Search Engine by Hand](https://ramwise.dev/blog/i-built-a-search-engine-by-hand/) |
| [`incremental-load-patterns`](incremental-load-patterns/) | The naive timestamp-watermark **tie bug**, live, and the composite-watermark fix | [Incremental Load Is Not One Thing](https://ramwise.dev/blog/incremental-load-patterns/) |
| [`semantic-sql`](semantic-sql/) | Semantic metadata plus a verification gate that blocks a hallucinated query before it runs | [A Text-to-SQL Prototype for Patient Data](https://ramwise.dev/blog/i-taught-an-llm-to-query-data-in-english/) |
| [`date-dimension`](date-dimension/) | Persist the deterministic core; compute the today-relative fields at the edge | [Your Date Dimension Is Not Static](https://ramwise.dev/blog/date-dimension-not-static/) |
| [`recipe-ratios`](recipe-ratios/) | A cookie's effective fat-to-flour ratio is a validation range — reject the physically-impossible recipe before the oven | [You Can Check a Cookie](https://ramwise.dev/blog/you-can-check-a-cookie/) |
| [`sql-pipeline-runner`](sql-pipeline-runner/) | `compile → run → assert → emit`: assertions gate the run before bad data lands, and exit codes say who to page | [Keep the Runner Dumb](https://ramwise.dev/blog/keep-the-runner-dumb/) |
| [`evidence-weight`](evidence-weight/) | Belief-strength and agreement as separate axes, independence discounting, and a conflict router that refuses "latest wins" | [Why Evidence Weight and Agreement Need Separate Scores](https://ramwise.dev/blog/weight-isnt-agreement/) |
| [`fastpitch-per-phoneme`](fastpitch-per-phoneme/) | Slow speech per phoneme (hold vowels, keep stops crisp) with FastPitch's per-token `pace` — a GPU/Colab notebook, not zero-dep | [Why I Generated Slower Speech Instead of Stretching Audio](https://ramwise.dev/blog/generate-slow-dont-slow-the-generation/) |
| [`preflight-check`](preflight-check/) | Fingerprint every file in a bulk-load set against a `name,type` baseline; name the golden/broken partition before the load runs — reports structure, never guesses a rename (needs `duckdb`) | [The Files That Break Your Bulk Load](https://ramwise.dev/blog/the-files-that-break-your-bulk-load/) |
| [`fabric-cicd-template`](fabric-cicd-template/) | Azure DevOps wiring for Microsoft Fabric deploys: one project-agnostic template, workspace GUIDs injected at runtime, item folders as the inventory (YAML/config, not runnable Python) | [Deploying Fabric Without a Debugger](https://ramwise.dev/blog/deploying-fabric-without-a-debugger/) |
| [`parquet-gpu-break-even`](parquet-gpu-break-even/) | A correctness-checked PyArrow/cuDF teaching benchmark plus the reproducibility bundle behind the published row-group results | [When Does GPU Parquet Actually Pay Off?](https://ramwise.dev/blog/gpu-parquet-break-even/) |
| [`gpu-data-engineering-boundary`](gpu-data-engineering-boundary/) | An output-complete notebook locating CPU/GPU crossover boundaries across queries, widths, codecs, and a larger-than-VRAM dataset | [DuckDB, Polars, and cuDF on One Analytical Pipeline](https://ramwise.dev/blog/gpu-data-engineering-boundary/) |
| [`spark-rapids-fallback-boundary`](spark-rapids-fallback-boundary/) | An output-complete notebook showing how CPU-only islands and host/device transitions change the value of Spark RAPIDS acceleration | [How Much of a Spark Plan Actually Runs on the GPU?](https://ramwise.dev/blog/spark-rapids-fallback-boundary/) |
| [`cuvs-vector-search-break-even`](cuvs-vector-search-break-even/) | An output-complete cuVS notebook comparing exact search, IVF variants, and CAGRA while calculating how many queries repay ANN index construction | [How Many Queries Pay for a Vector Index?](https://ramwise.dev/blog/gpu-vector-search-break-even/) |
| [`gpu-spatial-analytics-crossover`](gpu-spatial-analytics-crossover/) | An output-complete notebook separating spatial index construction, candidate generation, exact refinement, and the CPU/GPU crossover | [100 Million Points on One GPU](https://ramwise.dev/blog/gpu-spatial-analytics-crossover/) |
| [`gpu-ml-pipeline-boundary`](gpu-ml-pipeline-boundary/) | An output-complete notebook comparing CPU, accelerated, native-GPU, and mixed-placement classical ML pipelines | [I Moved an Entire ML Pipeline to the GPU](https://ramwise.dev/blog/gpu-ml-pipeline-boundary/) |
| [`custom-gpu-computing-cupy-cuda`](custom-gpu-computing-cupy-cuda/) | An output-complete notebook comparing NumPy, composed CuPy, fused CuPy, and handwritten CUDA with launch, transfer, and layout controls | [I Wrote the Same GPU Operation Six Ways](https://ramwise.dev/blog/custom-gpu-computing-cupy-cuda/) |

## Running

Each folder has its own `README` with exact instructions. The compact examples
generally run directly:

```bash
cd tiny-search
python tiny_search.py          # a demo
python test_tiny_search.py     # tests — every folder's tests also run under pytest
```

Notebook and benchmark companions document their own dependency, test, and
container environment. Run those suites from the companion's documented
environment; their GPU and data-engine dependencies are intentionally not
installed at the repository root.

## License

MIT — see [LICENSE](LICENSE).
