# Published evidence

- Benchmark matrix: `cuvs-break-even-full-v1-da875e2a0b`
- Measurement source: `gpu-lab` commit `3938ebd`
- Derived-report source: `gpu-lab` commit `e9fb7f7`
- Confirmed operating points: 90
- Measured but unconfirmed controls: 6

`published_results.csv` contains exact reference rows and ANN operating points
that retained at least one predeclared recall target across all three
confirmation replications. `unconfirmed_controls.csv` preserves selected
settings whose recall fell below every target in at least one fresh build.

Raw per-trial results and private host telemetry are not published. The full
derivation code is included under `../full-benchmark/`.
