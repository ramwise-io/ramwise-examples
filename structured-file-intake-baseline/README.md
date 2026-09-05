# Structured File Intake — Robust, Flexible Baseline

Companion material for **[I Tried to Make Structured File Intake Boring](https://ramwise.dev/blog/i-tried-to-make-structured-file-intake-boring/)** on [ramwise.dev](https://ramwise.dev).

The prototype ran inside Anatini, the local notebook-based data platform I used
for the experiment. The reusable subject here is the file-intake framework and
its guarantees, not the host platform.

The experiment asks a deliberately narrow question:

> Can a normal file source be onboarded mostly through configuration while still
> getting reliable identity, validation, duplicate protection, rejected-row
> handling, provenance, and safe retries?

The same engine processes five YAML source definitions spanning three unrelated
public-data shapes: synthetic clinical-trial data, CMS NPPES-shaped provider
archives, and GTFS transit feeds.

## Follow the notebook evidence

Read the three static marimo snapshots in this order:

- [Framework self-test](https://ramwise.dev/notebooks/structured-file-intake/framework-self-test.html) — verifies
  the intentionally small reusable surface: configuration loading, streaming
  identity, and source execution.
- [Ingestion Job and duplicate-safe rerun](https://ramwise.dev/notebooks/structured-file-intake/intake-examples.html) — shows the
  examples notebook loading the shared framework, retrieving `run_source()`, and
  rediscovering all configured files without loading another copy of accepted rows.
- [Verified results](https://ramwise.dev/notebooks/structured-file-intake/intake-results.html) — queries the
  retained DuckLake Intake state for delivery, attempt, rejection, supersession,
  and accepted-row provenance evidence.

Those links open rendered notebooks on ramwise.dev. Downloadable copies are kept
under [`exports/`](exports/) as retained artifacts; GitHub displays their HTML
source rather than rendering it. The hosted pages use the same files with pinned,
self-hosted marimo assets and do not execute the ingestion again.

## Captured result

At the published snapshot:

- 5 configured sources had retained deliveries;
- 8 content-identified deliveries existed: 5 loaded and 3 superseded;
- successful load attempts read 40 rows, accepted 38, and rejected 2;
- renamed byte-identical files and subsequent reruns produced 52 duplicate
  attempts without duplicating accepted rows; and
- every accepted Raw row carried delivery, attempt, Run, package-member,
  source-row, and load-time provenance.

The current snapshot contains 52 duplicate attempts. That count is intentionally
a point-in-time figure: safe reruns add
duplicate-attempt evidence while leaving the accepted data unchanged.

## What calls what

The implementation was tested as two projects rather than one blended notebook:

```text
Intake Framework
└── notebooks/intake.py
    └── returns load_source() and run_source()
                 ▲
                 │ execute once and retrieve run_source
                 │
Intake Examples
├── config/intake/*.yaml
├── files/inbox/*
└── notebooks/ingest_sources.py
    └── run_source(source_id) for five configured sources
        ├── accepted rows → intake_examples.raw
        └── delivery state → intake_examples.intake
```

The results notebook is downstream of that path: it queries retained state and
provenance but does not ingest the files itself.

## Evidence boundary

Only output-only notebook snapshots are published here. The embedded notebook
source fields present in standard marimo exports have been removed while retaining
the captured rendered output. There is no separate notebook source archive,
fixture bundle, DuckLake data directory, Run bundle, package environment, cache,
credential, or internal filesystem path.

The tiny fixtures are deterministic and public-data-shaped. They demonstrate the
portability of the ingestion approach; they are not SDTM, CMS, NPPES, or full GTFS
conformance implementations. The optional official NPPES large-file exercise is
also outside this baseline publication bundle.
