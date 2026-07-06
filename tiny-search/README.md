# tiny-search

A vector-space search engine built from scratch — inverted index, TF-IDF,
cosine ranking, and a minimal boolean layer. Pure standard library, no
dependencies.

Companion code for **[I Built a Search Engine by Hand](https://ramwise.dev/blog/i-built-a-search-engine-by-hand/)**.

It exists to make two lessons concrete:

1. **An inverted index is the right *shape* for the data.** A dense
   document-term matrix stores a cell for every term in every document — almost
   all of them zero. The inverted index (`term -> {doc_id: freq}`) stores only
   the postings that exist. See `test_inverted_index_stores_only_postings`.

2. **Scoring only touches the documents a query term points to.** You walk the
   postings lists for the query's terms and score those — never the whole
   corpus. That, not the cosine formula, is why inverted indexes exist. See
   `test_scoring_touches_only_matching_docs`.

## Run it

```bash
python tiny_search.py      # a tiny demo
python test_tiny_search.py # tests (also works under: pytest)
python benchmark.py        # the "demo looks great, benchmark disagrees" lesson
```

## The benchmark

`benchmark.py` reproduces the humbling part of the post: the engine looks great
by eye, then a benchmark against relevance judgments reveals what two
plausible-looking changes actually do —

```
mean precision@3, plain queries ......... 0.83
mean precision@3, + query expansion ..... 0.50   <- traps sneak in; precision falls
mean recall, no cutoff .................. 1.00
mean recall, + hard score cutoff ....... 0.83   <- relevant low-scorers dropped
```

The demo looked the same the whole time. The benchmark did not.

## What this is not

This is a teaching implementation, freshly written for the post. For real work,
use a real search library or a vector database — and, either way, keep a
benchmark that can embarrass you.
