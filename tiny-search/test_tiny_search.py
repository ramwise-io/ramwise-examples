"""Tests for tiny_search. Run with `python test_tiny_search.py` or `pytest`."""
from __future__ import annotations

from tiny_search import TinySearch, boolean, tokenize


def _engine() -> TinySearch:
    eng = TinySearch()
    eng.add("d1", "international travel guide flights visas")
    eng.add("d2", "budget travel cheap flights")
    eng.add("d3", "home gardening winter")
    eng.index()
    return eng


def test_tokenize():
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_inverted_index_stores_only_postings():
    eng = _engine()
    # "travel" appears in d1 and d2 only -- not d3.
    assert set(eng.postings["travel"]) == {"d1", "d2"}
    # a term that never appears has no postings entry at all (no zeros stored).
    assert "gardening" in eng.postings and "d1" not in eng.postings["gardening"]


def test_scoring_touches_only_matching_docs():
    eng = _engine()
    scores = eng.scores("gardening")
    # only d3 contains the term, so only d3 is scored
    assert set(scores) == {"d3"}


def test_search_ranks_relevant_first():
    eng = _engine()
    ranked = eng.search("international travel", k=3)
    assert ranked[0][0] == "d1"
    assert "d3" not in [d for d, _ in ranked]


def test_boolean():
    eng = _engine()
    assert boolean(eng, "travel", "AND", "flights") == {"d1", "d2"}
    assert boolean(eng, "travel", "NOT", "budget") == {"d1"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
