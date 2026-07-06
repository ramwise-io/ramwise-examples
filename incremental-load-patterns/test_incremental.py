"""Tests for incremental.py. Run with `python test_incremental.py` or `pytest`."""
from __future__ import annotations

from incremental import _drain_composite, _drain_naive, make_source

ALL_IDS = {1, 2, 3, 4, 5}


def test_naive_watermark_loses_tied_rows():
    # batch_limit=2 stops the run inside the tie group (ids 1..4 share a ts)
    loaded = set(_drain_naive(make_source(), batch_limit=2))
    assert loaded != ALL_IDS, "expected the naive watermark to drop rows"
    assert loaded.issubset(ALL_IDS)


def test_composite_watermark_loses_nothing():
    loaded = set(_drain_composite(make_source(), batch_limit=2))
    assert loaded == ALL_IDS


def test_composite_is_robust_across_batch_sizes():
    for limit in (1, 2, 3, 4, 5, 10):
        assert set(_drain_composite(make_source(), batch_limit=limit)) == ALL_IDS


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
