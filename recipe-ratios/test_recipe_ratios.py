"""Tests for recipe_ratios.py. Run with `python test_recipe_ratios.py` or `pytest`."""
from __future__ import annotations

from recipe_ratios import (
    FIXED,
    IMPOSSIBLE,
    Recipe,
    check_recipe,
    effective_fat_to_flour,
    verdict,
)


def test_impossible_recipe_will_not_set():
    c = check_recipe(IMPOSSIBLE)
    assert c.verdict == "structural_failure"
    assert c.bakeable is False
    assert c.ratio >= 0.90


def test_corrected_recipe_is_safe():
    c = check_recipe(FIXED)
    assert c.verdict == "safe"
    assert c.bakeable is True
    assert c.ratio < 0.80


def test_the_fix_is_mostly_less_chocolate():
    assert FIXED.butter_g == IMPOSSIBLE.butter_g          # butter unchanged
    assert FIXED.chocolate_g < IMPOSSIBLE.chocolate_g     # chocolate cut


def test_verdict_boundaries():
    assert verdict(0.79) == "safe"
    assert verdict(0.85) == "borderline"
    assert verdict(0.95) == "structural_failure"


def test_chocolate_carries_hidden_fat():
    # adding chocolate raises effective fat even with flour and butter held fixed
    base = Recipe("base", butter_g=200, chocolate_g=0, flour_g=300)
    choc = Recipe("choc", butter_g=200, chocolate_g=200, flour_g=300)
    assert effective_fat_to_flour(choc) > effective_fat_to_flour(base)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
