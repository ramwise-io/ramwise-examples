"""Check a cookie recipe like a data pipeline: is it physically bakeable?

Companion code for:
https://ramwise.dev/blog/you-can-check-a-cookie/

A cookie is an argument between fat and flour. Past a certain
effective-fat-to-flour ratio the fat wins -- it liquefies before the starch can
set -- and no bake time will rescue the middle. That ratio is a *validation
range*: you can reject an impossible recipe before the oven is ever involved,
the same way you'd reject a bad row before it hits a table.

This is a simplified version of the check (real cocoa-butter content varies by
chocolate). Pure standard library. No dependencies.

Run:  python recipe_ratios.py
"""
from __future__ import annotations

from dataclasses import dataclass

BUTTER_FAT = 0.82      # butter is ~82% fat; the rest is water and milk solids
COCOA_BUTTER = 0.30    # dark/semisweet chocolate is ~30% cocoa butter by weight

# effective-fat-to-flour zones
SAFE_MAX = 0.80        # below this, the structure holds with a proper chill
FAILURE_MIN = 0.90     # at/above this, it will not set at any time or temperature


@dataclass
class Recipe:
    name: str
    butter_g: float
    chocolate_g: float
    flour_g: float
    cocoa_butter: float = COCOA_BUTTER


def effective_fat_to_flour(r: Recipe) -> float:
    """Structural fat (butter fat + the hidden fat in chocolate) over flour."""
    fat = r.butter_g * BUTTER_FAT + r.chocolate_g * r.cocoa_butter
    return fat / r.flour_g


def verdict(ratio: float) -> str:
    if ratio < SAFE_MAX:
        return "safe"
    if ratio < FAILURE_MIN:
        return "borderline"       # chill is non-negotiable; no error tolerance
    return "structural_failure"   # will not set, regardless of technique


@dataclass
class Check:
    recipe: str
    ratio: float
    verdict: str
    bakeable: bool


def check_recipe(r: Recipe) -> Check:
    ratio = effective_fat_to_flour(r)
    v = verdict(ratio)
    return Check(r.name, round(ratio, 3), v, v != "structural_failure")


# The recipe that stayed gooey at 12 minutes, and its correction.
IMPOSSIBLE = Recipe("R2 (as written by an AI)", butter_g=220, chocolate_g=280, flour_g=290)
FIXED = Recipe("R2 (corrected)", butter_g=220, chocolate_g=175, flour_g=305)


if __name__ == "__main__":
    for r in (IMPOSSIBLE, FIXED):
        c = check_recipe(r)
        pct = f"{c.ratio * 100:.0f}%"
        flag = "BAKEABLE" if c.bakeable else "WILL NOT SET"
        print(f"{r.name:26}  effective fat:flour = {pct:>4}  ->  {c.verdict:18} {flag}")
    print()
    print("The AI recipe puts nearly as much chocolate as flour; counting the")
    print("hidden cocoa-butter fat, it lands over the ~90% cliff where a cookie")
    print("cannot set. The check catches it before the oven is ever involved.")
