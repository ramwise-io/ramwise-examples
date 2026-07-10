# recipe-ratios

Runnable example behind **[You Can Check a Cookie](https://ramwise.dev/blog/you-can-check-a-cookie/)**.
Pure standard library, no dependencies.

A cookie recipe is a set of ratios, and ratios have physical limits. Past a
certain **effective fat-to-flour** ratio the fat liquefies before the starch
can set, and no bake time rescues the middle — the cookie stays gooey. That
makes a recipe *checkable*: you can reject an impossible one before the oven is
ever involved, exactly like validating a row before it hits a table.

```bash
python recipe_ratios.py
```
```
R2 (as written by an AI)    effective fat:flour =  91%  ->  structural_failure WILL NOT SET
R2 (corrected)              effective fat:flour =  76%  ->  safe               BAKEABLE
```

The "impossible" recipe puts nearly as much chocolate as flour; once you count
the hidden cocoa-butter fat, it lands over the ~90% cliff. The corrected one —
mostly just less chocolate — drops back into the safe zone. Same check, before
anything is baked.

(Effective fat = butter × 0.82 + chocolate × ~0.30, over flour. The exact
threshold depends on the chocolate's real cocoa-butter content; this is the
simplified version.)

## Run the tests

```bash
python test_recipe_ratios.py   # also works under: pytest
```

The tests assert that the AI recipe fails the check, the corrected one passes,
and that chocolate's hidden fat counts toward the ratio.
