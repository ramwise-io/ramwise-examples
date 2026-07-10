# evidence-weight

Runnable example behind **[Weight Isn't Agreement](https://ramwise.dev/blog/weight-isnt-agreement/)**.
Pure standard library (`math`), no dependencies.

Most systems hand you one number for a claim, doing two jobs at once: *how much
support does this have* and *are the sources fighting about it*. Collapsing
those into a single "confidence" produces a specific lie — a high number on a
claim where the good sources are evenly split. This example keeps the two axes
apart, and shows two moves that matter:

```bash
python evidence.py
```
```
Volume can't buy truth (independence discounting):
  3 independent sources        -> weight 0.973  (consensus)
  5 copies of one blog         -> weight 0.853  (singular)   [naive count would say 0.998]

Weight is not agreement (adding opposition):
  before opposition            -> weight 0.973  (consensus)
  after 1 opposing source      -> weight 0.973  (contested)   <- weight unchanged, label flipped

Two observations disagree -> route, don't pick a winner:
  diff sources, same time      -> contestation
  same source, later time      -> self_correction
```

- **`compute_belief()`** — weight from supporting mass only (opposition never
  lowers it), contestation as a separate label, returned together. Correlated
  sources (same source or identical text) collapse toward one voice, so five
  reposts don't outweigh three real witnesses.
- **`classify_pair()`** — two observations about the same thing get *routed* by
  a small truth table (same source correcting itself? different sources at the
  same time disagreeing? one claim reinforced?), never resolved by "latest
  wins," which would discard the disagreement that was the whole point.

A teaching distillation of the ideas — not the full system.

## Run the tests

```bash
python test_evidence.py   # also works under: pytest
```

The tests assert that opposition doesn't move the weight, that duplicated
sources can't inflate it, that a claim can be high-weight *and* contested, and
that the router preserves genuine disagreement instead of picking a winner.
