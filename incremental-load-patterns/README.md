# incremental-load-patterns

Runnable examples behind **[Incremental Load Is Not One Thing](https://ramwise.dev/blog/incremental-load-patterns/)**.
Pure standard library (`sqlite3`), no dependencies.

"Incremental load" is not one technique — it's a family, and the right move is
usually the *weakest* one that still guarantees correctness for how your source
behaves. This example makes the sharpest failure mode concrete: the **tie bug**
in a naive timestamp watermark, and the composite-watermark fix.

## The tie bug, live

Four rows share one `modified_at` (they were written by a single transaction).
A run that stops partway through that group — a crash, a timeout, a page
boundary — saves `max(modified_at)` as its watermark, then next time asks for
`modified_at > that timestamp`. The tied rows it never reached are `=`, not `>`,
so they're excluded **forever**, silently:

```bash
python incremental.py
```
```
source rows:             [1, 2, 3, 4, 5]
naive watermark loaded:  [1, 2, 5]   <- dropped [3, 4] silently
composite watermark:     [1, 2, 3, 4, 5]   <- all rows, no loss
```

The fix is a composite `(timestamp, id)` keyset, paging in the same order it
compares:

```sql
WHERE modified_at > @last_ts
   OR (modified_at = @last_ts AND id > @last_id)
ORDER BY modified_at, id
```

## Run the tests

```bash
python test_incremental.py   # also works under: pytest
```

The tests assert that the naive watermark loses rows and that the composite one
loses nothing across every batch size.
