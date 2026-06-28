# Dedup audit (P0-1)

- Rows: 40432
- Exact duplicate groups (normalized text): 36 (81 rows involved)
- Exact-dup groups spanning multiple splits (LEAKAGE): 17
- Near-duplicate pairs (cosine >= 0.9): 241
- Near-dup pairs ACROSS splits (LEAKAGE): 75 {('train', 'val'): 19, ('test', 'train'): 51, ('test', 'val'): 5}
- Near-dup pairs with CONFLICTING labels (OR vs CG): 0

Pairs detail: results/dedup_pairs.csv (sorted by similarity).

## Verdict guidance
- Cross-split exact dups or many cross-split near-dups => re-split with
  group-aware splitting before Phase 1 and report both numbers.
- OR-vs-CG near-dup pairs are expected by construction (the CG reviews
  were generated FROM real ones) — report them, they are not leakage.