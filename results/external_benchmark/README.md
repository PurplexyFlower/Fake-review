# External corpora (zero-shot)

The 18 source-grouped Rs-QLoRA adapters were applied **without retraining** to
three independently collected human-versus-LLM review corpora. The adapters are
the matched `sgx{l,d,glm}` and pooled leave-one-generator-out `sgpwo{lfm,deepseek,glm}`
runs, each with seeds 1998, 7, and 42. Each adapter is loaded on the frozen
`unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit` checkpoint as in
`runs/evaluate_source_grouped.py` (512 tokens, softmax P(CG), threshold 0.5).

| Corpus | Source | Licence | Used (OR / CG) |
|---|---|---|---|
| MAiDE-up, English subset (Ignat et al., 2025) | `MichiganNLP/multilingual_reviews_deception` @ `1d622b0` | MIT | 996 / 1,000 |
| DetectAIRev (Agrahari et al., 2025) | HF `Sifi-world/DetectAIRev` @ `d887bce` | none stated | 4,500 / 4,500 (stratified sample, hotel domain excluded) |
| MAGE test split, Yelp rows (Li et al., 2024) | HF `yaful/MAGE` @ `342663f` | Apache-2.0 | 2,652 / 2,547 |

Both classes were cleaned identically: HTML tags and leading template labels
were removed, the training-time `normalize_text()` was applied, and texts under
three words, exact duplicates, texts under both labels, and texts present in any
training file were dropped. Filtering counts are in
`dataset/external_benchmark/manifest.json`. The DetectAIRev hotel domain was
excluded because its human text is distributed lowercased with stop words
removed.

## Files

- `summary.json`: mean ± SD over seeds for each detector and corpus, with
  per-domain and per-generator slices.
- `by_run.csv`: metrics for each run, corpus, and slice.
- `<corpus>_probs.csv`: per-example P(CG), one column per run. Review text is not
  included; the rows are rebuilt from the pinned upstream revisions with
  `runs/prepare_external_benchmark.py` (row order and `row_id` are deterministic).
- `reproduction_checks.json`: before scoring, each adapter re-scored 512 of its
  own stored source-grouped test predictions (mean |Δp| 0.0005 over 9,216 rows;
  2 label changes).

## Reproduce

```bash
python -I runs/prepare_external_benchmark.py --root .
python -I runs/evaluate_external_benchmark.py --root .
```

The raw downloads are not stored in git; `dataset/external_benchmark/sources.json`
records the pinned revisions.
