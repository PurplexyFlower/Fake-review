---
name: gen-fake-reviews
description: Generate modern AI-fake product reviews directly through Claude Code (no external API), grounded on real reviews, and append them to dataset/sota_fakes_claude.csv. Use when the user wants to add Claude-generated fakes to the dataset / cross-generator matrix. Args: optional batch size N (default 40).
---

# Generate modern AI-fake reviews via the harness

You (the model running this harness) are the generator — **do not call any external
API**. Produce the review text yourself and let the helper handle CSV I/O and
normalisation. Each fake is grounded on a real human review so it matches that
product's category, star rating, and length.

## Workflow (one batch per invocation)

1. **Get the worklist** (default N=40; use the number the user gave):
   ```
   .venv/Scripts/python.exe runs/claude_gen_io.py next 40
   ```
   (POSIX: `.venv/bin/python`.) It prints JSON: a list of
   `{src_idx, category, rating, target_len, real_text}` for reviews not yet generated.

2. **Write one new fake per item**, following the prompt rules below. Build a JSON
   array `[{ "src_idx": <id>, "text": "<your review>" }, ...]` and write it to the
   scratchpad with the Write tool.

3. **Append** (the helper normalises punctuation to ASCII and looks up category/rating):
   ```
   .venv/Scripts/python.exe runs/claude_gen_io.py append --in <scratch.json> --out dataset/sota_fakes_claude.csv
   ```

4. Report how many were appended and the running total. To continue, the user
   re-invokes the skill — it resumes (skips `src_idx` already written).

## The generation prompt (apply per item)

Act as a real customer writing a product review. For each worklist item, write a
**new, different** `{rating}`-star review for the **same product** as the
`real_text`, as if another genuine buyer wrote it.

- **Length:** about `target_len` words (match it — short stays short).
- **Ground, don't copy:** use `real_text` only to know the product; share **< ~40%**
  of its words. No paraphrasing of its sentences.
- **Vary the angle** across the batch (rotate, don't repeat): overall experience ·
  a specific feature liked/disliked · value for the price · how it compares to
  expectations · durability after some use · who you'd recommend it to · ease of
  use/setup. (Avoid making every review about shipping/packaging — that became a
  detectable tell.)
- **Voice:** natural, casual, specific. Plain everyday words. Vary openings — do
  **not** start most reviews with "Great", "Love", "I", or "This".
- **Output only the review text** — no preamble, quotes, titles, rating lines,
  markdown, or emoji.
- Keep the star sentiment consistent (1–2 = mostly negative, 4–5 = mostly positive,
  3 = mixed).

## Notes

- Output is `dataset/sota_fakes_claude.csv` (cols `src_idx,category,rating,label,text_,gen_model`,
  `gen_model=claude-code`). Build a balanced dataset with
  `runs/build_modern_dataset.py --fakes dataset/sota_fakes_claude.csv --out dataset/modern_reviews_claude.csv`,
  then add `("Claude", .../modern_reviews_claude.csv, "xc")` to `GENERATORS` in
  `runs/cross_gen_matrix.py` to include it as a fourth generator.
- This generates **one harness batch at a time** (tens–low-hundreds), not 20k — it's
  for quality samples and a Claude-generated generator, not the full corpus.
- Honesty: these are AI-generated, labelled `CG`. Don't claim they're human.
