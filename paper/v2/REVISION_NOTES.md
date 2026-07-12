# Revision notes

This is a scientific redesign of v1, not a line edit. The original manuscript
and both reviewer reports remain unchanged in `paper/v1/`.

## New center of the paper

The paper now asks what transfers across generators. Matched accuracy is only a
baseline; the primary evidence is the four-by-four transfer matrix, pooled
generator holdout, negative control, matched baselines, and efficiency comparison.

## Claims removed or corrected

- Removed the unsupported state-of-the-art claim.
- Replaced “fake/deceptive” with OR/CG terminology except when discussing work
  that actually labels deception.
- Removed the old single-run 98.43% headline and the hardware-mixed 4.3-hour
  comparison.
- States that DeBERTa-v3-base, not Rs-QLoRA, has the highest matched accuracy.
- Frames Rs-QLoRA's demonstrated advantage as convergence efficiency rather than
  accuracy superiority.
- Replaced the false “no generative AI used” declaration with a transparent draft
  disclosure that must be checked against the journal's current policy.

## New source-overlap finding

The modern datasets retain source IDs in their raw generation artifacts. An audit
shows that 69.3–81.5% of target test prompt sources occur in another modern
generator's training split; pooled LOGO overlap is 94.3–94.5%. Therefore:

- the transfer matrix does hold out generated texts and, in LOGO, exact target
  human texts;
- it holds out generator identity;
- it does **not** jointly hold out prompt-source identity;
- the correct claim is transfer under shared grounded-source support, not fully
  out-of-distribution generalization.

The direction asymmetries remain informative because opposite directions have
nearly identical source-overlap rates. Reproduce the audit with
`python runs/analyze_source_overlap.py`; the structured output is
`results/source_overlap.json`.

## Remaining submission work

- Confirm the journal's exact generative-AI disclosure wording.
- Compile using the Frontiers class and inspect table placement/line wrapping.
- Have all authors verify author contributions, URLs, and provider naming.
- If compute becomes available, repeat LOGO across seeds and run a source-grouped
  generator holdout. The manuscript already labels these as limitations and does
  not depend on them being completed.
