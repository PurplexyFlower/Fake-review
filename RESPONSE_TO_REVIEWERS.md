# Response to Reviewer 1 — evidence map

Status legend: **[DONE]** computed on the local RTX 5090; **[CLOUD]** scripted and
queued in `cloud/run_all.sh`, numbers to be filled from the H100/H200 run.

All test metrics are on the exact paper split (split seed 1998; 32,345 / 2,426 /
5,661 train/val/test) and reported both full and leakage-cleaned (5,606 rows).

---

## Headline (multi-seed, replaces the single-run number)

**[DONE]** Rs-QLoRA, rank 64, α=128, over 5 seeds {1998, 7, 42, 123, 2026}:

| metric | mean ± SD |
|---|---|
| test accuracy | **98.72 ± 0.08 %** |
| F1 (CG) | 98.69 ± 0.08 % |
| ROC-AUC | 0.9979 |
| PR-AUC | 0.9966 |
| leakage-cleaned acc (seed 1998) | 98.82 % (vs 98.83 % full) |

---

## NEW DIRECTION — modern AI-fake dataset & detector  [IN PROGRESS]

Motivated directly by the reviewer: the D3 probe shows our detector flags only
**4.0 %** of modern-LLM reviews, i.e. the Kaggle "computer-generated" half is an
obsolete GPT-2-era artifact. We therefore build and study a **modern** dataset:

- **Genuine half** = the dataset's real human reviews (OR, 20,216), unchanged.
- **Fake half** = freshly generated **modern-LLM fake reviews** (LFM2.5-1.2B-Instruct,
  a 2025 instruct model), each *grounded on a paired real review* so category,
  rating and length distributions match exactly and the fakes are product-specific
  and diverse (not templated). `runs/gen_fakes_sota.py` → `dataset/sota_fakes.csv`
  → `runs/build_modern_dataset.py` → `dataset/modern_reviews_dataset.csv`.
- **Detector** = Rs-QLoRA r64 trained on this set (`grids/m_modern.txt`), with a
  classical TF-IDF sanity baseline (to confirm the task is non-trivial) and
  transformer baselines.

This reframes the contribution as **detecting present-day AI-generated fake
reviews** and demonstrates the original CG data is no longer a useful proxy.
(Generator note: LFM2.5-1.2B is a validated *local* prototype; a stronger
non-reasoning instruct model via vLLM on the rented H100 is the quality-upgrade
path. The big reasoning models in LM Studio — qwen3.6-27b etc. — proved unusable
for bulk generation: chain-of-thought only, ~2 tok/s.)

## 1. "Fake" vs "computer-generated" terminology
**[DONE / writing]** Agreed. Re-scope the language to **computer-generated review
detection** throughout; reserve "deceptive/fake" for where it is justified.
**[DONE] D3 modern-LLM probe** (3,000 reviews generated with a current small
instruct model, LFM2.5-350M, across the same 10 categories; length-matched to the
dataset, median 17 words): the headline detector flags only **4.0 %** of them as
computer-generated (mean p_CG = 0.048) — it passes 96 % of modern machine text as
"genuine." This is concrete evidence that the Kaggle "computer-generated" label
captures **generator-specific (GPT-2-era) artifacts, not machine-text in general**,
exactly as the reviewer suspected. We will state this limitation explicitly and
scope the contribution to detecting *that* generator's reviews, not modern fakes.
(Validated: the same eval path reproduces the test accuracy 98.834 % to the digit,
so the 4 % is a real generalization gap, not a loading artifact.)

## 2. Single balanced dataset → external & imbalanced validity
- **[DONE] Imbalance simulation** (prevalence-reweighted, headline model; recall &
  FPR are prevalence-invariant: recall 98.95 %, FPR 1.28 %):

  | CG prevalence | precision | recall | PR-AUC |
  |---|---|---|---|
  | 50 % (test) | 98.73 % | 98.95 % | 0.9975 |
  | 10 % | 89.60 % | 98.95 % | 0.9798 |
  | 1 % | 43.91 % | 98.95 % | 0.8366 |

  → honest: ranking stays strong (PR-AUC), but precision at a fixed 0.5 threshold
  degrades under heavy imbalance — we recommend threshold calibration and report it.
- **[CLOUD] D1 cross-category LOCO**: train on 9 categories, test on each held-out
  category (4 largest). `grids/d1_loco.txt`.
- **[DONE] D3 modern-LLM transfer** (see #1): 4.0 % detection on modern machine
  reviews — the detector does not transfer to a different, newer generator.
- **[CLOUD] D2 external**: zero-shot on the Ott deceptive-opinion corpus (transfer)
  and a genuine-Amazon sample (false-positive rate on out-of-distribution real text).

## 3. State-of-the-art claim under identical conditions
- **[DONE] Same-split classical baselines** (5 seeds, identical split):
  TF-IDF+LogReg **94.31 %**, TF-IDF+LinearSVM **94.91 %** (ROC 0.987 / 0.990).
- **[CLOUD] Same-split transformer baselines**, full fine-tune, 3-LR sweep × 3 seeds:
  RoBERTa-base, DeBERTa-v3-base, ModernBERT-large, Gemma-3-270M — *identical tuning
  budget* (`runs/run_baselines.py`).
- **[DONE / writing]** Soften "state-of-the-art" to "competitive with / exceeding
  strong same-split baselines"; drop cross-paper SOTA comparison as primary evidence.

## 4. Component ablations
- **[DONE] Standard QLoRA vs Rs-QLoRA × rank** (3 seeds each; test acc mean ± SD):

  | rank | standard | rs |
  |---|---|---|
  | 8  | 96.01 ± 0.22 | 97.32 ± 0.34 |
  | 16 | 97.04 ± 0.08 | 97.87 ± 0.08 |
  | 32 | 97.30 ± 0.38 | 98.14 ± 0.17 |
  | 64 | 97.79 ± 0.09 | **98.72 ± 0.08** |

  Rs-QLoRA wins at every rank; rs@r16 (97.87) ≈ standard@r64 (97.79).
- **[DONE] Stability/overfitting** (`analyze_overfit.py`): rs has lower min eval-loss
  at every rank (rs/r64 0.076 vs std/r64 0.099) and converges in fewer steps.
- **[CLOUD] Scheduler & label-smoothing ablations**: cosine-restarts vs cosine vs
  linear; label-smoothing 0.01 vs 0 (`grids/b_ablations.txt`, B grid running).

## 5. Internal-baseline detail + "higher rank trains faster" anomaly
**[DONE]** Explained from convergence, not throughput. Higher rank reaches its best
validation accuracy in fewer optimizer steps and early-stops sooner:

| cell | best_step (to peak val) |
|---|---|
| rs/r8 | 2667 |
| rs/r16 | 2400 |
| rs/r32 | 1300 |
| rs/r64 | 1420 |
| std/r64 | 2433 |

Caveat we state explicitly: wall-clock/`steps_per_s` on the dev laptop are
power-throttle-confounded; `best_step` is the hardware-independent measure. The
**[CLOUD]** run reproduces all timings on one consistent GPU.
**[CLOUD]** ModernBERT-large / Gemma-3-270M get the identical same-split tuning budget (item 3).

## 6. Reproducibility
**[DONE]** Checkpoint `unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit`; head =
`Qwen2ForSequenceClassification` `score`, last-token pooling; trainable params at
r=64 = 35,194,624 / 529,229,184 (6.65 %); 4-bit NF4, LoRA on
q/k/v/o/gate/up/down+score, α=2·rank, max_seq 512, bf16. Split seed 1998
(32,345/2,426/5,661). Versions: torch 2.10.0+cu128, transformers 4.56.2, peft
0.19.1, bitsandbytes 0.49.2, unsloth 2026.6.3, trl 0.22.2. Full env reproduced by
`cloud/setup.sh`. Dedup audit: 36 exact-dup groups (81 rows), 0 label-conflicting
cross-split pairs; metrics reported full and leakage-cleaned.

## 7. Uncertainty
**[DONE]** Every cell is multi-seed (headline 5 seeds) with mean ± SD; ROC-AUC and
PR-AUC reported per run (per-example probabilities saved).

## 8. Memory efficiency
**[DONE]** Headline rs/r64 **training** peak (torch allocator): **2.03 GB allocated /
≤6.06 GB reserved**; allocated grows 1.70→2.03 GB across ranks 8→64. Restate the
"<4 GB" claim as peak *allocated* and note reserved exceeds it.
**[CLOUD]** training VRAM vs batch size {8,16,32,64} (`measure_vram_batch.py`) and
inference latency p50/p95 + peak inference VRAM (`measure_inference.py`).

## 9. Error analysis
**[DONE]** Headline model, 66 errors / 5,661 (confusion: TP 2734, TN 2861,
**FP 37** genuine→CG, **FN 29** CG→genuine):
- **By length**: 10–25 words 3.10 % error → 25–50 0.46 % → 50–100 0.09 % → ≥100 0.00 %.
- **By category**: Clothing 2.97 % (worst) … Movies 0.21 % (best).
- **By confidence**: errors mean conf 0.861 vs correct 0.993; 59 % of errors are
  high-confidence (≥0.90).
- Qualitative pattern (from `misclassified_test.csv`): short generic praise (genuine)
  → flagged CG; specific-but-terse machine text → passes. 6–8 examples for the paper.

## Minor
**[DONE / writing]** Soften "clearly demonstrate", "almost perfect"; check the
standalone JPEG figure pages against journal formatting; split related work into
narrow-binary vs broader-contextual, add the suggested "Trust at risk…" reference.

---

### Artifacts
`results/runs.csv` (all training runs) · `results/a1_summary.csv` ·
`results/baselines*.csv` · `results/external_eval/*.json` ·
`results/inference_bench.json` · `results/vram_batch.json` ·
`dataset/llm_generated_reviews.csv` (D3) · `misclassified_test.csv`.
