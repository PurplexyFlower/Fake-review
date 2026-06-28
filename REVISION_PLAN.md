# Rs-QLoRA Revision Plan (Reviewer Response Experiments)

Source: Reviewer 1 (18 May 2026) + Reviewer 2 (15 May 2026).
All runs on RTX 5090 Laptop (24 GB), bf16, Windows, `.venv` (torch 2.10.0+cu128,
transformers 4.56.2, peft 0.19.1, unsloth 2026.6.3, bitsandbytes 0.49.2, trl 0.22.2).
Hardware change from the paper's T4/fp16 is documented in the revision.

## Progress (session 2026-06-16)

Evidence map for the rebuttal lives in **`RESPONSE_TO_REVIEWERS.md`**. Cloud
automation (rent H100/H200, `uv` env, run everything) in **`cloud/`**
(`setup.sh`, `run_all.sh`, `README.md`).

- **[DONE locally]** A1/A2 grid summarized (`runs/summarize_ledger.py` →
  `results/a1_summary.csv`); error analysis + imbalance + leakage-clean
  (`runs/reviewer_analysis.py`); overfitting/stability (`runs/analyze_overfit.py`);
  C1/C2 TF-IDF baselines (`runs/baseline_tfidf.py` → `results/baselines.csv`,
  94.31 / 94.91 %); training VRAM-by-rank + E4 timing (in `a1_summary.csv`).
- **[RUNNING locally]** B ablations grid (background); D3 modern-LLM review
  generation via LM Studio (`runs/gen_llm_reviews.py` → `dataset/llm_generated_reviews.csv`).
- **[SCRIPTED → cloud]** transformer baselines C3/C4/C5 (`runs/run_baselines.py`,
  `baseline_transformer.py`), D1 LOCO (`grids/d1_loco.txt`), D2 external
  (`runs/prepare_external.py` + `runs/eval_external.py`), D3 eval
  (`eval_external.py`), inference latency + batch VRAM sweep
  (`measure_inference.py`, `measure_vram_batch.py`).
- `runs/_paths.py` resolves the headline run dir dynamically so a fresh cloud
  rerun (new timestamps) works; `run_queue.py` is now OS-aware (bin vs Scripts).

## Conventions (read once)

- **Frozen protocol** (every run unless the row says otherwise): Qwen2.5-0.5B
  4-bit (`unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit`), LoRA on
  q/k/v/o/gate/up/down + `score` head, batch 32 x accum 2, grad-checkpointing
  "unsloth", lr 5e-6, warmup 0.06, 6 epochs max, adamw_8bit, wd 0.03,
  cosine_with_restarts (2 cycles), label smoothing 0.01, eval/save every 100
  steps, early stop patience 8, load best (val accuracy), max_seq 512, bf16.
- **Split seed is ALWAYS 1998** (reproduces the paper's exact 32345/2426/5661
  split). `--seed` only changes training randomness. Never change the split seed.
- **Seed set**: 1998, 7, 42 (3-seed cells); + 123, 2026 for 5-seed headline.
- **Alpha rule**: alpha = 2 x rank everywhere (paper: r=64, alpha=128), so the
  standard-scaling factor alpha/r = 2 is constant across ranks and the only
  difference between `standard` and `rs` is the /r vs /sqrt(r) denominator.
- Every run: `python runs/train_run.py ...` writes `results/<run_id>/`
  (config.json, metrics.json, test_probs.csv, val_probs.csv, adapter/) and
  appends one row to `results/runs.csv` (the ledger — single source of truth).
- Queue runs with `python runs/run_queue.py grids/<file>.txt --lanes 1|2`.
- Per-example probabilities are always saved -> ROC-AUC / PR-AUC / threshold /
  imbalance / error analyses never require retraining.

## Phase 0 — De-risk (BEFORE any GPU spend)

- [ ] **P0-0 Fix GPU power cap.** Card draws ~33 W at 88 C (should be 100-175 W).
      Plugged in + vendor performance profile + cooling. All time estimates
      below assume this is fixed. Verify: `nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv` during a run.
- [x] **P0-1 Dedup audit** (R1-6): DONE 12 Jun 2026 -> `results/dedup_report.md`.
      Verdict: minor leakage, NOT material — 36 exact-dup groups (81/40432 rows),
      241 near-dup pairs (>=0.90), 75 cross-split (51 train<->test), 0 label
      conflicts. Max possible headline shift ~0.01 pp => keep the paper split.
      **Convention: all test metrics reported BOTH full (5661) and
      leakage-cleaned (5606 rows; exclusions in `results/test_exclusions.csv`).**
      Dedup methodology goes in the reproducibility appendix.
- [ ] **P0-2 Concurrency A/B**: run one A1 line solo, note steps/sec from the
      ledger; then two lines with `--lanes 2`; keep whichever aggregate is higher.

## Phase 1 — Rs-QLoRA central claim (R2-Req1, R1-4) — THE SPINE

- [ ] **A1 grid**: {standard, rs} x rank {8,16,32,64} x seeds {1998,7,42} = 24 runs
      `python runs/run_queue.py grids/a1_grid.txt --lanes 2`
- [ ] **A2**: rs/r64 extra seeds {123, 2026} (5-seed headline cell) — included
      at the end of `grids/a1_grid.txt`.
- [ ] Deliverable: table mean+-std of test acc / F1 / ROC-AUC / PR-AUC per cell;
      val-loss curves overlay per rank (stability claim). Honest outcome either way.

## Phase 2 — Single-factor ablations (R1-4)

- [ ] **B1** scheduler=cosine (no restarts) x3 seeds
- [ ] **B2** scheduler=linear x3 seeds
- [ ] **B3** label_smoothing=0 x3 seeds
      `python runs/run_queue.py grids/b_ablations.txt --lanes 2`

## Phase 3 — Same-split baselines (R2-Req3, R1-3, R1-5)

- [ ] **C1/C2** TF-IDF + LogReg / linear SVM (CPU, same split, 5 seeds) — script TBD
- [ ] **C3** RoBERTa-base full FT: LR sweep {1e-5,2e-5,5e-5} @1 seed -> best x3 seeds
- [ ] **C4** DeBERTa-v3-base: same recipe
- [ ] **C5** ModernBERT-large + Gemma-3-270M re-run on OUR split, same sweep budget
- [ ] Rebuttal sentence: "every baseline received the identical tuning budget."

## Phase 4 — OOD & external validity (R2-Req2, R1-2, R1-1)

- [ ] **D1** Cross-category LOCO: `train_run.py --holdout-category <cat>` for the
      4 largest categories x1 seed
- [ ] **D2** Zero-shot eval of the 5 headline models on Ott et al. deceptive-opinion
      corpus (and/or YelpChi) — eval-only script
- [ ] **D3** Generate ~2-5k modern-LLM reviews (same categories), zero-shot eval
      (answers "GPT-2-era artifacts" + the CG-vs-deceptive terminology point)
- [ ] **D4** Imbalance simulation (10% / 1% fake prevalence): post-process saved
      test_probs.csv; report PR-AUC + precision at fixed operating points

## Phase 5 — Efficiency (R2-Req4, R1-8) — partially DONE

- [x] Training VRAM measured (`measure_vram.py`): paper config = **~1.0 GB
      allocated / ~1.2 GB torch-reserved / ~5.9 GB total process (nvidia-smi)**.
      Report ALL THREE + tools; restate the "<4 GB" claim in allocator terms.
- [ ] Batch-size dependence: re-run measure_vram.py at batch {8,16,32,64}
- [ ] **E2** Inference: batch-1 latency p50/p95 over 1k samples, throughput @64,
      peak inference VRAM, 4-bit vs merged-fp16
- [x] Throughput per training run: auto-logged to ledger (steps_per_s)
- [ ] **E4** r=16 vs r=64 timing: read directly from A1 ledger rows; write the
      honest explanation of the paper's timing anomaly

## Phase 6 — Error analysis (R1-9) — partially DONE

- [x] 64 test errors dumped with confidence/category/rating (`misclassified_test.csv`)
- [ ] Error rate vs review length (binned), per-category table, confidence
      histograms (errors vs correct), 6-8 qualitative examples in the paper
      (pattern found: generic-praise genuine -> flagged fake; specific-but-
      incoherent fake -> passes)

## Writing-only (no GPU)

- [ ] Soften "state-of-the-art", "clearly demonstrate", "almost perfect"
- [ ] Terminology: "computer-generated review detection" or justify via D3
- [ ] Related-work subsections + the 9 suggested references
- [ ] Reproducibility appendix: exact checkpoint, head/pooling
      (Qwen2ForSequenceClassification `score`, last-token pooling), trainable
      params 35,194,624/529,229,184 (6.65%), versions, seeds, split procedure,
      dedup result from P0-1
- [ ] Multi-seed tables with mean+-std; add ROC-AUC + PR-AUC everywhere

## Budget

~40 GPU runs ~= 65-85 h at fixed clocks (halve wall-time if 2 lanes help).
Order: P0 -> A1 -> C -> B -> D, with E/F on CPU in parallel.
