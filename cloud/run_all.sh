#!/usr/bin/env bash
# Rs-QLoRA fake-review study — one script that runs the experiments, in priority
# order, resumable. A failing phase does NOT abort the rest (set -e is OFF).
#
#   bash cloud/run_all.sh                # SCOPE=modern (default): the current
#                                        # research direction only — modern-fake
#                                        # study + 4x4 cross-generator matrix
#   SCOPE=all bash cloud/run_all.sh      # ...plus the legacy reviewer-rebuttal
#                                        # suite (A1/B/D1 grids, 36 transformer
#                                        # baselines, analyses, efficiency) —
#                                        # MANY extra GPU-hours
#   LANES=2 bash cloud/run_all.sh        # 2 concurrent training lanes
#
# Resumability: every phase skips work already in its ledger. NOTE the repo
# ships results/runs.csv — a fresh clone therefore SKIPS runs completed
# locally (A1/A2, b1 s1998, m1 s1998). That's usually what you want; delete
# results/runs.csv first if you need clean single-hardware timings.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"; cd "$REPO"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1
# shellcheck disable=SC1091
source .venv/bin/activate
PY=python
LANES="${LANES:-1}"
SCOPE="${SCOPE:-modern}"

step(){ echo; echo "============================ $* ============================"; }
opt(){ [ -f "$1" ]; }   # file-exists guard

echo "SCOPE=$SCOPE  LANES=$LANES   (SCOPE=all adds the legacy rebuttal suite)"

# ---- P0: dedup artifacts (cheap; needed by legacy leakage-cleaned metrics) ----
step "P0 dedup audit + exclusions"
opt results/dedup_pairs.csv     || $PY runs/dedup_audit.py
opt results/test_exclusions.csv || $PY runs/make_exclusions.py

# ================= PRIORITY: MODERN-FAKE STUDY + CROSS-GEN =================

# ---- datasets: modern two from HF, GPT-2 built from the shipped Kaggle CSV ----
step "Datasets (fetch/build)"
MOD=dataset/modern_reviews_deepseek.csv
opt "$MOD" || $PY runs/fetch_hf_dataset.py \
  --repo "${MODERN_REPO:-Flowerly/modern-fake-reviews}" --out "$MOD"
opt dataset/modern_reviews_lfm.csv || $PY runs/fetch_hf_dataset.py \
  --repo "${MODERN_LFM_REPO:-Flowerly/modern-fake-reviews-lfm}" \
  --out dataset/modern_reviews_lfm.csv
opt dataset/modern_reviews_gpt2.csv || $PY runs/build_modern_dataset.py \
  --fakes "dataset/fake reviews dataset.csv" --out dataset/modern_reviews_gpt2.csv
# GLM dataset ships in the repo; rebuild from its fakes file if missing
opt dataset/modern_reviews_glm.csv || { opt dataset/sota_fakes_glm.csv && \
  $PY runs/build_modern_dataset.py --fakes dataset/sota_fakes_glm.csv \
      --out dataset/modern_reviews_glm.csv; }

# ---- modern study: Rs-QLoRA + standard QLoRA on the DeepSeek set ----
step "Modern study: Rs-QLoRA r64 x3 + standard r64 x3 (m_modern grid)"
$PY runs/run_queue.py grids/m_modern.txt --lanes "$LANES"

# ---- 4x4 cross-generator matrix (the headline) ----
step "Cross-gen: one detector per generator x3 seeds (xgen grid)"
$PY runs/run_queue.py grids/xgen.txt --lanes "$LANES"
step "Cross-gen LOGO: pooled 3-generator datasets + detectors"
opt dataset/pooled_wo_glm.csv || $PY runs/build_pooled_dataset.py
$PY runs/run_queue.py grids/logo.txt --lanes "$LANES"
step "Cross-gen: transfer matrix + LOGO (TF-IDF and Rs-QLoRA, with FPR)"
$PY runs/cross_gen_matrix.py

# ---- supporting evidence on the modern set ----
step "Modern study: TF-IDF sanity baseline"
grep -q "tfidf_logreg_ds" results/baselines.csv 2>/dev/null || \
  $PY runs/baseline_tfidf.py --data "$MOD" --tag _ds
step "Modern study: head-to-head (by-length + cross-generator recall)"
$PY runs/head_to_head.py || true
step "Modern study: transformer baselines (RoBERTa/DeBERTa/ModernBERT)"
if ! grep -q "_modern" results/baselines_transformer.csv 2>/dev/null; then
  for m in roberta-base microsoft/deberta-v3-base answerdotai/ModernBERT-large; do
    $PY runs/baseline_transformer.py --model "$m" --data "$MOD" --tag _modern \
        --lr 2e-5 --seed 1998 --batch 16
  done
fi

# ======================= LEGACY REBUTTAL SUITE (SCOPE=all) ==================
if [ "$SCOPE" = "all" ]; then
  step "Legacy Phase 1  A1/A2 grid"
  $PY runs/run_queue.py grids/a1_grid.txt --lanes "$LANES"
  step "Legacy Phase 2  B ablations"
  $PY runs/run_queue.py grids/b_ablations.txt --lanes "$LANES"
  step "Legacy Phase 4  D1 LOCO"
  $PY runs/run_queue.py grids/d1_loco.txt --lanes "$LANES"
  step "Legacy Phase 3  baselines (TF-IDF + 4 transformers, LR x seed sweep)"
  $PY runs/run_baselines.py
  step "Legacy analyses  summary / error+imbalance / overfitting"
  $PY runs/summarize_ledger.py
  $PY runs/reviewer_analysis.py
  $PY runs/analyze_overfit.py
  step "Legacy Phase 4  D2/D3 zero-shot evals (need the a1 headline adapter)"
  $PY runs/prepare_external.py || true
  opt dataset/external/ott_deceptive.csv && \
    $PY runs/eval_external.py --csv dataset/external/ott_deceptive.csv \
        --name d2_ott --label-col label --pos-value CG || true
  opt dataset/llm_generated_reviews.csv && \
    $PY runs/eval_external.py --csv dataset/llm_generated_reviews.csv \
        --name d3_llm --label-col label --pos-value CG || true
  step "Legacy Phase 5  efficiency (VRAM vs batch, inference latency)"
  $PY runs/measure_vram_batch.py || true
  $PY runs/measure_inference.py || true
else
  echo; echo "[scope] legacy rebuttal suite skipped (run with SCOPE=all to include)"
fi

step "DONE — results/ (runs.csv, baselines*.csv, a1_summary.csv, external_eval/)
      Headline: the cross-gen matrix printed above (also rerun: python runs/cross_gen_matrix.py)"
