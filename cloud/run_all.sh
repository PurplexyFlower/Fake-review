#!/usr/bin/env bash
# Rs-QLoRA revision — run every reviewer-response experiment, in order, resumable.
# Each phase skips work already recorded in its ledger, so re-running after an
# interruption continues where it left off. A failing phase does NOT abort the
# rest (set -e is intentionally OFF).
#
#   bash cloud/run_all.sh           # 1 GPU lane (default)
#   LANES=2 bash cloud/run_all.sh   # 2 concurrent training lanes (needs VRAM)
#
# Recommended on a fresh box: do NOT copy the local results/ ledger, so all
# numbers come from one consistent GPU (no Windows power-throttle confound).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"; cd "$REPO"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1
# shellcheck disable=SC1091
source .venv/bin/activate
PY=python
LANES="${LANES:-1}"

step(){ echo; echo "============================ $* ============================"; }
opt(){ [ -f "$1" ]; }   # file-exists guard

# ---- P0: dedup artifacts (needed for leakage-cleaned metrics) ----
step "P0 dedup audit + exclusions"
opt results/dedup_pairs.csv     || $PY runs/dedup_audit.py
opt results/test_exclusions.csv || $PY runs/make_exclusions.py

# ---- Phase 1: A1 scaling x rank x seed grid (+ A2 headline seeds) ----
step "Phase 1  A1/A2 grid"
$PY runs/run_queue.py grids/a1_grid.txt --lanes "$LANES"

# ---- Phase 2: single-factor ablations (scheduler, label smoothing) ----
step "Phase 2  B ablations"
$PY runs/run_queue.py grids/b_ablations.txt --lanes "$LANES"

# ---- Phase 4 D1: leave-one-category-out cross-category eval ----
step "Phase 4  D1 LOCO"
$PY runs/run_queue.py grids/d1_loco.txt --lanes "$LANES"

# ---- Phase 3: same-split baselines (TF-IDF + 4 transformers, full FT) ----
step "Phase 3  baselines"
$PY runs/run_baselines.py

# ---- Analyses (CPU, from artifacts on disk) ----
step "Analyses  summary / error+imbalance / overfitting"
$PY runs/summarize_ledger.py
$PY runs/reviewer_analysis.py
$PY runs/analyze_overfit.py

# ---- Phase 4 D2: external corpora (zero-shot transfer + FPR) ----
step "Phase 4  D2 external corpora"
$PY runs/prepare_external.py
opt dataset/external/ott_deceptive.csv && \
  $PY runs/eval_external.py --csv dataset/external/ott_deceptive.csv \
      --name d2_ott --label-col label --pos-value CG
opt dataset/external/amazon_genuine.csv && \
  $PY runs/eval_external.py --csv dataset/external/amazon_genuine.csv \
      --name d2_amazon_fpr

# ---- Phase 4 D3: modern-LLM reviews (csv generated locally, shipped in repo) ----
step "Phase 4  D3 modern-LLM detection"
opt dataset/llm_generated_reviews.csv && \
  $PY runs/eval_external.py --csv dataset/llm_generated_reviews.csv \
      --name d3_llm --label-col label --pos-value CG

# ---- MODERN-FAKE STUDY (redirected primary direction) ----
# Real human reviews (OR) vs DeepSeek-v4-pro fakes (CG), pulled from the Hub.
MOD=dataset/modern_reviews_deepseek.csv
step "Modern study: fetch dataset from HF"
opt "$MOD" || $PY runs/fetch_hf_dataset.py --repo "${MODERN_REPO:-Flowerly/modern-fake-reviews}" --out "$MOD"
if opt "$MOD"; then
  step "Modern study: Rs-QLoRA r64 x3 seeds + standard QLoRA r64 (m_modern grid)"
  $PY runs/run_queue.py grids/m_modern.txt --lanes "$LANES"
  step "Modern study: classical sanity baseline (TF-IDF)"
  $PY runs/baseline_tfidf.py --data "$MOD" --tag _modern
  step "Modern study: transformer baselines"
  for m in roberta-base microsoft/deberta-v3-base answerdotai/ModernBERT-large; do
    $PY runs/baseline_transformer.py --model "$m" --data "$MOD" --tag _modern \
        --lr 2e-5 --seed 1998 --batch 16
  done
  step "Modern study: head-to-head TF-IDF vs Rs-QLoRA (by length) + cross-generator"
  # cross-generator needs the 2nd generator's (1.2B) fakes; fetch from HF if absent
  LFM=dataset/modern_reviews_lfm.csv
  opt "$LFM" || opt dataset/sota_fakes.csv || \
    $PY runs/fetch_hf_dataset.py --repo "${MODERN_LFM_REPO:-Flowerly/modern-fake-reviews-lfm}" --out "$LFM" || true
  $PY runs/head_to_head.py || true
else
  echo "[skip] could not obtain $MOD (need internet for HF, or ship the CSV)"
fi

# ---- 3x3 CROSS-GENERATOR MATRIX (the robustness headline) ----
step "Cross-gen 3x3: datasets"
# normalized OR-vs-GPT2 from the shipped Kaggle set; modern two from HF
opt dataset/modern_reviews_gpt2.csv || \
  $PY runs/build_modern_dataset.py --fakes "dataset/fake reviews dataset.csv" \
      --out dataset/modern_reviews_gpt2.csv
opt dataset/modern_reviews_lfm.csv || \
  $PY runs/fetch_hf_dataset.py --repo "${MODERN_LFM_REPO:-Flowerly/modern-fake-reviews-lfm}" \
      --out dataset/modern_reviews_lfm.csv
opt dataset/modern_reviews_deepseek.csv || \
  $PY runs/fetch_hf_dataset.py --repo "${MODERN_REPO:-Flowerly/modern-fake-reviews}" \
      --out dataset/modern_reviews_deepseek.csv
step "Cross-gen 3x3: train one detector per generator (xgen grid)"
$PY runs/run_queue.py grids/xgen.txt --lanes "$LANES"
step "Cross-gen 3x3: TF-IDF vs Rs-QLoRA transfer matrix"
$PY runs/cross_gen_matrix.py

# ---- Phase 5: efficiency (training VRAM vs batch, inference latency) ----
step "Phase 5  efficiency"
$PY runs/measure_vram_batch.py
$PY runs/measure_inference.py

step "ALL DONE — see results/ (runs.csv, baselines*.csv, a1_summary.csv,
       external_eval/, inference_bench.json, vram_batch.json)"
