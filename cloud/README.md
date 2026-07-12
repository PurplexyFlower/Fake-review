# Running the Rs-QLoRA experiments on a rented GPU (H100/H200)

Two scripts, `uv` for a reproducible env. The **modern fake-review study** is the
primary direction; the original reviewer-rebuttal experiments still run too.

## What to copy to the instance

Just the code — **datasets are pulled from the Hub**, not shipped:

```
runs/   grids/   cloud/   REVISION_PLAN.md
dataset/"fake reviews dataset.csv"        # original Kaggle set (rebuttal phases)
```

Modern datasets are fetched automatically from HF:
- `Flowerly/modern-fake-reviews`     — real OR vs **DeepSeek-v4-pro** fakes (primary)
- `Flowerly/modern-fake-reviews-lfm` — real OR vs **LFM2.5-1.2B** fakes (cross-gen)

(Override repo ids with `MODERN_REPO` / `MODERN_LFM_REPO` if you forked them.)
Exact generation prompts, sampling settings, artifact hashes, provider records,
and known unknowns are maintained in `DATA_PROVENANCE.md` and
`dataset/provenance.json`.

## 1. Setup

```bash
bash cloud/setup.sh
```

Installs `uv`, creates `.venv` (Python 3.12), and the pinned stack (torch 2.10
cu128, transformers 4.56.2, peft 0.19.1, bitsandbytes 0.49.2, unsloth 2026.6.3,
trl 0.22.2, scikit-learn/scipy/sentencepiece, openai, datasets). Ends with a CUDA
+ import sanity block — **read it before continuing.** The torch/unsloth pin is the
one part not testable off-Linux; if it complains, the re-pin lines are the knobs.

**Gemma-3-270m** (a transformer baseline) is gated: `export HF_TOKEN=hf_...` first,
or drop it from `runs/run_baselines.py --models ...`.

## 2. Run

```bash
bash cloud/run_all.sh            # 1 GPU lane
LANES=2 bash cloud/run_all.sh    # 2 concurrent training lanes
```

For a rented instance where the SSH session may disconnect, install
`cloud/fake-review-supervisor.conf` under `/etc/supervisor/conf.d/`; its wrapper
keeps the process managed and mirrors the full console output to
`results/run_all.log`.

### Modern study (primary)
| step | output |
|---|---|
| fetch `modern-fake-reviews` → CSV | `dataset/modern_reviews_deepseek.csv` |
| human-only label-permutation negative control ×1 | `results/runs.csv` (`ctrl_orperm`) |
| Rs-QLoRA r64 ×3 seeds + **standard QLoRA r64 ×3** (`grids/m_modern.txt`) | `results/runs.csv` |
| TF-IDF sanity baseline | `results/baselines.csv` |
| transformer baselines (RoBERTa / DeBERTa-v3 / ModernBERT-large) | `results/baselines_transformer.csv` |
| **head-to-head** TF-IDF vs Rs-QLoRA by review length + **cross-generator** (detect 1.2B fakes) | stdout |

The head-to-head is the headline result: a contextual model beats bag-of-words
most on **short reviews** and on the **unseen generator** (cross-gen), even though
both hit ~99% in-distribution.

### Original rebuttal phases (still included)
A1 scaling×rank×seed, B ablations, D1 LOCO, same-split baselines, error/imbalance
analyses, efficiency (VRAM/latency). All resumable (skip what's already in the
ledger); a failing phase never aborts the rest.

## Budget
Default modern scope: 23 LoRA runs (one human-only control, 6 modern-study,
12 cross-generator, 4 LOGO) plus 3 transformer baselines. The full rebuttal adds
~40 LoRA + baseline runs. Whole thing is a few hours on one H100; less on H200
or with `LANES=2`.

## Individual pieces
```bash
source .venv/bin/activate
python runs/fetch_hf_dataset.py --repo Flowerly/modern-fake-reviews --out dataset/modern_reviews_deepseek.csv
python runs/run_queue.py grids/m_modern.txt --lanes 1
python runs/head_to_head.py
# Rs-QLoRA cross-gen recall on 1.2B fakes (after a modern run exists):
python runs/eval_external.py --adapter results/m1_rs_r64_s1998_*/adapter \
       --csv dataset/sota_fakes_1p2b_norm.csv --label-col label --pos-value CG --name m1_on_1p2b
```
