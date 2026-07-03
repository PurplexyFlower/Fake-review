# Modern AI-Generated Fake Review Detection — Cross-Generator Study

Detectors trained on yesterday's fake-review data do not catch today's fakes.
This repo builds a **cross-generator benchmark** — the same 20k real human Amazon
reviews paired against fakes from four generators spanning a decade of capability
(GPT-2-era → LFM2.5-1.2B → DeepSeek-v4-pro → GLM-5.2) — and measures how
detection **transfers** between them.

**Headline (TF-IDF detection recall; train on row, test on column):**

| train \ test | GPT-2 | LFM | DeepSeek | GLM |
|---|---|---|---|---|
| **GPT-2**    | 94.1% | 1.1% | 1.4% | 0.6% |
| **LFM**      | 0.7% | 99.3% | 38.0% | 32.5% |
| **DeepSeek** | 0.7% | 82.1% | 98.7% | 95.1% |
| **GLM**      | 0.6% | 88.1% | 96.1% | 98.4% |

(Per-detector FPR on human reviews: 0.5–4.9%, so low recalls are genuine misses.)

- **In-distribution detection is trivial** (~98% diagonal — even bag-of-words).
- **Old ↔ modern is a hard wall (~1%)**: the widely-used Kaggle "Fake Reviews"
  dataset (GPT-2-era) is obsolete — a detector trained on it flags only ~4% of
  modern LLM reviews, and vice-versa.
- **Frontier models converge (95–96%)**: DeepSeek↔GLM transfer almost perfectly.
- **Weak→frontier fails (33–38%)**: train on the strongest generator you have.

Full narrative, methodology, and caveats: **[REPORT.md](REPORT.md)**.
Reviewer-rebuttal evidence for the original Rs-QLoRA paper:
[RESPONSE_TO_REVIEWERS.md](RESPONSE_TO_REVIEWERS.md).

## Datasets

Same real-human half everywhere; only the fake half changes. Balanced 1:1,
ASCII-normalized, fakes grounded on paired real reviews (matched category,
star rating, length — no trivial cues). All CSVs ship in `dataset/`; the two
modern sets are also on the Hub:

| dataset | fake generator | where |
|---|---|---|
| `modern_reviews_gpt2.csv` | GPT-2-era (Kaggle CG) | repo (built from Kaggle CSV) |
| `modern_reviews_lfm.csv` | LFM2.5-1.2B | repo + [`Flowerly/modern-fake-reviews-lfm`](https://huggingface.co/datasets/Flowerly/modern-fake-reviews-lfm) |
| `modern_reviews_deepseek.csv` | DeepSeek-v4-pro | repo + [`Flowerly/modern-fake-reviews`](https://huggingface.co/datasets/Flowerly/modern-fake-reviews) |
| `modern_reviews_glm.csv` | GLM-5.2 | repo + [`Flowerly/modern-fake-reviews-glm`](https://huggingface.co/datasets/Flowerly/modern-fake-reviews-glm) |

## Quickstart

### 1. Reproduce the headline matrix — CPU, ~5 minutes, no GPU

```bash
git clone https://github.com/PurplexyFlower/Fake-review.git && cd Fake-review
uv venv && source .venv/bin/activate          # or: python -m venv .venv
uv pip install datasets scikit-learn scipy
python runs/cross_gen_matrix.py --no-neural
```

### 2. The full experiment — one script (Linux GPU box, e.g. H100/H200)

```bash
bash cloud/setup.sh          # uv env: torch 2.10 cu128, unsloth, transformers…
                             # read its CUDA/import verify block before continuing
bash cloud/run_all.sh        # priority pipeline (see below)
```

`run_all.sh` is resumable (re-run after any interruption; completed runs are
skipped via `results/runs.csv` — delete that file first if you want clean
single-hardware timings) and ordered by priority:

| phase (default `SCOPE=modern`) | what |
|---|---|
| datasets | fetch modern sets from HF, build GPT-2 set from Kaggle CSV |
| `m_modern` grid | Rs-QLoRA r64 ×3 seeds + standard QLoRA ×3 on the DeepSeek set |
| `xgen` grid + matrix | one detector per generator → **4×4 TF-IDF + Rs-QLoRA transfer matrix (with FPR)** |
| supporting | TF-IDF sanity baseline, by-length head-to-head, 3 transformer baselines |

`SCOPE=all bash cloud/run_all.sh` appends the legacy reviewer-rebuttal suite
(scaling×rank grids, ablations, LOCO, a 36-run transformer sweep, error/efficiency
analyses) — many extra GPU-hours; only needed to regenerate the original paper's
rebuttal numbers. `LANES=2` runs two training lanes in parallel.

## Regenerating fakes (optional — needs API keys in `.env`)

`.env` (never committed): `DEEPSEEK_API_KEY`, `GLM_API_KEY`, `CLAUDE_API_KEY`,
`HF_TOKEN` — only needed to generate new fakes or push datasets; reproducing
results needs none of them.

```bash
# any OpenAI-compatible provider (DeepSeek shown; GLM: api.z.ai + GLM_API_KEY)
python runs/gen_fakes_sota.py --model deepseek-v4-pro \
  --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY \
  --no-think --out dataset/sota_fakes_deepseek.csv
# then build a balanced dataset from it
python runs/build_modern_dataset.py --fakes dataset/sota_fakes_deepseek.csv \
  --out dataset/modern_reviews_deepseek.csv
```

There is also a Claude Code skill (`.claude/skills/gen-fake-reviews`) that
generates fakes through the coding harness itself, no API key.

## Repository layout

```
runs/      generators, training harness (train_run/run_queue), baselines,
           cross_gen_matrix, head_to_head, analyses, HF push/fetch, benchmarks
grids/     experiment definitions (m_modern, xgen, a1, b_ablations, d1_loco)
cloud/     setup.sh (uv env) + run_all.sh (the one script) + README
dataset/   all CSVs (real + generated); no raw scraping, research use
results/   metrics only — runs.csv ledger, baselines, per-run test_probs/JSON
           (model weights are NOT in git; retrain via the grids, minutes on H100)
REPORT.md  full write-up: motivation, method, results, next steps
```

## Known caveats (honest list)

- The three modern generators share one grounded prompt template; frontier
  convergence (DeepSeek↔GLM 95%+) could partly reflect the shared template —
  though LFM (same template, 33–38% transfer) argues capability, not prompt,
  drives it. Template-varied generation is future work.
- Fakes' "shipping/packaging" prompt angle mildly over-represents delivery
  vocabulary (documented on the HF dataset cards).
- Off-diagonal matrix cells are slightly optimistic: generator datasets share
  grounding products across their train/test splits.
- `setup.sh`'s pinned stack (torch 2.10 cu128 + unsloth 2026.6.3) mirrors a
  working Windows env; check its verify block on first Linux boot.
