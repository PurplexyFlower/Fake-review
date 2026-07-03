# Detecting Modern AI-Generated Fake Reviews — Project Report

## TL;DR

We started from a paper that detects the Kaggle "Fake Reviews" dataset's
*computer-generated* half with a small LLM (Qwen2.5-0.5B + Rs-QLoRA) and got
~98.7% accuracy. Probing it revealed the core problem: that detector flags only
**4%** of **modern** LLM-generated reviews — the Kaggle "fake" data is GPT-2-era
and **obsolete**. So we pivoted: keep the real human reviews, generate fresh fakes
with several modern LLMs (LFM-1.2B, DeepSeek-v4-pro, GLM-5.2), and study detection
of *today's* AI fakes.

The headline finding is a **cross-generator matrix**: detecting fakes
*in-distribution* is trivial for everyone (even bag-of-words ≈ 99%), but
**transfer across generators is the real story** — old↔modern detection collapses
to ~1%, while modern frontier models (DeepSeek, GLM) **converge to a shared
fingerprint** and transfer to each other at 95–96%. The defensible contribution is
therefore not "we detect fakes" but: **you need current-frontier training data;
obsolete or weak-model data does not transfer.**

---

## 1. Background: the original paper and the reviewer's doubts

- **Task as published:** binary classification of the Kaggle *Fake Reviews
  Dataset* — 40,432 reviews, balanced: 20,216 real human **OR**(iginal) and
  20,216 **CG** (computer-generated, produced years ago by a GPT-2-class model).
- **Method:** Qwen2.5-0.5B (4-bit) with **Rs-QLoRA** (rank-stabilized LoRA,
  scaling α/√r), cosine-restart schedule, label smoothing.
- **Reviewer 1** correctly pushed back: "computer-generated ≠ deceptive fake";
  single balanced dataset; SOTA claims not under matched conditions; missing
  ablations; reproducibility gaps; no uncertainty estimates; thin error analysis.

We built a reproducible harness (`runs/`) to answer those points with evidence —
multi-seed runs, ablations, error analysis, leakage-cleaned metrics, etc. (see
`RESPONSE_TO_REVIEWERS.md`). That work stands, but it also exposed a deeper issue.

## 2. The pivot: the Kaggle "fake" data is obsolete

We generated modern LLM reviews and ran the published detector on them
(`runs/eval_external.py`, the "D3" probe):

> **A detector trained on the Kaggle GPT-2-era CG reviews flags only 4.0% of
> modern LLM-generated reviews** (mean P(fake)=0.05). It passes 96% of modern
> machine text as "genuine."

We validated this isn't a loading bug (the same eval path reproduces the test
accuracy 98.834% to the digit). So the model learned **generator-specific GPT-2
artifacts**, not "machine-text in general." The original dataset is no longer a
useful proxy for real-world AI fakes.

**New goal:** keep the authentic human half, replace the machine half with **modern
LLM fakes**, and study (a) how detectable they are and (b) whether detectors
generalize across LLM generators.

## 3. What we are building: a cross-generator benchmark

A family of datasets that share the **same real-human half** and differ only in the
**generator** of the fake half, spanning the capability range:

| generator | role | source |
|---|---|---|
| GPT-2-era (Kaggle CG) | obsolete | original dataset |
| **LFM2.5-1.2B** | weak modern | local (LM Studio) |
| **DeepSeek-v4-pro** | frontier | API |
| **GLM-5.2** | frontier | API |
| (Opus / Claude-Code) | frontier (samples) | API / harness — dropped as redundant |

Each becomes a row/column in a **cross-generator matrix**: train a detector on
generator *i*, test it on generator *j*'s held-out fakes.

## 4. Datasets and how they were made

For every generator we build a **balanced (1:1), normalized** dataset of real OR vs
that generator's fakes (`runs/build_modern_dataset.py`), then study it.

**Grounded generation** (`runs/gen_fakes_sota.py`, `gen_fakes_opus.py`): each fake
is conditioned on a **paired real review** — same category, same star rating,
matched length — and asked to write a *different* review for the same product (with
a per-draft "angle" so they don't cluster, and a Jaccard<0.6 anti-copy filter).
This makes category/rating/length **marginals match** the human half, so a detector
can't win on trivial cues. Generation is provider-agnostic (any OpenAI-compatible
endpoint: DeepSeek, GLM, GPT; Anthropic via its SDK), resumable, and
category-balanced.

**Normalization:** the LLMs emit smart quotes / em-dashes (Unicode); humans don't.
Left alone that's a trivial "em-dash ⇒ fake" tell, so both halves are normalized to
one ASCII convention. (We *found* this the hard way — see §6.)

**Public datasets (Hugging Face):**
- `Flowerly/modern-fake-reviews` — real OR vs **DeepSeek-v4-pro**
- `Flowerly/modern-fake-reviews-lfm` — real OR vs **LFM2.5-1.2B**

## 5. Methodology and rigor (why the numbers are trustworthy)

- **Negative control:** label the *real* reviews randomly into two fake classes and
  run the whole pipeline → **49.2%** (chance). The pipeline isn't leaking; high
  accuracy elsewhere is real signal, not an artifact.
- **Tell audits** (`runs/diagnose_tells.py`): we inspect the top discriminative
  features and induced-artifact rates, and fixed/documented the ones we caused.
- **Frozen protocol:** fixed split (seed 1998), multi-seed headline, identical
  hyperparameters across cells; metrics reported full and leakage-cleaned.

## 6. Results

### 6.1 In-distribution detection is *easy* — for everyone
| dataset | TF-IDF (bag-of-words) | Rs-QLoRA |
|---|---|---|
| original GPT-2 CG | 94.3 / 94.9% | 98.7% (5-seed 98.72±0.08) |
| modern DeepSeek | **98.9%** | **99.8%** |
| modern LFM-1.2B | 99.5% | — |

A dumb bag-of-words model already hits ~99% on modern fakes. **In-distribution
accuracy alone cannot justify a neural detector** — a reviewer would (rightly) ask
"why the GPU?"

### 6.2 Where the model *does* help: short reviews
Rs-QLoRA's edge over TF-IDF is concentrated where surface n-grams run out of signal:

| review length | TF-IDF | Rs-QLoRA | gap |
|---|---|---|---|
| 8–15 words | 96.4% | 99.3% | **+2.9** |
| 40–70 words | 99.7% | 100% | +0.3 |

Overall that's an ~80% **relative** error reduction (1.17%→0.23%), concentrated on
short text — modest in absolute terms but real and defensible.

### 6.3 You cannot prompt your way to "undetectable" fakes
Eleven prompt strategies (`runs/gen_human_like_test.py`) to make DeepSeek fakes
blend into real reviews: **novel** generation plateaus at **~90%** separable
(bag-of-words). The *only* way to reach near-chance (59%) was near-**copying** real
reviews (77% word overlap = plagiarism, not generation). Conclusion: **novel modern
LLM reviews carry an irreducible stylometric fingerprint.**

### 6.4 The headline: cross-generator transfer
Train a detector on one generator, test on each generator's held-out fakes
(`runs/cross_gen_matrix.py`). **TF-IDF detection recall:**

```
train\test   GPT2    LFM    DeepSeek   GLM
GPT2         94.1%   1.1%   1.4%       0.6%
LFM           0.7%  99.3%   38.0%     32.5%
DeepSeek      0.7%  82.1%   98.7%     95.1%
GLM           0.6%  88.1%   96.1%     98.4%
```
Per-detector FPR on its own human test set: GPT2 4.9%, LFM 0.5%, DeepSeek 1.0%,
GLM 1.1% — low recalls are genuine misses, not a flag-everything artifact.
Three regimes:
1. **Old ↔ modern = a hard wall (~1% both ways).** The Kaggle dataset is useless
   for modern fakes *and* vice-versa — the obsolescence result, shown symmetrically.
2. **Frontier ↔ frontier = high transfer (95–96%).** DeepSeek and GLM catch each
   other's fakes almost perfectly — **modern frontier LLMs converge to a similar
   fingerprint.** (New, and good news for real-world detection.)
3. **Weak → frontier fails (33–38%); frontier → weak holds (82–88%).** Train on a
   crude generator and you miss frontier fakes; train on a frontier model and you
   catch weak ones. **Train on the strongest generator you have.**

(A small Rs-QLoRA cross-gen check, DeepSeek→LFM, gives 85.2% vs TF-IDF 82.7% — a
modest model edge; the full neural matrix is the next experiment.)

### 6.5 The refined thesis
Not the vague "detectors don't transfer," but the precise and defensible:
> **Effective modern fake-review detection requires *current-frontier* training
> data. Obsolete (GPT-2) or weak-model data does not transfer; but because frontier
> generators converge, a detector trained on one current frontier model generalizes
> across them.** In-distribution detection is trivial; the science is robustness.

## 7. What's in the repo

```
runs/   generators (gen_fakes_sota/opus, claude_gen_io), training harness
        (train_run, run_queue), baselines (tfidf, transformer, run_baselines),
        analyses (reviewer_analysis, diagnose_tells, summarize_ledger,
        analyze_overfit), cross_gen_matrix, head_to_head, eval_external,
        build_modern_dataset, fetch_hf_dataset, push_to_hf, measure_*
grids/  experiment definitions (a1 scaling×rank, b ablations, d1 LOCO,
        m_modern, xgen cross-generator)
cloud/  uv setup.sh + run_all.sh for an H100/H200 (runs the full pipeline)
.claude/skills/gen-fake-reviews  generate fakes through the Claude Code harness
dataset/  real set + all generated/modern CSVs (also on HF)
results/  metrics only (runs.csv ledger, baselines, summaries, test_probs, JSON)
REVISION_PLAN.md · RESPONSE_TO_REVIEWERS.md  reviewer rebuttal + evidence map
```
(Model weights ~4.3 GB are **not** in git — they live on HF / are regeneratable.)

## 8. Next steps

1. **Neural 4×4 cross-generator matrix on an H100/H200** (`cloud/run_all.sh` →
   `grids/xgen.txt` + `cross_gen_matrix.py`): does Rs-QLoRA **close the weak→frontier
   gap** or **breach the old↔modern wall** where bag-of-words can't? This is the
   experiment that decides how much the neural model is worth.
2. **Same-split transformer baselines on the modern data** (RoBERTa / DeBERTa-v3 /
   ModernBERT-large) for a fair "is a model needed" comparison.
3. **More frontier generators** (Gemini 3.5 Flash, GPT-5.5) to widen the matrix and
   pre-empt "you only tested one/two models."
4. **Multi-seed** the modern headline for error bars.
5. **Rewrite the paper** around the cross-generator / obsolescence thesis (§6.5),
   softening in-distribution SOTA claims, and fold in the reviewer responses.

## 9. Reproduce

```bash
# 1. env (cloud)            bash cloud/setup.sh
# 2. everything             bash cloud/run_all.sh          # H100/H200
# 3. cross-gen preview      python runs/cross_gen_matrix.py --no-neural   # CPU
# 4. a generator            python runs/gen_fakes_sota.py --model glm-5.2 \
#      --base-url https://api.z.ai/api/paas/v4/ --api-key-env GLM_API_KEY --no-think \
#      --out dataset/sota_fakes_glm.csv
```
Datasets pull from HF automatically; API keys live in `.env` (never committed).
