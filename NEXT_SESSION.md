# Next-session handoff

Use this file to resume the project in a new Codex session without relying on
the previous chat history. Start the next session with:

> Read `NEXT_SESSION.md`, inspect the current `main` branch, and continue from
> the stated open work. Do not create a branch or pull request. Preserve v1.

## Collaboration rules

- Work directly on `main`; do not create branches or pull requests.
- Do not commit or push unless explicitly requested in that session.
- The user's requests are directional rather than literal specifications. Prefer
  the simplest scientifically defensible design over patches or special cases.
- If evidence breaks a desired claim, change the claim and explain why. Never
  conceal a limitation to make the paper appear complete.
- Preserve `paper/v1/` exactly. Revised work belongs in `paper/v2/`.
- Never place access tokens, API keys, SSH private keys, or other credentials in
  tracked files. A Hugging Face token was shared in an earlier chat and should
  be treated as sensitive and rotated independently.

## Repository and publication state

- Repository: `https://github.com/PurplexyFlower/Fake-review`
- Primary branch: `main`
- Adapter repository: `KandirResearch/fake-review-rs-qlora-adapters`
- Modern datasets:
  - `Flowerly/modern-fake-reviews-lfm`
  - `Flowerly/modern-fake-reviews`
  - `Flowerly/modern-fake-reviews-glm`
- Dataset cards and adapter artifacts were uploaded and verified before the
  manuscript rewrite.

## Paper files

- Original manuscript: `paper/v1/1862126_Manuscript.TEX`
- Original reviews: `paper/v1/peer_review1.txt` and `peer_review2.txt`
- Revised source: `paper/v2/1862126_Manuscript_Revised.tex`
- Readable PDF: `paper/v2/1862126_Manuscript_Revised.pdf`
- Bibliography: `paper/v2/references.bib`
- Reviewer evidence map: `paper/v2/RESPONSE_TO_REVIEWERS.md`
- Revision rationale: `paper/v2/REVISION_NOTES.md`
- `paper/v2/FrontiersinHarvard.cls` is a clearly marked **preview-only** local
  compatibility class. It is not the official publisher class and must not be
  submitted as if it were.

The local preview is 11 pages. The compatibility class accepts Frontiers' short
title and short author optional arguments; this was fixed after the first PDF
showed duplicated header text and stray brackets. Compile the final submission
against the official current Frontiers template when it is available.

## Revised scientific center

The original paper treated high matched accuracy on a GPT-2-era dataset as broad
AI-review detection. The revised paper instead asks what transfers across
generators. Its central evidence is:

1. matched performance across four generator domains;
2. a four-by-four generator transfer matrix;
3. pooled leave-one-generator-out (LOGO) evaluation;
4. a human-only random-label negative control;
5. matched TF-IDF and transformer baselines;
6. standard versus rank-stabilized QLoRA efficiency;
7. rank-by-scaling ablation on the historical domain;
8. modern-run error analysis.

Use OR for human-authored reviews and CG for computer-generated reviews.
Machine authorship is not equivalent to deception, so avoid calling all CG text
“fake” or “deceptive.” The state-of-the-art claim was removed.

## Final results currently reported

### Matched Rs-QLoRA accuracy, three seeds

| Generator | Mean ± sample SD |
|---|---:|
| GPT-2 | 98.497% ± 0.154% |
| LFM | 99.776% ± 0.097% |
| DeepSeek | 99.694% ± 0.057% |
| GLM | 99.252% ± 0.071% |

### Cross-generator CG recall, three-seed means

| Train / test | GPT-2 | LFM | DeepSeek | GLM |
|---|---:|---:|---:|---:|
| GPT-2 | 98.1% | 0.3% | 0.5% | 0.1% |
| LFM | 0.1% | 99.8% | 53.4% | 44.5% |
| DeepSeek | 0.3% | 85.5% | 99.8% | 97.2% |
| GLM | 0.2% | 91.7% | 98.2% | 99.3% |

- Mean diagonal recall: 99.3%.
- Mean all-generator off-diagonal recall: 39.3%.
- Mean modern-only off-diagonal recall: 78.4%.
- Human FPR by training generator: GPT-2 1.18%, LFM 0.26%, DeepSeek
  0.36%, GLM 0.87%.
- GPT-2 and the modern generators are nearly disconnected domains.
- DeepSeek and GLM detectors transfer broadly across the tested modern data;
  the LFM-trained decision boundary is narrower.

### LOGO, single neural seed

| Held-out generator | Rs recall | Rs FPR | TF-IDF recall | TF-IDF FPR |
|---|---:|---:|---:|---:|
| GPT-2 | 0.1% | 0.48% | 0.8% | 1.65% |
| LFM | 91.1% | 1.92% | 84.2% | 6.40% |
| DeepSeek | 95.1% | 2.13% | 92.3% | 5.60% |
| GLM | 96.7% | 2.01% | 90.4% | 6.02% |

### Critical source-overlap correction

The modern generators share the same human grounding-source pool. Raw generation
files retained `src_idx`, allowing this to be audited after the main runs:

- pairwise source overlap between another generator's training split and a
  target generator's test split: 69.3–81.5%;
- pooled-other-generators overlap in LOGO: 94.3–94.5%.

Therefore LOGO holds out generator identity and excludes exact held-out OR text,
but it does **not** jointly hold out prompt-source identity. The correct claim is
“generator-held-out transfer under shared grounded-source support,” not fully
out-of-distribution generalization. The directional asymmetries remain useful
because opposite directions have nearly equal overlap. Reproduce this audit with
`runs/analyze_source_overlap.py`; results are in `results/source_overlap.json`.

### Negative control

- 20,214 human reviews with balanced random pseudo-labels.
- Accuracy 49.31%, ROC-AUC 48.95%, PR-AUC 49.14%.
- This argues against a general label-channel artifact, but does not remove the
  prompt-source dependence above.

### Matched DeepSeek baselines

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| RoBERTa-base | 99.435% | 99.982% |
| DeBERTa-v3-base | 99.806% | 99.996% |
| ModernBERT-large | 99.700% | 99.996% |
| Rs-QLoRA, three-seed mean | 99.617% | 99.960% |

DeBERTa has the highest matched accuracy. Rs-QLoRA's supported advantage is
efficient adapter adaptation, not peak matched-distribution accuracy.

### Rs-QLoRA versus standard QLoRA, DeepSeek/H100/three seeds

- Rs accuracy: 99.617% ± 0.071%.
- Standard accuracy: 99.523% ± 0.088%.
- Paired mean difference: +0.094 percentage points; do not claim accuracy
  superiority from three seeds at a ceiling.
- Rs runtime: 17.3 minutes; best step 800.
- Standard runtime: 32.0 minutes; best step 2,533.
- Runtime reduction: 45.9%, or 1.85× faster.
- Peak PyTorch allocation: about 2.05 GiB; process-level `nvidia-smi` use about
  3.1 GiB including CUDA/process overhead.

### Canonical modern error analysis

- DeepSeek seed-1998 published adapter: 24 errors / 5,660, accuracy 99.576%.
- 14 OR→CG false positives and 10 CG→OR false negatives.
- Error rate falls with length: 1.55% at 8–14 words, 0.64% at 15–24,
  0.33% at 25–49, 0.16% at 50–99, and 0.08% at 100+.
- Mean confidence: errors 0.874, correct predictions 0.995.
- 15 of 24 errors still have confidence at least 0.90.

## Data provenance boundaries

Read `DATA_PROVENANCE.md` and `dataset/provenance.json` before modifying data
claims. Modern reviews were grounded on category, rating, the first 400 source
characters, and target length. The common generation protocol used temperatures
0.85–1.09, `top_p=0.95`, up to five attempts, Jaccard similarity below 0.6,
normalization, and exact deduplication.

The raw and final artifacts are reproducible by hash. Byte-identical regeneration
is not possible because API decoding was unseeded, provider aliases are mutable,
and the exact LFM serving build was not retained. Do not invent provider revision
IDs retrospectively.

## Main evidence files

- `PAPER_FINDINGS.md`
- `DATA_PROVENANCE.md`
- `results/cross_gen_results.json`
- `results/source_overlap.json`
- `results/runs.csv`
- `results/baselines_transformer.csv`
- `results/adapter_manifest.json`
- `results/*/metrics.json`
- `results/*/test_probs.csv`

## Remaining work

1. Read the PDF visually from beginning to end and edit for narrative flow,
   concision, and journal style. The scientific content is drafted; it still
   needs an author-level editorial pass.
2. Compile against the official current Frontiers class, not the preview class,
   and resolve publisher-template formatting.
3. Confirm the journal's exact generative-AI disclosure requirements. The draft
   includes an honest disclosure and must not revert to v1's “no AI used” claim.
4. Have all authors verify contributions, affiliation, correspondence details,
   model/provider names, URLs, and final claims.
5. If additional compute becomes justified, the highest-value experiment is a
   source-grouped generator holdout. Secondary priorities are multi-seed LOGO
   and adversarial/human-edit robustness. Do not imply these are already done.

## Suggested next-session first action

Open `paper/v2/1862126_Manuscript_Revised.pdf` beside
`paper/v2/1862126_Manuscript_Revised.tex`, perform a full visual/editorial review,
and list proposed changes before modifying the scientific claims.
