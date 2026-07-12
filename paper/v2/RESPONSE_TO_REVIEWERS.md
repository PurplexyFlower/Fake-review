# Response to reviewers: revision evidence map

This file maps the two v1 reports to the redesigned manuscript. It is an internal
draft for preparing the journal response; reviewer quotations should be copied
from the submission system before sending a formal response.

## Changes that alter the paper's premise

The revision no longer presents a high matched score on one GPT-2-era dataset as
general machine-authorship detection. It distinguishes human-authored (OR),
computer-generated (CG), and deceptive content; evaluates four generator domains;
and centers cross-generator transfer. “State of the art” and “clearly demonstrates”
claims were removed.

During revision we found an additional limitation not raised explicitly in the
reports. Modern generators share grounding-source reviews. Pairwise target-source
overlap is 69.3–81.5%, and pooled generator-holdout overlap is 94.3–94.5%. The
paper therefore calls LOGO a generator-held-out test under shared source support,
not fully out-of-distribution generalization. This audit is reproducible through
`runs/analyze_source_overlap.py` and `results/source_overlap.json`.

## Reviewer 1

### CG is not equivalent to fake or deceptive

Addressed throughout. The revised Introduction defines OR and CG and reserves
deception terminology for data or prior work in which intent is labeled.

### One balanced dataset and insufficient external validation

Substantially addressed, but not completely resolved. The revision uses four
generator domains, a full cross-generator matrix, pooled generator holdout,
prevalence reweighting, and a human-only negative control. It still covers one
English product-review source population and does not test a second platform or
adversarial edits. Those boundaries are explicit in the Abstract, Discussion,
and Limitations. No unsupported external-domain claim is made.

### Unmatched state-of-the-art comparison

Addressed. The claim was removed. RoBERTa-base, DeBERTa-v3-base, ModernBERT-large,
and TF–IDF use frozen study splits. The paper states that DeBERTa has the best
matched accuracy and that Rs-QLoRA's supported benefit is adaptation efficiency.

### Standard versus rank-stabilized LoRA and rank ablations

Addressed with two experiments. A matched modern H100 comparison isolates standard
versus rank-stabilized scaling at rank 64 across three seeds. A complete historical
2×4 ablation compares ranks 8, 16, 32, and 64 under both scalings, with three seeds
per cell. The latter is clearly labeled historical rather than used as modern
generalization evidence. Scheduler and label-smoothing effects are not claimed as
separate contributions because the available evidence is not a balanced ablation.

### Internal baselines and the faster high-rank result

Addressed. All modern transformer settings are reported. The paper explains the
time difference as earlier validation convergence and early stopping, not greater
per-step throughput. Standard and Rs runs use the same H100 hardware.

### Reproducibility details

Addressed. The revision records the exact base checkpoint, quantization, target
modules, classification setup, trainable/total parameter counts, maximum length,
optimizer, batch size, accumulation, scheduler, early stopping, seeds, split
procedure, library versions, memory procedure, artifact URLs, and data-generation
protocol. Provider aliases and missing immutable revisions are disclosed rather
than reconstructed.

### Multi-seed uncertainty and AUC metrics

Addressed where runs exist. Generator-specific and scaling comparisons use three
seeds and report sample SD. ROC–AUC and PR–AUC are recorded. LOGO and full-finetune
baselines remain single-seed and are labeled as such in the Abstract, Methods,
Results, and Limitations.

### VRAM reporting

Addressed. The paper distinguishes peak PyTorch allocation (about 2.05 GiB) from
process-level `nvidia-smi` use (about 3.1 GiB), names the batch/sequence settings,
and avoids treating training allocation as a general inference benchmark.

### Error analysis

Addressed on the canonical published modern run. The revision reports the confusion
counts, length bins, category maxima, confidence distribution, and qualitative
error patterns. It states that subgroup rankings are descriptive for one run.

### Related work organization and tone

Addressed. Related work is separated into deceptive reviews, machine-generated
text generalization, and efficient adaptation. Foundational opinion-spam work and
directly relevant generator-transfer/robustness work are cited. Suggested papers
outside the narrowed authorship-detection question were not added merely to expand
the bibliography.

## Reviewer 2

### Rs-QLoRA contribution was not isolated

Addressed by the matched scaling comparison and historical rank grid. The revised
claim is faster convergence at comparable ceiling-level accuracy, not automatic
accuracy superiority.

### Balanced data does not represent real prevalence

Addressed analytically and bounded. Using observed modern LOGO recall/FPR, the
paper shows that expected positive predictive value falls to roughly 31–33% at
1% CG prevalence and 83–84% at 10% prevalence at the unchanged threshold. It calls
for calibration and human review rather than claiming a deployment trial.

### Stronger same-split baselines

Addressed with TF–IDF logistic regression, RoBERTa-base, DeBERTa-v3-base, and
ModernBERT-large. Hyperparameters and seed counts are reported.

### Resource claims were underdocumented

Addressed as described for Reviewer 1. The old mixed-device 4.3-hour comparison
was removed from the revised evidence chain.

### Generalization, uncertainty, ROC–AUC, and PR–AUC

Addressed with the four-generator matrix, generator holdout, negative control,
multi-seed matched cells, and probability-based metrics. Remaining single-seed
and source-overlap limitations are stated directly.

### Broader classic-review and fake-news citations

The revision adds the foundational opinion-spam and deceptive-review literature.
It does not merge fake-news detection into the core review because news veracity
and review authorship are different tasks; the paper's terminology section makes
that distinction explicit.

## Work intentionally not represented as complete

- No second review platform or language was evaluated.
- No adversarial paraphrase or human-edit robustness test was run.
- LOGO has one neural training seed.
- Modern transformer baselines have one seed each.
- The current split is not jointly grouped by grounding-source ID.
- Provider aliases cannot be converted retrospectively into immutable model
  revisions.

These are limitations and next experiments, not text to be patched over with
stronger wording.
