# Modern fake-review study: paper findings

Status: core experiments, transformer baselines, artifact recovery, and the
canonical seed-1998 external evaluation are complete. Structured evidence was
recovered from the H100 instance on 2026-07-12.

## Experimental scope

- Binary task: genuine human reviews (OR) versus generated reviews (CG).
- Generators: GPT-2-era reviews, LFM2.5-1.2B, DeepSeek-v4-pro, and GLM-5.2.
- Adapter model: Qwen2.5-0.5B, 4-bit base, rank-64 LoRA adapters.
- Frozen split seed: 1998.
- Single-generator transfer results average three training seeds: 1998, 7, 42.
- LOGO results use one pooled-training seed (1998) per held-out generator.

## Finding 1: matched-distribution detection is near ceiling

| Generator-specific Rs-QLoRA detector | Test accuracy, mean ± sample SD |
|---|---:|
| GPT-2 | 98.497% ± 0.154% |
| LFM | 99.776% ± 0.097% |
| DeepSeek | 99.694% ± 0.057% |
| GLM | 99.252% ± 0.071% |

These results establish that each generator can be separated from the matched
human distribution. They do not, by themselves, establish generator-independent
AI detection.

## Finding 2: transfer is structured by generator family and age

Rs-QLoRA detection recall, with rows denoting the training generator and columns
the test generator:

| Train \\ test | GPT-2 | LFM | DeepSeek | GLM |
|---|---:|---:|---:|---:|
| GPT-2 | 98.1% | 0.3% | 0.5% | 0.1% |
| LFM | 0.1% | 99.8% | 53.4% | 44.5% |
| DeepSeek | 0.3% | 85.5% | 99.8% | 97.2% |
| GLM | 0.2% | 91.7% | 98.2% | 99.3% |

- Mean diagonal recall: 99.3%.
- Mean off-diagonal recall across all four generators: 39.3%.
- Mean off-diagonal recall restricted to the three modern generators: 78.4%.
- The asymmetry is substantive: DeepSeek/GLM detectors transfer broadly across
  modern generators, while the LFM detector transfers only partially.
- GPT-2 and the modern generators are almost disjoint detection domains in both
  directions.

The supported claim is therefore not a universal AI-text detector. The evidence
supports a transferable modern-generator signal plus a separate GPT-2-era
fingerprint.

## Finding 3: pooled modern training generalizes to unseen modern generators

| Held-out generator | Rs-QLoRA recall | Human false-positive rate | TF-IDF recall | TF-IDF FPR |
|---|---:|---:|---:|---:|
| GPT-2 | 0.1% | 0.48% | 0.8% | 1.65% |
| LFM | 91.1% | 1.92% | 84.2% | 6.40% |
| DeepSeek | 95.1% | 2.13% | 92.3% | 5.60% |
| GLM | 96.7% | 2.01% | 90.4% | 6.02% |

For an unseen modern generator, pooled Rs-QLoRA training delivers 91–97% recall
at approximately 2% human FPR and consistently improves on TF-IDF. Neither model
generalizes from the three modern generators to GPT-2. LOGO uses one seed, so
uncertainty across training seeds remains to be measured.

## Finding 4: the negative control behaves exactly as required

The human-only dataset contains 20,214 genuine reviews with balanced random
pseudo-labels. Rs-QLoRA obtains:

- Test accuracy: 49.31%.
- AUROC: 48.95%.
- PR-AUC: 49.14%.

Chance performance argues against label leakage or a pipeline artifact capable
of producing the headline results without a genuine class signal.

## Finding 5: Rs-QLoRA's clearest advantage is efficiency, not matched-test accuracy

On the DeepSeek dataset:

- Rs-QLoRA mean accuracy: 99.617% ± 0.071%.
- Standard QLoRA mean accuracy: 99.523% ± 0.088%.
- Mean paired difference: +0.094 percentage points; only three seeds, so this is
  not sufficient evidence of an accuracy advantage.
- Across all three matched H100 seeds, Rs-QLoRA averaged 17.3 minutes and a best
  checkpoint at step 800, versus 32.0 minutes and step 2,533 for standard QLoRA.
  This is a 45.9% runtime reduction (1.85× speedup).
- Adapter training peaked near 2.05 GiB PyTorch allocation (about 3.1 GiB visible
  to `nvidia-smi` including CUDA/process overhead).

The defensible interpretation is substantially faster convergence at essentially
the same ceiling-level matched-test quality.

## Finding 6: full-finetuning baselines remain competitive

| Model | Test accuracy | AUROC |
|---|---:|---:|
| RoBERTa-base | 99.435% | 99.982% |
| DeBERTa-v3-base | 99.806% | 99.996% |
| ModernBERT-large | 99.700% | 99.996% |
| Rs-QLoRA, three-seed mean | 99.617% | 99.960% |

DeBERTa is the best matched-distribution baseline, exceeding mean Rs-QLoRA by
0.189 percentage points. Rs-QLoRA beats RoBERTa by 0.183 points and trails
ModernBERT by 0.082 points while storing only an adapter over a quantized 0.5B
base.

## Supporting head-to-head result

On the primary test set, Rs-QLoRA reaches 99.58% versus 98.83% for TF-IDF. The
largest gains occur on short reviews. On a separate 1.2B-generator fake set,
recall is 84.26% for Rs-QLoRA versus 82.73% for TF-IDF.

## Claim boundaries and required paper language

- Do not describe the system as a universal AI detector.
- Do describe strong transfer among the tested modern generators, with explicit
  failure on the GPT-2-era domain.
- Generator attribution is a follow-up multiclass/open-set task; the current
  binary detector matrix demonstrates fingerprints but does not identify the
  source model directly.
- Generator identity is confounded with each dataset's prompting, sampling, and
  provenance. Attribution claims require matched prompts and decoding settings.
- LOGO neural estimates currently use one seed.
- The original DeepSeek seed-1998 metric record has no saved adapter and is
  retained as historical evidence. A canonical same-configuration H100 rerun
  (`m1_rs_r64_s1998_07121622`) supplies the published weight; the Hub manifest
  distinguishes the superseded metric-only record without rewriting it.

## Local evidence

- `results/cross_gen_results.json`: structured TF-IDF, Rs-QLoRA, and LOGO matrices.
- `results/runs.csv`: adapter experiment ledger.
- `results/baselines_transformer.csv`: full-finetuning baselines.
- `results/run_all.log` and `results/transformer_baselines.log`: execution logs.
- `results/head_to_head.log`: canonical neural-versus-TF-IDF slice comparison.
- `results/*/{config.json,metrics.json,test_probs.csv,val_probs.csv}`: per-run evidence.
- `results/adapter_manifest.json`: dataset and adapter-file SHA-256 manifest.
- `dataset/control_or_permutation_s1998.meta.json`: negative-control provenance.
- `DATA_PROVENANCE.md` and `dataset/provenance.json`: exact modern-data
  generation protocol, raw/final hashes, provider records, and known unknowns.
