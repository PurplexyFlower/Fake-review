# Analysis

Recomputes the reported statistics and figures from the per-run outputs in this
repository. Nothing is read from a pre-aggregated summary except the row-level
transfer matrix (`results/cross_gen_results.json`) and the row-level source-overlap
audit (`results/source_overlap.json`).

```bash
python analysis/recompute_results.py     # writes analysis/numbers.json
python analysis/make_figures.py          # writes analysis/figures/
```

`numbers.json` covers:

- matched performance (row-level and source-grouped);
- directional transfer and leave-one-generator-out;
- the instance-paired Rs-QLoRA / standard QLoRA comparison;
- both rank analyses;
- the baselines;
- error analysis;
- prevalence-adjusted precision;
- the external corpora.

Requires Python 3.10+, numpy, scikit-learn, and matplotlib. No GPU is needed.
