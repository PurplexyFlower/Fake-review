# Source-grouped DeepSeek comparisons: summaries

The 48 jobs defined in `grids/source_grouped_rebuild/plan.json`, all on the
source-grouped DeepSeek partition with seeds 1998, 7, and 42:

- paired Rs-QLoRA and standard QLoRA (rank 64, alpha 128), run on the same GPU
  instance for each seed;
- Rs-QLoRA with a single-cycle cosine schedule, a linear schedule, and no label
  smoothing;
- fully fine-tuned RoBERTa-base, DeBERTa-v3-base, and ModernBERT-large;
- TF-IDF logistic regression and LinearSVC, matched and pooled leave-one-generator-out
  (deterministic, so the three seeds give identical predictions).

Files:

- `by_seed.csv`: metrics, resources, and host for every job. `archive` points to the
  job's folder in `results/source_grouped_followup/`.
- `summary.json`: mean ± sample SD per condition, and the seed-wise paired comparison.
- `error_analysis.json`: error analysis of the paired Rs-QLoRA seed-1998 job.

Training time is the elapsed `trainer.train()` call. Jobs ran on several instances,
so wall times are comparable only within an instance-matched pair.
