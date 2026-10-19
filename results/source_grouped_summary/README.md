# Source-grouped experiment: summaries

Aggregates of the 27 source-grouped runs in `results/sg*_rs_r*/` (directional,
leave-one-generator-out, and fixed-multiplier rank), produced by
`runs/aggregate_source_grouped_results.py` and checked by
`runs/audit_source_grouped_results.py`.

- `source_grouped_summary.json`: mean ± sample SD over seeds 1998, 7, and 42.
- `source_grouped_{directional,logo,fixed_scale_rank}_by_seed.csv`: per-seed metrics.
- `source_grouped_prevalence_*.csv`: expected alert precision, recall, and alerts per
  10,000 reviews at assumed CG prevalences of 0.1%, 0.5%, 1%, 5%, and 10%, at
  threshold 0.5 and along the full threshold grid.
- `rank_resource_and_parameter_audit.json`: trainable parameters and peak allocated
  memory for the fixed-multiplier rank series.
- `independent_audit.json`: independent recomputation of every threshold and ranking
  metric from the stored predictions, and the zero source-overlap checks.

The partition is defined in `dataset/source_grouped/manifest.json`.
