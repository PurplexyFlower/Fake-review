# Source-grouped experiment runbook

The source-grouped experiment trains **27 Rs-QLoRA adapters** on three single-GPU
instances: nine directional runs, nine leave-one-generator-out (LOGO) runs, and
nine fixed-multiplier rank runs. Each condition uses training seeds 1998, 7, and 42.

## Local checks

```powershell
python runs\build_source_grouped_datasets.py
python runs\validate_source_grouped_plan.py
python -m unittest discover -s tests -v
```

Expected invariants:

- 20,200 source IDs and 20,199 exact-text source groups;
- the same 80/6/14 source assignment for LFM, DeepSeek, and GLM;
- 16,160 / 1,211 / 2,829 source IDs in train / validation / test;
- zero source-group overlap across partitions;
- nine runs in each of the three shards;
- fixed-multiplier rank cells `(r, alpha) = (16, 64), (36, 96), (64, 128)`, so
  `alpha / sqrt(r) = 16` in every cell.

## Run one shard per instance

The runs used H100 80GB instances with an amd64 container, CUDA 12.8 or newer,
and at least 50 GB of local disk. Install the same checkout on every instance:

```bash
bash cloud/setup_source_grouped.sh
INSTANCE_ID=1 bash cloud/start_source_grouped_instance.sh
```

Use `INSTANCE_ID=2` and `INSTANCE_ID=3` on the other machines. Each instance runs
one training lane (no DDP). The script:

1. rebuilds and hashes the source-grouped datasets;
2. validates the common split and its grid shard;
3. records hardware and software identity;
4. trains the nine assigned adapters, skipping verified completed cells;
5. evaluates the directional or held-out-generator test sets, enforcing zero
   train/test source-group overlap;
6. stores per-example probabilities;
7. writes a hash-manifested result package
   (`results/source_grouped_instance_N_package.zip`).

The setup creates an isolated `.venv`. Core dependencies are pinned in
`cloud/source_grouped_constraints.txt`, and the Qwen base model is pinned to
revision `10413c288cb9629acdf60b3e0229f3ba75efe413`. When the instance image
provides Supervisor, `cloud/source_grouped_instance_N.conf` runs the same steps
as Supervisor programs instead of the detached launcher.

The queue is resumable: a run is skipped only when its configuration, dataset
hash, metrics, probabilities, and adapter are all present. Incomplete runs
restart from the beginning.

## Aggregate locally

```powershell
python runs\ingest_source_grouped_packages.py `
  path\to\source_grouped_instance_1_package.zip `
  path\to\source_grouped_instance_2_package.zip `
  path\to\source_grouped_instance_3_package.zip

python runs\validate_source_grouped_plan.py --completed-grid grids\source_grouped_directional.txt
python runs\validate_source_grouped_plan.py --completed-grid grids\source_grouped_logo.txt
python runs\validate_source_grouped_plan.py --completed-grid grids\source_grouped_rank_fixed_scale.txt
python runs\aggregate_source_grouped_results.py
python runs\audit_source_grouped_results.py
```

Aggregation produces three-seed directional, LOGO, and fixed-multiplier rank
tables plus expected alert precision at 0.1%, 0.5%, 1%, 5%, and 10% CG
prevalence (`results/source_grouped_summary/`).

## Follow-up jobs

The baselines, the paired Rs/standard QLoRA comparison, and the training-recipe
variants on the source-grouped DeepSeek partition are defined in
`grids/source_grouped_rebuild/plan.json` and run with `cloud/run_rebuild.sh`
(`runs/rebuild_worker.py`). Their per-job outputs are in
`results/source_grouped_followup/`, and the summary is in
`results/source_grouped_followup_summary/`.
