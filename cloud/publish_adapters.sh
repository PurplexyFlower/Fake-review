#!/usr/bin/env bash
# Publish the paper's saved PEFT adapters and reproducibility evidence to one
# resumable Hugging Face model repository. Adapter weights stay on the instance.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

: "${HF_REPO_ID:?set HF_REPO_ID to the destination model repository}"
TAGS="m1,m1s,xg,xl,xd,xglm,pwogpt2,pwolfm,pwodeepseek,pwoglm,ctrl_orperm"

python runs/build_adapter_manifest.py \
  --tags "$TAGS" --output results/adapter_manifest.json

hf upload-large-folder "$HF_REPO_ID" results --repo-type model --private \
  --include README.md PAPER_FINDINGS.md adapter_manifest.json runs.csv \
            cross_gen_results.json baselines_transformer.csv head_to_head.log \
            "external_eval/*.json" \
            "*/adapter/*" "*/adapter/**/*" "*/config.json" "*/metrics.json" \
  --num-workers 4

python runs/verify_hf_artifacts.py --repo "$HF_REPO_ID"
echo "published and remotely verified: https://huggingface.co/$HF_REPO_ID"
