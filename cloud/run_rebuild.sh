#!/usr/bin/env bash
# Bounded foreground workers for native Supervisor. No automatic retraining.
set -euo pipefail
cd /workspace/rsqlora-review
: "${REBUILD_SHARD:?set REBUILD_SHARD to 1, 2, or 3}"
case "$REBUILD_SHARD" in 1|2|3) ;; *) exit 2 ;; esac
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
case "${REBUILD_FAMILY:?set gpu or cpu}" in
  gpu|cpu) exec python runs/rebuild_worker.py worker --shard "$REBUILD_SHARD" --family "$REBUILD_FAMILY" ;;
  *) exit 2 ;;
esac
