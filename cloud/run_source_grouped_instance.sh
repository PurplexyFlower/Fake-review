#!/usr/bin/env bash
# Run one balanced source-grouped shard on one single-GPU cloud instance.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

: "${INSTANCE_ID:?set INSTANCE_ID to 1, 2, or 3}"
case "$INSTANCE_ID" in
  1|2|3) ;;
  *) echo "INSTANCE_ID must be 1, 2, or 3" >&2; exit 2 ;;
esac

# One worker per checkout, even if a second SSH session tries another shard ID.
# Children inherit the lock, so an orphaned training process still blocks a
# duplicate worker after an unexpected parent-process exit.
command -v flock >/dev/null || { echo "missing flock (util-linux)" >&2; exit 2; }
mkdir -p results/source_grouped_control
exec 9>results/source_grouped_control/worker.lock
if ! flock -n 9; then
  echo "a source-grouped worker is already active in this checkout" >&2
  exit 75
fi

if [ ! -x .venv/bin/python ]; then
  echo "missing .venv; run: bash cloud/setup.sh" >&2
  exit 2
fi
source .venv/bin/activate

# The protocol is deliberately single-GPU. On an accidentally multi-GPU offer,
# expose only one device unless the caller already selected one explicitly.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [ "$GPU_COUNT" != "1" ]; then
  echo "source-grouped protocol requires exactly one visible GPU; got $GPU_COUNT" >&2
  exit 2
fi

GRID="grids/source_grouped/instance_${INSTANCE_ID}_full.txt"
python runs/build_source_grouped_datasets.py
python runs/validate_source_grouped_plan.py
CAPTURE_ARGS=(--instance-id "$INSTANCE_ID" --grid "$GRID")
if [ -n "${VAST_OFFER_ID:-}" ]; then
  CAPTURE_ARGS+=(--offer-id "$VAST_OFFER_ID")
fi
python runs/capture_source_grouped_environment.py "${CAPTURE_ARGS[@]}"

echo "instance=$INSTANCE_ID mode=full grid=$GRID"
python runs/run_queue.py "$GRID" --lanes 1 --require-adapter
python runs/validate_source_grouped_plan.py --completed-grid "$GRID"
python runs/evaluate_source_grouped.py --grid "$GRID"
python runs/package_source_grouped_results.py \
  --instance-id "$INSTANCE_ID" --grid "$GRID"
echo "SOURCE-GROUPED SHARD COMPLETE: instance=$INSTANCE_ID mode=full"
