#!/usr/bin/env bash
# Detach one cloud shard from SSH. Does not rent, stop, or delete an instance.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

: "${INSTANCE_ID:?set INSTANCE_ID to 1, 2, or 3}"
case "$INSTANCE_ID" in
  1|2|3) ;;
  *) echo "INSTANCE_ID must be 1, 2, or 3" >&2; exit 2 ;;
esac
command -v flock >/dev/null || { echo "missing flock (util-linux)" >&2; exit 2; }
command -v nohup >/dev/null || { echo "missing nohup (coreutils)" >&2; exit 2; }
if [ ! -x .venv/bin/python ]; then
  echo "missing .venv; run: bash cloud/setup.sh" >&2
  exit 2
fi

CONTROL=results/source_grouped_control
mkdir -p "$CONTROL"
exec 8>"$CONTROL/start.lock"
if ! flock -n 8; then
  echo "another launcher is starting a worker; inspect $CONTROL" >&2
  exit 75
fi
if ! flock -n "$CONTROL/worker.lock" true; then
  echo "worker already active; do not restart it. Inspect $CONTROL/*.log"
  exit 0
fi

LOG="$CONTROL/instance_${INSTANCE_ID}_$(date -u +%Y%m%dT%H%M%S)_$$.log"
export INSTANCE_ID
export PYTHONUNBUFFERED=1
nohup bash "$HERE/run_source_grouped_instance.sh" >"$LOG" 2>&1 < /dev/null 8>&- &
WORKER_PID=$!
printf '%s\n' "$WORKER_PID" >"$CONTROL/worker.pid"
printf '%s\n' "$LOG" >"$CONTROL/latest_log.txt"
sleep 2
if ! kill -0 "$WORKER_PID" 2>/dev/null; then
  echo "worker exited during startup; inspect: $LOG" >&2
  tail -n 30 "$LOG" >&2
  exit 1
fi
echo "worker launched: pid=$WORKER_PID instance=$INSTANCE_ID"
echo "monitor: tail -f $LOG"
echo "launch is not completion; verify the final ZIP and its SHA-256 locally."
