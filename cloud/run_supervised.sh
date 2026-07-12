#!/usr/bin/env bash
# Supervisor-friendly wrapper for a long cloud run. It mirrors stdout/stderr to
# results/run_all.log while keeping the process in the foreground for supervisor.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"
mkdir -p results
exec > >(tee -a results/run_all.log) 2>&1
echo "[supervised] start $(date -Is)  SCOPE=${SCOPE:-modern}  LANES=${LANES:-1}"
exec bash cloud/run_all.sh
