"""Run queued train_run.py jobs, N lanes at a time.

  python runs/run_queue.py grids/a1_grid.txt --lanes 2

Each non-empty, non-# line of the grid file is a set of train_run.py arguments.
Lane logs go to results/queue_logs/<line#>_<args>.log. Already-completed lines
(matching run prefix in results/runs.csv) are skipped, so the queue is resumable.
First-ever launches are staggered (--stagger, default 180 s) so concurrent
Triton/unsloth kernel compilation doesn't race.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# venv interpreter differs by OS (Windows: Scripts/python.exe, POSIX: bin/python)
PY = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if not PY.exists():  # fall back to the interpreter running this script
    PY = Path(sys.executable)
LEDGER = ROOT / "results" / "runs.csv"
LOGS = ROOT / "results" / "queue_logs"


def run_prefix(argline: str) -> str:
    """Mirror train_run.py's run_id prefix for dedup against the ledger."""
    def grab(flag, default=None):
        m = re.search(rf"--{flag}(?:\s+|=)(\S+)", argline)
        return m.group(1) if m else default
    tag, scaling, rank, seed = (grab("tag"), grab("scaling"),
                                grab("rank"), grab("seed"))
    loco = grab("holdout-category")
    base = f"{tag}_{scaling}_r{rank}_s{seed}"
    return base + (f"_loco-{loco}" if loco else "")


def completed_prefixes() -> set:
    if not LEDGER.exists():
        return set()
    with open(LEDGER, newline="", encoding="utf-8") as f:
        return {re.sub(r"_\d{8}$", "", row["run_id"])
                for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_file")
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--stagger", type=int, default=180,
                    help="seconds between initial lane launches")
    args = ap.parse_args()

    lines = [ln.strip() for ln in Path(args.grid_file).read_text().splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    done = completed_prefixes()
    queue = [(i, ln) for i, ln in enumerate(lines, 1)
             if run_prefix(ln) not in done]
    skipped = len(lines) - len(queue)
    print(f"{len(lines)} jobs, {skipped} already in ledger, {len(queue)} to run, "
          f"{args.lanes} lane(s)")
    LOGS.mkdir(parents=True, exist_ok=True)

    active = []  # (proc, line_no, logfile)
    first_launches = 0
    while queue or active:
        for proc, n, log in active[:]:
            if proc.poll() is not None:
                status = "OK" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
                print(f"[line {n}] {status}  ({log.name})")
                active.remove((proc, n, log))
        while queue and len(active) < args.lanes:
            n, argline = queue.pop(0)
            log = LOGS / f"{n:03d}_{run_prefix(argline)}.log"
            f = open(log, "w", encoding="utf-8")
            proc = subprocess.Popen(
                [str(PY), str(ROOT / "runs" / "train_run.py")] + argline.split(),
                stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
            print(f"[line {n}] started: {argline}")
            active.append((proc, n, log))
            first_launches += 1
            if queue and len(active) < args.lanes and first_launches <= args.lanes:
                time.sleep(args.stagger)
        time.sleep(20)
    print("Queue complete.")


if __name__ == "__main__":
    sys.exit(main())
