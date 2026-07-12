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
import hashlib
import json
import os
import re
import shlex
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
DEFAULT_DATA = ROOT / "dataset" / "fake reviews dataset.csv"


def grab(argline, flag, default=None):
    m = re.search(rf"--{flag}(?:\s+|=)(\S+)", argline)
    return m.group(1) if m else default


def run_prefix(argline: str) -> str:
    """Mirror train_run.py's run_id prefix for dedup against the ledger."""
    tag, scaling, rank, seed = (grab(argline, "tag"), grab(argline, "scaling"),
                                grab(argline, "rank"), grab(argline, "seed"))
    loco = grab(argline, "holdout-category")
    base = f"{tag}_{scaling}_r{rank}_s{seed}"
    return base + (f"_loco-{loco}" if loco else "")


def completed_rows():
    if not LEDGER.exists():
        return []
    with open(LEDGER, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


_HASH_CACHE = {}


def file_sha256(path):
    path = Path(path).resolve()
    if path not in _HASH_CACHE:
        _HASH_CACHE[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _HASH_CACHE[path]


def is_completed(argline, rows, require_adapter=False):
    prefix = run_prefix(argline)
    matches = [row for row in rows
               if re.sub(r"_\d{8}$", "", row["run_id"]) == prefix]
    if not matches or not require_adapter:
        return bool(matches)

    data_arg = grab(argline, "data", str(DEFAULT_DATA))
    data_path = Path(data_arg)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    expected_hash = file_sha256(data_path)
    for row in matches:
        run_dir = ROOT / "results" / row["run_id"]
        config_path = run_dir / "config.json"
        adapter = run_dir / "adapter"
        if not (config_path.exists() and (run_dir / "metrics.json").exists()
                and (run_dir / "test_probs.csv").exists()
                and (adapter / "adapter_config.json").exists()
                and (adapter / "adapter_model.safetensors").exists()):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_rank = int(grab(argline, "rank"))
        expected = {
            "tag": grab(argline, "tag"),
            "scaling": grab(argline, "scaling"),
            "rank": expected_rank,
            "alpha": int(grab(argline, "alpha", 2 * expected_rank)),
            "seed": int(grab(argline, "seed")),
            "scheduler": grab(argline, "scheduler", "cosine_with_restarts"),
            "label_smoothing": float(grab(argline, "label-smoothing", 0.01)),
            "holdout_category": grab(argline, "holdout-category"),
        }
        same_config = all(config.get(k) == v for k, v in expected.items())
        if same_config and config.get("data_sha256") == expected_hash:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_file")
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--stagger", type=int, default=180,
                    help="seconds between initial lane launches")
    ap.add_argument("--require-adapter", action="store_true",
                    help="skip only when adapter + metrics exist for the current data bytes")
    args = ap.parse_args()

    lines = [ln.strip() for ln in Path(args.grid_file).read_text().splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    done = completed_rows()
    queue = [(i, ln) for i, ln in enumerate(lines, 1)
             if not is_completed(ln, done, args.require_adapter)]
    skipped = len(lines) - len(queue)
    print(f"{len(lines)} jobs, {skipped} already in ledger, {len(queue)} to run, "
          f"{args.lanes} lane(s)")
    LOGS.mkdir(parents=True, exist_ok=True)

    active = []  # (proc, line_no, logfile)
    first_launches = 0
    failed = []
    while queue or active:
        for proc, n, log in active[:]:
            if proc.poll() is not None:
                status = "OK" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
                print(f"[line {n}] {status}  ({log.name})")
                if proc.returncode != 0:
                    failed.append((n, proc.returncode, log))
                active.remove((proc, n, log))
        while queue and len(active) < args.lanes:
            n, argline = queue.pop(0)
            log = LOGS / f"{n:03d}_{run_prefix(argline)}.log"
            f = open(log, "w", encoding="utf-8")
            proc = subprocess.Popen(
                [str(PY), str(ROOT / "runs" / "train_run.py")] + shlex.split(argline),
                stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
            print(f"[line {n}] started: {argline}")
            active.append((proc, n, log))
            first_launches += 1
            if queue and len(active) < args.lanes and first_launches <= args.lanes:
                time.sleep(args.stagger)
        time.sleep(20)
    if failed:
        print("Queue incomplete; failed jobs:")
        for n, rc, log in failed:
            print(f"  line {n}: rc={rc} ({log})")
        return 1
    print("Queue complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
