"""Shared path helpers so analysis/eval scripts don't hardcode a timestamped run
dir (a fresh rerun on new hardware produces a different timestamp)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def headline_run_dir():
    """Latest completed headline run: Rs-QLoRA r64, paper split seed 1998, with a
    saved adapter. Falls back across timestamps so reruns Just Work."""
    cands = [d for d in RESULTS.glob("a1_rs_r64_s1998_*")
             if d.is_dir() and (d / "adapter").exists()]
    if not cands:
        raise FileNotFoundError(
            "no headline run found (results/a1_rs_r64_s1998_*/adapter). "
            "Run: python runs/train_run.py --tag a1 --scaling rs --rank 64 --seed 1998")
    return sorted(cands)[-1]
