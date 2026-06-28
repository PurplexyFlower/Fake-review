"""Phase 3 orchestrator: run all same-split baselines, resumable.

 - C1/C2: TF-IDF + LogReg/SVM (once, CPU)        -> results/baselines.csv
 - C3/C4/C5: 4 transformer models, full fine-tune,
   LR sweep x 3 seeds each                        -> results/baselines_transformer.csv

Sequential (one GPU job at a time). Skips any (model, lr, seed) already in the
ledger, so it is safe to re-run after an interruption.

  python runs/run_baselines.py                 # everything
  python runs/run_baselines.py --models roberta-base deberta-v3-base
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if not PY.exists():
    PY = Path(sys.executable)
RESULTS = ROOT / "results"

# per-model train batch (large models -> smaller batch; grad-accum keeps eff=32)
MODELS = {
    "roberta-base": 32,
    "microsoft/deberta-v3-base": 16,
    "answerdotai/ModernBERT-large": 16,
    "google/gemma-3-270m": 16,
}
LRS = [1e-5, 2e-5, 5e-5]
SEEDS = [1998, 7, 42]


def done_transformer():
    p = RESULTS / "baselines_transformer.csv"
    if not p.exists():
        return set()
    return {(r["model"], r["lr"], r["seed"])
            for r in csv.DictReader(open(p, encoding="utf-8"))}


def tfidf_done():
    p = RESULTS / "baselines.csv"
    return p.exists() and p.stat().st_size > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--skip-tfidf", action="store_true")
    args = ap.parse_args()

    if not args.skip_tfidf and not tfidf_done():
        print(">> C1/C2 TF-IDF baselines")
        subprocess.run([str(PY), str(ROOT / "runs" / "baseline_tfidf.py")],
                       cwd=str(ROOT), check=False)

    have = done_transformer()
    for model in args.models:
        short = model.split("/")[-1]
        batch = MODELS.get(model, 16)
        for lr in LRS:
            for seed in SEEDS:
                key = (short, f"{lr:g}", str(seed))
                if key in have:
                    print(f"skip {key} (in ledger)")
                    continue
                print(f">> {short}  lr={lr:g}  seed={seed}  batch={batch}")
                rc = subprocess.run(
                    [str(PY), str(ROOT / "runs" / "baseline_transformer.py"),
                     "--model", model, "--lr", str(lr), "--seed", str(seed),
                     "--batch", str(batch)],
                    cwd=str(ROOT)).returncode
                if rc != 0:
                    print(f"!! FAILED {key} rc={rc} (continuing)")
    print("baselines done.")


if __name__ == "__main__":
    main()
