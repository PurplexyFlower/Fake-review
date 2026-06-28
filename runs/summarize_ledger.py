"""Aggregate results/runs.csv into the reviewer-response tables:
 - A1: mean +- std per (scaling, rank) cell for acc / F1 / ROC / PR
 - E4: convergence (best_step) and wall-clock per cell  [timing anomaly]
 - Phase 5: peak VRAM (allocated / reserved) per rank
Pure analysis, no GPU. Also emits results/a1_summary.csv.
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "runs.csv"
OUT = ROOT / "results" / "a1_summary.csv"


def ms(vals):
    vals = [v for v in vals if v is not None]
    return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)


rows = list(csv.DictReader(open(LEDGER, encoding="utf-8")))
cells = defaultdict(list)
for r in rows:
    if r["tag"] in ("a1", "a2"):
        cells[(r["scaling"], int(r["rank"]))].append(r)

f = lambda r, k: float(r[k]) if r[k] else None
order = sorted(cells, key=lambda k: (k[0], k[1]))

print("== A1: test metrics, mean +- std ==")
hdr = f"{'cell':<14}{'n':>2}  {'acc':>16}{'f1_CG':>16}{'roc_auc':>9}{'pr_auc':>9}"
print(hdr); print("-" * len(hdr))
summary = []
for key in order:
    rs = cells[key]
    acc = ms([f(r, "test_acc") for r in rs])
    f1 = ms([f(r, "test_f1_CG") for r in rs])
    roc = ms([f(r, "test_roc_auc") for r in rs])
    pr = ms([f(r, "test_pr_auc") for r in rs])
    bs = ms([f(r, "best_step") for r in rs])
    rt = ms([f(r, "train_runtime_s") for r in rs])
    pa = ms([f(r, "peak_alloc_gb") for r in rs])
    prv = ms([f(r, "peak_reserved_gb") for r in rs])
    cell = f"{key[0]}/r{key[1]}"
    print(f"{cell:<14}{len(rs):>2}  {acc[0]*100:7.3f}+-{acc[1]*100:5.3f}"
          f"{f1[0]*100:8.3f}+-{f1[1]*100:5.3f}{roc[0]:9.4f}{pr[0]:9.4f}")
    summary.append({
        "cell": cell, "scaling": key[0], "rank": key[1], "n": len(rs),
        "acc_mean": acc[0], "acc_std": acc[1], "f1_CG_mean": f1[0],
        "f1_CG_std": f1[1], "roc_auc_mean": roc[0], "pr_auc_mean": pr[0],
        "best_step_mean": bs[0], "runtime_s_mean": rt[0],
        "peak_alloc_gb": pa[0], "peak_reserved_gb": prv[0],
    })

print("\n== E4: convergence & wall-clock (best_step / runtime) ==")
print(f"{'cell':<14}{'best_step':>11}{'runtime_min':>13}")
for s in summary:
    print(f"{s['cell']:<14}{s['best_step_mean']:>11.0f}{s['runtime_s_mean']/60:>13.1f}")

print("\n== Phase 5: peak VRAM by config (GB) ==")
print(f"{'cell':<14}{'alloc':>8}{'reserved':>10}")
for s in summary:
    print(f"{s['cell']:<14}{s['peak_alloc_gb']:>8.2f}{s['peak_reserved_gb']:>10.2f}")

with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)
print(f"\nwrote {OUT}")
