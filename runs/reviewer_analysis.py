"""Reviewer-response analyses computed ONLY from artifacts already on disk
(no retraining): leakage-cleaned metrics, error analysis (length / category /
confidence), and imbalance simulation. Headline model = Rs-QLoRA r64.
"""
import pyarrow  # noqa: F401  (must import before heavy stacks in this venv)
import csv
import statistics as st
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from _paths import headline_run_dir

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "fake reviews dataset.csv"
SPLIT_SEED = 1998
HEAD = headline_run_dir()   # latest Rs-QLoRA r64 / split-seed-1998 run with adapter


def load_probs(run_dir):
    rows = list(csv.DictReader(open(run_dir / "test_probs.csv", encoding="utf-8")))
    y = np.array([1 if r["true"] == "CG" else 0 for r in rows])
    p = np.array([float(r["p_CG"]) for r in rows])
    return rows, y, p


# ---- reconstruct exact test-split orig_idx order (same ops as train_run) ----
full = load_dataset("csv", data_files=str(DATA))["train"]
full = full.add_column("orig_idx", list(range(len(full))))
sp = full.train_test_split(test_size=0.2, seed=SPLIT_SEED)
sp2 = sp["test"].train_test_split(test_size=0.7, seed=SPLIT_SEED)
test = sp2["test"]
orig_idx = test["orig_idx"]
split_label = test["label"]

rows, y, p = load_probs(HEAD)
assert len(rows) == len(test), (len(rows), len(test))
# validate ordering: true labels in test_probs must equal split labels in order
agree = sum(1 for r, l in zip(rows, split_label) if r["true"] == l) / len(rows)
print(f"order-validation label agreement: {agree:.4f}  (1.0 => row i == test[i])")

# ---- full vs leakage-cleaned metrics ----
excl_orig = {int(r["orig_idx"]) for r in
             csv.DictReader(open(ROOT / "results" / "test_exclusions.csv"))}
keep = np.array([oi not in excl_orig for oi in orig_idx])


def report(tag, y, p):
    pred = (p >= 0.5).astype(int)
    print(f"  {tag:26} n={len(y):5d}  acc={accuracy_score(y,pred)*100:.3f}  "
          f"f1_CG={f1_score(y,pred,pos_label=1)*100:.3f}  "
          f"roc={roc_auc_score(y,p):.4f}  pr={average_precision_score(y,p):.4f}")


print("\n== full vs leakage-cleaned (headline rs_r64 seed1998) ==")
report("full test (5661)", y, p)
report(f"cleaned ({keep.sum()} rows)", y[keep], p[keep])
print(f"  excluded test rows: {(~keep).sum()}  (of {len(excl_orig)} flagged orig idx)")

# ---- confusion matrix + error analysis ----
pred = (p >= 0.5).astype(int)
tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
err = pred != y
print(f"\n== confusion (thr 0.5) ==  TP={tp} TN={tn} FP={fp} FN={fn}  "
      f"errors={int(err.sum())}  acc={accuracy_score(y,pred)*100:.3f}")
print(f"  FP = genuine(OR) flagged as CG: {fp}")
print(f"  FN = CG passed as genuine(OR): {fn}")

# error rate by review length (word count)
wc = np.array([len(r["text"].split()) for r in rows])
bins = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 10**9)]
print("\n== error rate by review length (words) ==")
for lo, hi in bins:
    m = (wc >= lo) & (wc < hi)
    if m.sum():
        print(f"  [{lo:>3},{hi if hi<10**9 else 'inf':>4}): n={m.sum():5d}  "
              f"err={err[m].mean()*100:5.2f}%  ({int(err[m].sum())} errs)")

# per-category error table
cats = {}
for i, r in enumerate(rows):
    cats.setdefault(r["category"], []).append(i)
print("\n== per-category error rate ==")
for c in sorted(cats, key=lambda c: -np.mean(err[cats[c]])):
    idx = cats[c]
    print(f"  {c:32} n={len(idx):5d}  err={err[idx].mean()*100:5.2f}%  "
          f"({int(err[idx].sum())})")

# confidence: model confidence on errors vs correct
conf = np.maximum(p, 1 - p)
print("\n== model confidence (max class prob) ==")
print(f"  correct: mean={conf[~err].mean():.4f}  median={np.median(conf[~err]):.4f}")
print(f"  errors : mean={conf[err].mean():.4f}  median={np.median(conf[err]):.4f}")
hi_conf_err = ((conf >= 0.90) & err).sum()
print(f"  high-confidence (>=0.90) errors: {hi_conf_err} / {int(err.sum())} "
      f"({hi_conf_err/max(err.sum(),1)*100:.1f}% of errors)")

# ---- imbalance simulation (prevalence reweighting, no resampling) ----
# keep all positives; downweight negatives so CG prevalence = pi
n_pos = int(y.sum()); n_neg = int((y == 0).sum())
print("\n== imbalance simulation (reweighted negatives, thr 0.5) ==")
print(f"  base test prevalence CG = {n_pos/len(y):.3f}")
fpr = fp / n_neg; tpr = tp / n_pos
for pi in (0.50, 0.10, 0.01):
    w = np.where(y == 1, 1.0, (n_pos / n_neg) * (1 - pi) / pi)
    prec = precision_score(y, pred, sample_weight=w)
    rec = recall_score(y, pred, sample_weight=w)
    prauc = average_precision_score(y, p, sample_weight=w)
    # closed-form precision check from TPR/FPR
    prec_cf = (pi * tpr) / (pi * tpr + (1 - pi) * fpr)
    print(f"  pi={pi:.2f}: precision={prec*100:5.2f}% (cf {prec_cf*100:5.2f})  "
          f"recall={rec*100:5.2f}%  PR-AUC={prauc:.4f}")
print(f"  (recall/TPR={tpr*100:.2f}% and FPR={fpr*100:.2f}% are prevalence-invariant)")
