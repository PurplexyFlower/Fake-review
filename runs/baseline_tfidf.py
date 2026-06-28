"""C1/C2: classical same-split baselines (TF-IDF + Logistic Regression / linear SVM).

CPU only. Uses the EXACT paper split (split seed 1998, same ops as train_run.py)
so the numbers are directly comparable to the Rs-QLoRA models. Reports test
metrics over 5 seeds (mean +- std), full and leakage-cleaned, and writes
per-model test probabilities + a ledger row to results/baselines.csv.
"""
import pyarrow  # noqa: F401  must precede heavy stacks in this venv
import csv
import statistics as st
from pathlib import Path

import numpy as np
from datasets import load_dataset
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "fake reviews dataset.csv"
RESULTS = ROOT / "results"
LEDGER = RESULTS / "baselines.csv"
SPLIT_SEED = 1998
SEEDS = [1998, 7, 42, 123, 2026]
LEDGER_COLS = ["model", "seed", "test_acc", "test_f1_OR", "test_f1_CG",
               "test_roc_auc", "test_pr_auc", "test_acc_cleaned"]


def paper_split(data):
    full = load_dataset("csv", data_files=data)["train"]
    full = full.add_column("orig_idx", list(range(len(full))))
    sp = full.train_test_split(test_size=0.2, seed=SPLIT_SEED)
    sp2 = sp["test"].train_test_split(test_size=0.7, seed=SPLIT_SEED)
    return sp["train"], sp2["train"], sp2["test"]


def y_of(ds):
    return np.array([1 if x == "CG" else 0 for x in ds["label"]])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--tag", default="", help="suffix on model name (e.g. _modern)")
    args = ap.parse_args()

    train, val, test = paper_split(args.data)
    ytr, yte = y_of(train), y_of(test)
    Xtr_txt, Xte_txt = train["text_"], test["text_"]

    # leakage-clean only applies to the original paper set (its dedup audit)
    excl_path = RESULTS / "test_exclusions.csv"
    if args.data == str(DATA) and excl_path.exists():
        excl = {int(r["orig_idx"]) for r in csv.DictReader(open(excl_path, encoding="utf-8"))}
        keep = np.array([oi not in excl for oi in test["orig_idx"]])
    else:
        keep = np.ones(len(test["orig_idx"]), dtype=bool)

    # word(1-2) + char(3-5) tf-idf — standard strong text-clf featurization
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                          max_features=200_000)
    cvec = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb",
                           ngram_range=(3, 5), min_df=2, max_features=200_000)
    from scipy.sparse import hstack
    Xtr = hstack([vec.fit_transform(Xtr_txt), cvec.fit_transform(Xtr_txt)]).tocsr()
    Xte = hstack([vec.transform(Xte_txt), cvec.transform(Xte_txt)]).tocsr()
    print(f"train {Xtr.shape}  test {Xte.shape}")

    RESULTS.mkdir(exist_ok=True)
    rows = []

    def evaluate(name, p_cg, seed):
        pred = (p_cg >= 0.5).astype(int)
        m = {
            "model": name, "seed": seed,
            "test_acc": accuracy_score(yte, pred),
            "test_f1_OR": f1_score(yte, pred, pos_label=0),
            "test_f1_CG": f1_score(yte, pred, pos_label=1),
            "test_roc_auc": roc_auc_score(yte, p_cg),
            "test_pr_auc": average_precision_score(yte, p_cg),
            "test_acc_cleaned": accuracy_score(yte[keep], pred[keep]),
        }
        rows.append(m)
        return m

    lr_name, svm_name = "tfidf_logreg" + args.tag, "tfidf_linsvm" + args.tag
    for seed in SEEDS:
        lr = LogisticRegression(C=4.0, max_iter=2000, random_state=seed)
        lr.fit(Xtr, ytr)
        evaluate(lr_name, lr.predict_proba(Xte)[:, 1], seed)

        svm = LinearSVC(C=0.5, random_state=seed)
        svm.fit(Xtr, ytr)
        # map SVM margin -> [0,1] pseudo-prob for AUC/threshold (rank-preserving)
        evaluate(svm_name, expit(svm.decision_function(Xte)), seed)

    # write ledger
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    # summary mean +- std
    print(f"\n== same-split baselines ({args.data}), mean +- std over 5 seeds ==")
    for model in (lr_name, svm_name):
        sub = [r for r in rows if r["model"] == model]
        def ms(k):
            vals = [r[k] for r in sub]
            return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)
        a = ms("test_acc"); f = ms("test_f1_CG")
        roc = ms("test_roc_auc"); pr = ms("test_pr_auc"); ac = ms("test_acc_cleaned")
        print(f"  {model:14}  acc={a[0]*100:.3f}+-{a[1]*100:.3f}  "
              f"f1_CG={f[0]*100:.3f}+-{f[1]*100:.3f}  "
              f"roc={roc[0]:.4f}  pr={pr[0]:.4f}  acc_clean={ac[0]*100:.3f}")


if __name__ == "__main__":
    main()
