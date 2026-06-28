"""Audit a modern dataset for spurious 'tells' that would inflate TF-IDF:
top discriminative features, induced-punctuation/word checks, length-tail leak,
near-duplicate train/test leakage. Usage: python runs/diagnose_tells.py --data <csv>"""
import argparse
import csv
import numpy as np
from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

SPLIT_SEED = 1998
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
A = ap.parse_args()

ds = load_dataset("csv", data_files=A.data)["train"]
sp = ds.train_test_split(test_size=0.2, seed=SPLIT_SEED)
sp2 = sp["test"].train_test_split(test_size=0.7, seed=SPLIT_SEED)
tr, te = sp["train"], sp2["test"]
ytr = np.array([1 if x == "CG" else 0 for x in tr["label"]])
yte = np.array([1 if x == "CG" else 0 for x in te["label"]])

# --- top discriminative features (word 1-2 + char 3-5) ---
wv = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=200_000)
cv = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000)
Xtr = hstack([wv.fit_transform(tr["text_"]), cv.fit_transform(tr["text_"])]).tocsr()
Xte = hstack([wv.transform(te["text_"]), cv.transform(te["text_"])]).tocsr()
names = np.array(list(wv.get_feature_names_out()) + ["CHR:" + c for c in cv.get_feature_names_out()])
lr = LogisticRegression(C=4.0, max_iter=2000).fit(Xtr, ytr)
acc = accuracy_score(yte, lr.predict(Xte))
print(f"TF-IDF test acc = {acc*100:.2f}%")
co = lr.coef_[0]
print("\nTOP 20 -> CG (fake):")
for i in np.argsort(co)[-20:][::-1]:
    print(f"  {co[i]:+.2f}  {names[i]!r}")
print("\nTOP 20 -> OR (human):")
for i in np.argsort(co)[:20]:
    print(f"  {co[i]:+.2f}  {names[i]!r}")

# --- induced-artifact frequency checks (all rows) ---
texts = ds["text_"]; labs = np.array([1 if x == "CG" else 0 for x in ds["label"]])
def frac(substr, y):
    sel = [substr in t for t in texts]
    sel = np.array(sel)
    return sel[labs == y].mean()
print("\n== induced-artifact frequency (CG vs OR) ==")
for s in [" - ", "...", "Arrived", "arrived", "pack", "shipp", "deliver", "Came super"]:
    print(f"  {s!r:14} CG={frac(s,1)*100:5.1f}%  OR={frac(s,0)*100:5.1f}%")

# --- length tail leak ---
wl = np.array([len(t.split()) for t in te["text_"]])
print(f"\nlength-only ROC-AUC (test): {roc_auc_score(yte, -wl):.4f}")
m = wl <= 130
print(f"TF-IDF acc on test reviews <=130 words ({m.sum()}/{len(m)}): "
      f"{accuracy_score(yte[m], lr.predict(Xte)[m])*100:.2f}%")

# --- near-duplicate train/test leakage (shared 5-gram shingles) ---
def shingles(t):
    w = t.lower().split()
    return {" ".join(w[i:i+5]) for i in range(len(w)-4)} or {t.lower()}
tr_sh = set().union(*[shingles(t) for t in tr["text_"]])
dup = sum(1 for t in te["text_"] if len(shingles(t) & tr_sh) >= 2)
print(f"\ntest reviews sharing >=2 5-gram shingles with train: {dup}/{len(te)} "
      f"({dup/len(te)*100:.1f}%)")
