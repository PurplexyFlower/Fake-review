"""Head-to-head: TF-IDF vs Rs-QLoRA on the modern (DeepSeek) fake-review task.
Answers "is a neural model actually needed?" by comparing where they differ:
 (1) in-distribution accuracy overall + BY REVIEW LENGTH (short text = sparse n-grams)
 (2) cross-generator: detect 1.2B-generated fakes after training only on DeepSeek

Rs-QLoRA in-dist preds come from results/m1_*/test_probs.csv; its cross-generator
recall from results/external_eval/m1_on_1p2b.json (produced by:
  eval_external.py --adapter results/m1_*/adapter --csv dataset/sota_fakes_deepseek? )
TF-IDF is trained here on the DeepSeek train split.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runs"))
from build_modern_dataset import normalize_text  # same ASCII normalisation as training

DS = ROOT / "dataset" / "modern_reviews_deepseek.csv"
# the 2nd generator's fakes for cross-gen: raw fakes file, or a full lfm dataset
# (CG rows), whichever is present (e.g. fetched from HF on the cloud box)
FAKES_1P2B_CANDS = [ROOT / "dataset" / "sota_fakes.csv",
                    ROOT / "dataset" / "modern_reviews_lfm.csv"]
FAKES_1P2B = next((p for p in FAKES_1P2B_CANDS if p.exists()), FAKES_1P2B_CANDS[0])
FAKES_1P2B_NORM = ROOT / "dataset" / "sota_fakes_1p2b_norm.csv"  # normalized for eval
SPLIT_SEED = 1998


def m1_dir():
    c = [d for d in (ROOT / "results").glob("m1_rs_r64_s1998_*")
         if (d / "test_probs.csv").exists()]
    if not c:
        raise SystemExit("no finished m1 run yet (results/m1_rs_r64_s1998_*/test_probs.csv)")
    return sorted(c)[-1]


# ---- split (same seed as training) ----
ds = load_dataset("csv", data_files=str(DS))["train"]
sp = ds.train_test_split(test_size=0.2, seed=SPLIT_SEED)
sp2 = sp["test"].train_test_split(test_size=0.7, seed=SPLIT_SEED)
tr, te = sp["train"], sp2["test"]
ytr = np.array([1 if x == "CG" else 0 for x in tr["label"]])
yte = np.array([1 if x == "CG" else 0 for x in te["label"]])

# ---- TF-IDF ----
wv = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
cv = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2)
Xtr = hstack([wv.fit_transform(tr["text_"]), cv.fit_transform(tr["text_"])]).tocsr()
Xte = hstack([wv.transform(te["text_"]), cv.transform(te["text_"])]).tocsr()
lr = LogisticRegression(C=4.0, max_iter=2000).fit(Xtr, ytr)
tf_pred = lr.predict(Xte)

# ---- Rs-QLoRA in-dist preds (align by order; validate) ----
run = m1_dir()
probs = list(csv.DictReader(open(run / "test_probs.csv", encoding="utf-8")))
assert len(probs) == len(te), (len(probs), len(te))
rs_true = np.array([1 if r["true"] == "CG" else 0 for r in probs])
agree = (rs_true == yte).mean()
rs_pred = np.array([1 if float(r["p_CG"]) >= 0.5 else 0 for r in probs])
wl = np.array([len(t.split()) for t in te["text_"]])
print(f"order-check label agreement: {agree:.4f}  (run: {run.name})\n")

print(f"{'slice':<16}{'n':>6}{'TF-IDF':>9}{'Rs-QLoRA':>10}{'diff':>7}")
print("-" * 48)
def line(tag, m):
    if m.sum() == 0:
        return
    a = accuracy_score(yte[m], tf_pred[m]) * 100
    b = accuracy_score(yte[m], rs_pred[m]) * 100
    print(f"{tag:<16}{int(m.sum()):>6}{a:>8.2f}%{b:>9.2f}%{b-a:>+7.2f}")
line("overall", np.ones(len(yte), bool))
print("by length:")
for lo, hi in [(0, 8), (8, 15), (15, 25), (25, 40), (40, 70), (70, 10**9)]:
    line(f"  {lo}-{hi if hi < 10**9 else 'inf'}w", (wl >= lo) & (wl < hi))

# ---- cross-generator: detect 1.2B fakes (trained only on DeepSeek) ----
# normalise the 1.2B fakes the SAME way as training, and write a CSV both
# detectors share (so the comparison isn't confounded by smart quotes etc.)
print("\n== cross-generator: recall on UNSEEN 1.2B-generated fakes ==")
if FAKES_1P2B.exists():
    # take CG rows only (a full dataset has both classes; a raw fakes file is all CG)
    f12 = [normalize_text(r["text_"]) for r in csv.DictReader(open(FAKES_1P2B, encoding="utf-8"))
           if r.get("label", "CG") == "CG"]
    f12 = [t for t in f12 if len(t.split()) >= 3]
    print(f"(cross-gen 1.2B source: {FAKES_1P2B.name})")
    with open(FAKES_1P2B_NORM, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["text_", "label"])
        for t in f12:
            w.writerow([t, "CG"])
    X12 = hstack([wv.transform(f12), cv.transform(f12)]).tocsr()
    tf_recall = (lr.predict(X12) == 1).mean()
    print(f"  TF-IDF   recall on 1.2B fakes: {tf_recall*100:.2f}%   (n={len(f12)})")
else:
    print("  (1.2B fakes csv not found)")
j = ROOT / "results" / "external_eval" / "m1_on_1p2b.json"
if j.exists():
    d = json.loads(j.read_text())
    print(f"  Rs-QLoRA recall on 1.2B fakes: {d.get('recall_CG', d.get('flag_rate_CG'))*100:.2f}%")
else:
    print(f"  Rs-QLoRA: run\n    eval_external.py --adapter {run/'adapter'} "
          f"--csv {FAKES_1P2B_NORM} --label-col label --pos-value CG --name m1_on_1p2b")
