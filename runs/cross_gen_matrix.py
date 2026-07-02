"""3x3 cross-generator matrix: train a detector on each generator's fakes, test it
on every generator's held-out fakes. The realistic robustness question — does a
detector trained on yesterday's generator catch tomorrow's?

Generators (all share the same real-human OR half; only the fake half differs,
all normalised the same way):
  GPT2     dataset/modern_reviews_gpt2.csv      (Kaggle GPT-2-era CG)
  LFM      dataset/modern_reviews_lfm.csv       (LFM2.5-1.2B)
  DeepSeek dataset/modern_reviews_deepseek.csv  (DeepSeek-v4-pro)

Reports, for TF-IDF and (if the adapters exist) Rs-QLoRA, a 3x3 matrix of
DETECTION RECALL = fraction of generator-j fakes the i-trained detector flags.
Diagonal = in-distribution; off-diagonal = cross-generator transfer.

  python runs/cross_gen_matrix.py            # TF-IDF always; neural if adapters present
  python runs/cross_gen_matrix.py --no-neural
"""
import argparse
import numpy as np
from pathlib import Path
from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SEED = 1998
GENERATORS = [
    ("GPT2",     ROOT / "dataset" / "modern_reviews_gpt2.csv",     "xg"),
    ("LFM",      ROOT / "dataset" / "modern_reviews_lfm.csv",      "xl"),
    ("DeepSeek", ROOT / "dataset" / "modern_reviews_deepseek.csv", "xd"),
    ("GLM",      ROOT / "dataset" / "modern_reviews_glm.csv",      "xglm"),
]


def split(csv_path):
    ds = load_dataset("csv", data_files=str(csv_path))["train"]
    sp = ds.train_test_split(test_size=0.2, seed=SPLIT_SEED)
    sp2 = sp["test"].train_test_split(test_size=0.7, seed=SPLIT_SEED)
    return sp["train"], sp2["test"]


def load_all():
    data = {}
    for name, csv_path, tag in GENERATORS:
        tr, te = split(csv_path)
        data[name] = {
            "tag": tag,
            "tr_text": tr["text_"],
            "tr_y": np.array([1 if x == "CG" else 0 for x in tr["label"]]),
            "te_fake": [t for t, l in zip(te["text_"], te["label"]) if l == "CG"],
            "te_or": [t for t, l in zip(te["text_"], te["label"]) if l == "OR"],
        }
    return data


def print_matrix(title, mat, names):
    print(f"\n== {title} ==")
    print(f"{'train\\test':>12}" + "".join(f"{n:>11}" for n in names))
    for i, n in enumerate(names):
        row = "".join(f"{mat[i][j]*100:>10.1f}%" for j in range(len(names)))
        print(f"{n:>12}{row}")
    diag = np.mean([mat[i][i] for i in range(len(names))])
    off = np.mean([mat[i][j] for i in range(len(names)) for j in range(len(names)) if i != j])
    print(f"  mean in-distribution (diag) = {diag*100:.1f}%   "
          f"mean cross-generator (off-diag) = {off*100:.1f}%   "
          f"transfer gap = {(diag-off)*100:.1f} pts")


def tfidf_matrix(data, names):
    models = {}
    wv = {}; cv = {}
    for n in names:
        w = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
        c = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2)
        X = hstack([w.fit_transform(data[n]["tr_text"]),
                    c.fit_transform(data[n]["tr_text"])]).tocsr()
        models[n] = LogisticRegression(C=4.0, max_iter=2000).fit(X, data[n]["tr_y"])
        wv[n], cv[n] = w, c
    mat = np.zeros((len(names), len(names)))
    fpr = {}
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            Xj = hstack([wv[ni].transform(data[nj]["te_fake"]),
                         cv[ni].transform(data[nj]["te_fake"])]).tocsr()
            mat[i][j] = (models[ni].predict(Xj) == 1).mean()
        # FPR anchor: recall alone is gameable (flag-everything = 100% recall).
        Xor = hstack([wv[ni].transform(data[ni]["te_or"]),
                      cv[ni].transform(data[ni]["te_or"])]).tocsr()
        fpr[ni] = (models[ni].predict(Xor) == 1).mean()
    print_matrix("TF-IDF cross-generator detection recall", mat, names)
    print("  detector FPR on its own human test set: "
          + "  ".join(f"{n}={fpr[n]*100:.2f}%" for n in names))
    return mat


def neural_matrix(data, names):
    import torch
    from unsloth import FastModel
    from transformers import AutoModelForSequenceClassification
    adapters = {}
    for n in names:
        cands = [d for d in (ROOT / "results").glob(f"{data[n]['tag']}_rs_r64_s1998_*")
                 if (d / "adapter").exists()]
        if not cands:
            print(f"\n[neural skipped] no adapter for {n} "
                  f"(results/{data[n]['tag']}_rs_r64_s1998_*/adapter) — run grids/xgen.txt")
            return None
        adapters[n] = sorted(cands)[-1] / "adapter"
    mat = np.zeros((len(names), len(names)))
    fpr = {}

    def flag_rate(model, tok, dev, texts):
        flagged = 0
        for k in range(0, len(texts), 64):
            enc = tok(texts[k:k+64], truncation=True, max_length=512,
                      padding=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                p = torch.softmax(model(**enc).logits.float(), -1)[:, 1]
            flagged += int((p >= 0.5).sum())
        return flagged / max(len(texts), 1)

    for i, ni in enumerate(names):
        model, tok = FastModel.from_pretrained(
            model_name=str(adapters[ni]), auto_model=AutoModelForSequenceClassification,
            max_seq_length=512, dtype=None, num_labels=2, load_in_4bit=True,
            full_finetuning=False)
        FastModel.for_inference(model); model.eval()
        dev = next(model.parameters()).device
        for j, nj in enumerate(names):
            mat[i][j] = flag_rate(model, tok, dev, data[nj]["te_fake"])
        fpr[ni] = flag_rate(model, tok, dev, data[ni]["te_or"])
        del model; torch.cuda.empty_cache()
    print_matrix("Rs-QLoRA cross-generator detection recall", mat, names)
    print("  detector FPR on its own human test set: "
          + "  ".join(f"{n}={fpr[n]*100:.2f}%" for n in names))
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-neural", action="store_true")
    args = ap.parse_args()
    names = [g[0] for g in GENERATORS]
    missing = [str(g[1]) for g in GENERATORS if not g[1].exists()]
    if missing:
        raise SystemExit("missing datasets:\n  " + "\n  ".join(missing))
    data = load_all()
    print(f"generators: {names}  (fakes per held-out test set: "
          f"{[len(data[n]['te_fake']) for n in names]})")
    tfidf_matrix(data, names)
    if not args.no_neural:
        try:
            neural_matrix(data, names)
        except Exception as e:
            print(f"\n[neural skipped] {str(e)[:120]}")


if __name__ == "__main__":
    main()
