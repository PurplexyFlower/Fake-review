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
            "key": csv_path.stem.split("_")[-1],   # gpt2 / lfm / deepseek / glm
            "tr_text": tr["text_"],
            "tr_y": np.array([1 if x == "CG" else 0 for x in tr["label"]]),
            "te_fake": [t for t, l in zip(te["text_"], te["label"]) if l == "CG"],
            "te_or": [t for t, l in zip(te["text_"], te["label"]) if l == "OR"],
        }
    return data


def fit_tfidf(texts, y):
    w = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
    c = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    X = hstack([w.fit_transform(texts), c.fit_transform(texts)]).tocsr()
    lr = LogisticRegression(C=4.0, max_iter=2000).fit(X, y)
    return lambda t: (lr.predict(hstack([w.transform(t), c.transform(t)]).tocsr()) == 1).mean()


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


def adapter_dirs(tag):
    """All trained adapters for a tag, any seed (matrix averages across seeds)."""
    return sorted(d / "adapter" for d in (ROOT / "results").glob(f"{tag}_rs_r64_s*")
                  if (d / "adapter").exists())


def neural_flagger(adapter):
    import torch
    from unsloth import FastModel
    from transformers import AutoModelForSequenceClassification
    model, tok = FastModel.from_pretrained(
        model_name=str(adapter), auto_model=AutoModelForSequenceClassification,
        max_seq_length=512, dtype=None, num_labels=2, load_in_4bit=True,
        full_finetuning=False)
    FastModel.for_inference(model); model.eval()
    dev = next(model.parameters()).device

    def flag_rate(texts):
        flagged = 0
        for k in range(0, len(texts), 64):
            enc = tok(texts[k:k+64], truncation=True, max_length=512,
                      padding=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                p = torch.softmax(model(**enc).logits.float(), -1)[:, 1]
            flagged += int((p >= 0.5).sum())
        return flagged / max(len(texts), 1)

    def close():
        import torch as _t
        nonlocal model
        del model
        _t.cuda.empty_cache()
    return flag_rate, close


def neural_matrix(data, names):
    adapters = {}
    for n in names:
        dirs = adapter_dirs(data[n]["tag"])
        if not dirs:
            print(f"\n[neural skipped] no adapter for {n} "
                  f"(results/{data[n]['tag']}_rs_r64_s*/adapter) — run grids/xgen.txt")
            return None
        adapters[n] = dirs
    n_seeds = {n: len(adapters[n]) for n in names}
    mat = np.zeros((len(names), len(names)))
    fpr = {}
    for i, ni in enumerate(names):
        recalls = np.zeros((len(adapters[ni]), len(names)))
        fprs = []
        for s, ad in enumerate(adapters[ni]):
            flag, close = neural_flagger(ad)
            for j, nj in enumerate(names):
                recalls[s][j] = flag(data[nj]["te_fake"])
            fprs.append(flag(data[ni]["te_or"]))
            close()
        mat[i] = recalls.mean(axis=0)
        fpr[ni] = float(np.mean(fprs))
    print_matrix(f"Rs-QLoRA cross-generator detection recall "
                 f"(mean over seeds: {n_seeds})", mat, names)
    print("  detector FPR on its own human test set: "
          + "  ".join(f"{n}={fpr[n]*100:.2f}%" for n in names))
    return mat


def logo_tfidf(data, names):
    """Leave-one-generator-out: TF-IDF trained on the pooled 3-generator set
    (same 80% train protocol as train_run), tested on the held-out generator."""
    rows = []
    for n in names:
        pooled = ROOT / "dataset" / f"pooled_wo_{data[n]['key']}.csv"
        if not pooled.exists():
            print(f"\n[LOGO skipped] {pooled.name} missing — run runs/build_pooled_dataset.py")
            return
        ds = load_dataset("csv", data_files=str(pooled))["train"]
        tr = ds.train_test_split(test_size=0.2, seed=SPLIT_SEED)["train"]
        y = np.array([1 if x == "CG" else 0 for x in tr["label"]])
        predict = fit_tfidf(tr["text_"], y)
        rows.append((n, predict(data[n]["te_fake"]), predict(data[n]["te_or"])))
    print("\n== LOGO: TF-IDF trained on the OTHER 3 generators (pooled, budget-matched) ==")
    for n, rec, f in rows:
        print(f"  held-out {n:>9}: recall={rec*100:5.1f}%   FPR={f*100:.2f}%"
              "   (note: human half is shared across sets by design)")


def logo_neural(data, names):
    print("\n== LOGO: Rs-QLoRA trained on the OTHER 3 generators ==")
    any_found = False
    for n in names:
        dirs = adapter_dirs(f"pwo{data[n]['key']}")
        if not dirs:
            print(f"  held-out {n:>9}: no adapter (run grids/logo.txt)")
            continue
        any_found = True
        recs, fprs = [], []
        for ad in dirs:
            flag, close = neural_flagger(ad)
            recs.append(flag(data[n]["te_fake"]))
            fprs.append(flag(data[n]["te_or"]))
            close()
        print(f"  held-out {n:>9}: recall={np.mean(recs)*100:5.1f}%   "
              f"FPR={np.mean(fprs)*100:.2f}%   (seeds={len(dirs)})")
    return any_found


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
    logo_tfidf(data, names)
    if not args.no_neural:
        try:
            neural_matrix(data, names)
            logo_neural(data, names)
        except Exception as e:
            print(f"\n[neural skipped] {str(e)[:120]}")


if __name__ == "__main__":
    main()
