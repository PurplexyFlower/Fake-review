"""4x4 cross-generator matrix: train a detector on each generator's fakes, test it
on every generator's held-out fakes. The realistic robustness question — does a
detector trained on yesterday's generator catch tomorrow's?

Generators (all share the same real-human OR half; only the fake half differs,
all normalised the same way):
  GPT2     dataset/modern_reviews_gpt2.csv      (Kaggle GPT-2-era CG)
  LFM      dataset/modern_reviews_lfm.csv       (LFM2.5-1.2B)
  DeepSeek dataset/modern_reviews_deepseek.csv  (DeepSeek-v4-pro)
  GLM      dataset/modern_reviews_glm.csv       (GLM-5.2)

Reports, for TF-IDF and Rs-QLoRA, a 4x4 matrix of
DETECTION RECALL = fraction of generator-j fakes the i-trained detector flags.
Diagonal = in-distribution; off-diagonal = cross-generator transfer.

  python runs/cross_gen_matrix.py            # TF-IDF always; neural if adapters present
  python runs/cross_gen_matrix.py --no-neural
"""
import argparse
import csv
import hashlib
import json
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
    return mat, fpr


def adapter_dirs(tag, expected_seeds, data_path=None):
    """One completed adapter for every expected seed; fail on partial matrices."""
    ledger = ROOT / "results" / "runs.csv"
    if not ledger.exists():
        raise RuntimeError("results/runs.csv is missing")
    with ledger.open(newline="", encoding="utf-8") as f:
        completed = {row["run_id"] for row in csv.DictReader(f)}

    by_seed = {}
    expected_hash = (hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
                     if data_path is not None else None)
    for run_dir in sorted((ROOT / "results").glob(f"{tag}_rs_r64_s*")):
        adapter = run_dir / "adapter"
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        if (run_dir.name not in completed or not adapter.exists()
                or not config_path.exists() or not metrics_path.exists()):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (config.get("tag") != tag or config.get("scaling") != "rs"
                or int(config.get("rank", -1)) != 64):
            continue
        if expected_hash is not None and config.get("data_sha256") != expected_hash:
            continue
        seed = int(config["seed"])
        if seed in expected_seeds:
            by_seed[seed] = adapter  # sorted timestamps: keep the latest rerun

    missing = [seed for seed in expected_seeds if seed not in by_seed]
    if missing:
        raise RuntimeError(
            f"incomplete adapters for tag={tag}: missing seeds {missing}; "
            f"expected {list(expected_seeds)}")
    return [by_seed[seed] for seed in expected_seeds]


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
        adapters[n] = adapter_dirs(data[n]["tag"], (1998, 7, 42))
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
    run_ids = {n: [adapter.parent.name for adapter in adapters[n]] for n in names}
    return mat, fpr, n_seeds, run_ids


def logo_tfidf(data, names):
    """Leave-one-generator-out: TF-IDF trained on the pooled 3-generator set
    (same 80% train protocol as train_run), tested on the held-out generator."""
    rows = []
    for n in names:
        pooled = ROOT / "dataset" / f"pooled_wo_{data[n]['key']}.csv"
        if not pooled.exists():
            print(f"\n[LOGO skipped] {pooled.name} missing — run runs/build_pooled_dataset.py")
            raise RuntimeError(
                f"{pooled.name} missing — run runs/build_pooled_dataset.py")
        ds = load_dataset("csv", data_files=str(pooled))["train"]
        pooled_or = {t.strip().casefold() for t, label in
                     zip(ds["text_"], ds["label"]) if label == "OR"}
        heldout_or = {t.strip().casefold() for t in data[n]["te_or"]}
        overlap = pooled_or & heldout_or
        if overlap:
            raise RuntimeError(
                f"{pooled.name} leaks {len(overlap)} held-out human texts; "
                "rebuild it with: python runs/build_pooled_dataset.py")
        tr = ds.train_test_split(test_size=0.2, seed=SPLIT_SEED)["train"]
        y = np.array([1 if x == "CG" else 0 for x in tr["label"]])
        predict = fit_tfidf(tr["text_"], y)
        rows.append((n, predict(data[n]["te_fake"]), predict(data[n]["te_or"])))
    print("\n== LOGO: TF-IDF trained on the OTHER 3 generators "
          "(balanced; held-out humans excluded) ==")
    for n, rec, f in rows:
        print(f"  held-out {n:>9}: recall={rec*100:5.1f}%   FPR={f*100:.2f}%"
              "   (held-out human texts excluded from pooled data)")
    return [{"held_out": n, "recall": float(rec), "fpr": float(f)}
            for n, rec, f in rows]


def logo_neural(data, names):
    print("\n== LOGO: Rs-QLoRA trained on the OTHER 3 generators ==")
    rows = []
    for n in names:
        pooled = ROOT / "dataset" / f"pooled_wo_{data[n]['key']}.csv"
        dirs = adapter_dirs(f"pwo{data[n]['key']}", (1998,), pooled)
        recs, fprs = [], []
        for ad in dirs:
            flag, close = neural_flagger(ad)
            recs.append(flag(data[n]["te_fake"]))
            fprs.append(flag(data[n]["te_or"]))
            close()
        print(f"  held-out {n:>9}: recall={np.mean(recs)*100:5.1f}%   "
              f"FPR={np.mean(fprs)*100:.2f}%   (seeds={len(dirs)})")
        rows.append({"held_out": n, "recall": float(np.mean(recs)),
                     "fpr": float(np.mean(fprs)),
                     "run_ids": [adapter.parent.name for adapter in dirs]})
    return rows


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
    tfidf_mat, tfidf_fpr = tfidf_matrix(data, names)
    logo_tfidf_rows = logo_tfidf(data, names)
    result = {
        "generators": names,
        "dataset_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path, _tag in GENERATORS
        },
        "tfidf": {
            "recall_matrix": tfidf_mat.tolist(),
            "fpr": {name: float(tfidf_fpr[name]) for name in names},
        },
        "logo_tfidf": logo_tfidf_rows,
    }
    if not args.no_neural:
        neural_mat, neural_fpr, n_seeds, run_ids = neural_matrix(data, names)
        result["rs_qlora"] = {
            "recall_matrix": neural_mat.tolist(),
            "fpr": {name: float(neural_fpr[name]) for name in names},
            "seed_counts": n_seeds,
            "run_ids": run_ids,
        }
        result["logo_rs_qlora"] = logo_neural(data, names)
    out = ROOT / "results" / "cross_gen_results.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nstructured results -> {out}")


if __name__ == "__main__":
    main()
