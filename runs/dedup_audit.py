"""P0-1: duplicate / near-duplicate audit with cross-split leakage check (R1-6).

  python runs/dedup_audit.py            # threshold 0.90
  python runs/dedup_audit.py --near 0.95

Writes results/dedup_report.md and results/dedup_pairs.csv.
"""
import argparse
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "fake reviews dataset.csv"
RESULTS = ROOT / "results"
SPLIT_SEED = 1998


def normalize(t: str) -> str:
    t = re.sub(r"<[^>]+>|&\w+;", " ", t.lower())   # strip html/entities
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", type=float, default=0.90,
                    help="cosine similarity threshold for near-duplicates")
    args = ap.parse_args()

    full = load_dataset("csv", data_files=str(DATA))["train"]
    n = len(full)
    texts = [normalize(t) for t in full["text_"]]
    labels = full["label"]

    # Which split does each original row belong to? (same ops/seed as training)
    base = list(range(n))
    sp = full.add_column("orig_idx", base).train_test_split(
        test_size=0.2, seed=SPLIT_SEED)
    sp2 = sp["test"].train_test_split(test_size=0.7, seed=SPLIT_SEED)
    split_of = {}
    for name, ds in (("train", sp["train"]), ("val", sp2["train"]),
                     ("test", sp2["test"])):
        for i in ds["orig_idx"]:
            split_of[i] = name

    # ---- Exact duplicates (normalized) ----
    groups = defaultdict(list)
    for i, t in enumerate(texts):
        groups[hashlib.md5(t.encode()).hexdigest()].append(i)
    exact_groups = {h: g for h, g in groups.items() if len(g) > 1}
    exact_rows = sum(len(g) for g in exact_groups.values())

    # ---- Near-duplicates: char tf-idf + nearest neighbor ----
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          max_features=200_000, sublinear_tf=True)
    X = vec.fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute")
    nn.fit(X)
    pairs = []
    for start in range(0, n, 2000):
        dist, idx = nn.kneighbors(X[start:start + 2000])
        for r in range(dist.shape[0]):
            i = start + r
            j = int(idx[r][1]) if int(idx[r][0]) == i else int(idx[r][0])
            sim = 1.0 - float(dist[r][1] if int(idx[r][0]) == i else dist[r][0])
            if sim >= args.near and i < j:
                pairs.append((i, j, sim))

    cross = [(i, j, s) for i, j, s in pairs if split_of[i] != split_of[j]]
    cross_kinds = Counter(tuple(sorted((split_of[i], split_of[j])))
                          for i, j, _ in cross)
    cross_exact = [(g, {split_of[i] for i in g}) for g in exact_groups.values()
                   if len({split_of[i] for i in g}) > 1]
    label_mismatch = sum(1 for i, j, _ in pairs if labels[i] != labels[j])

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "dedup_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "sim", "split_i", "split_j", "label_i", "label_j",
                    "text_i", "text_j"])
        for i, j, s in sorted(pairs, key=lambda p: -p[2]):
            w.writerow([i, j, f"{s:.4f}", split_of[i], split_of[j],
                        labels[i], labels[j],
                        full["text_"][i][:300], full["text_"][j][:300]])

    lines = [
        "# Dedup audit (P0-1)", "",
        f"- Rows: {n}",
        f"- Exact duplicate groups (normalized text): {len(exact_groups)} "
        f"({exact_rows} rows involved)",
        f"- Exact-dup groups spanning multiple splits (LEAKAGE): {len(cross_exact)}",
        f"- Near-duplicate pairs (cosine >= {args.near}): {len(pairs)}",
        f"- Near-dup pairs ACROSS splits (LEAKAGE): {len(cross)} "
        f"{dict(cross_kinds)}",
        f"- Near-dup pairs with CONFLICTING labels (OR vs CG): {label_mismatch}",
        "",
        "Pairs detail: results/dedup_pairs.csv (sorted by similarity).",
        "",
        "## Verdict guidance",
        "- Cross-split exact dups or many cross-split near-dups => re-split with",
        "  group-aware splitting before Phase 1 and report both numbers.",
        "- OR-vs-CG near-dup pairs are expected by construction (the CG reviews",
        "  were generated FROM real ones) — report them, they are not leakage.",
    ]
    (RESULTS / "dedup_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
