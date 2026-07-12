"""Build leave-one-generator-out (LOGO) pooled training sets.

For each held-out generator G, pool the TRAIN splits of the other three
generators' datasets into one balanced, approximately budget-matched set:
  - OR (human) half: deduped union, excluding every human text in G's held-out
    test split. The same humans appear in all generator datasets under different
    row orders, so this exclusion is required for an honest LOGO FPR.
  - CG (fake) half: equal share from each of the 3 generators, subsampled so
    total fakes == total humans (balanced) and the dataset size matches a
    single-generator set — isolating the effect of generator DIVERSITY from
    the effect of simply having more data.

Output: dataset/pooled_wo_{gpt2|lfm|deepseek|glm}.csv
(train a detector on it with grids/logo.txt; evaluate with cross_gen_matrix.py)
"""
import csv
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SEED = 1998
GENS = {
    "gpt2":     ROOT / "dataset" / "modern_reviews_gpt2.csv",
    "lfm":      ROOT / "dataset" / "modern_reviews_lfm.csv",
    "deepseek": ROOT / "dataset" / "modern_reviews_deepseek.csv",
    "glm":      ROOT / "dataset" / "modern_reviews_glm.csv",
}
COLS = ["category", "rating", "label", "text_"]


def train_split(csv_path):
    """Exact same train split as train_run.py / cross_gen_matrix.py."""
    ds = load_dataset("csv", data_files=str(csv_path))["train"]
    return ds.train_test_split(test_size=0.2, seed=SPLIT_SEED)["train"]


def test_or_texts(csv_path):
    """Human texts in the exact 14% test split used by cross_gen_matrix.py."""
    ds = load_dataset("csv", data_files=str(csv_path))["train"]
    outer = ds.train_test_split(test_size=0.2, seed=SPLIT_SEED)
    test = outer["test"].train_test_split(test_size=0.7, seed=SPLIT_SEED)["test"]
    return {t.strip().casefold() for t, label in zip(test["text_"], test["label"])
            if label == "OR"}


def main():
    splits = {name: train_split(p) for name, p in GENS.items()}
    for holdout in GENS:
        sources = [n for n in GENS if n != holdout]
        heldout_or = test_or_texts(GENS[holdout])
        # OR half: deduped union of the three train splits' humans
        or_rows, seen = [], set()
        cg_by_src = {n: [] for n in sources}
        for n in sources:
            tr = splits[n]
            for c, r, l, t in zip(tr["category"], tr["rating"], tr["label"], tr["text_"]):
                if l == "OR":
                    k = t.strip().casefold()
                    if k not in heldout_or and k not in seen:
                        seen.add(k)
                        or_rows.append({"category": c, "rating": r, "label": l, "text_": t})
                else:
                    cg_by_src[n].append({"category": c, "rating": r, "label": l, "text_": t})
        # CG half: equal share per generator, total == len(or_rows)  (balanced,
        # budget-matched to a single-generator training set)
        share = len(or_rows) // len(sources)
        rng = random.Random(SPLIT_SEED)
        cg_rows = []
        for n in sources:
            pool = cg_by_src[n]
            rng.shuffle(pool)
            cg_rows.extend(pool[:share])
        rows = or_rows[:len(cg_rows)] + cg_rows
        rng.shuffle(rows)
        out = ROOT / "dataset" / f"pooled_wo_{holdout}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader(); w.writerows(rows)
        n_or = sum(1 for r in rows if r["label"] == "OR")
        overlap = sum(1 for r in rows
                      if r["label"] == "OR"
                      and r["text_"].strip().casefold() in heldout_or)
        if overlap:
            raise RuntimeError(
                f"pooled_wo_{holdout} leaks {overlap} held-out human texts")
        print(f"pooled_wo_{holdout}: {len(rows)} rows  (OR={n_or}, CG={len(rows)-n_or}, "
              f"{share}/generator from {sources}, held-out OR overlap={overlap})")


if __name__ == "__main__":
    main()
