"""Build leave-one-generator-out (LOGO) pooled training sets.

For each held-out generator G, pool the TRAIN splits of the other three
generators' datasets into one balanced, budget-matched training set:
  - OR (human) half: deduped union (the same humans appear in all datasets)
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


def main():
    splits = {name: train_split(p) for name, p in GENS.items()}
    for holdout in GENS:
        sources = [n for n in GENS if n != holdout]
        # OR half: deduped union of the three train splits' humans
        or_rows, seen = [], set()
        cg_by_src = {n: [] for n in sources}
        for n in sources:
            tr = splits[n]
            for c, r, l, t in zip(tr["category"], tr["rating"], tr["label"], tr["text_"]):
                if l == "OR":
                    k = t.lower()
                    if k not in seen:
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
        print(f"pooled_wo_{holdout}: {len(rows)} rows  (OR={n_or}, CG={len(rows)-n_or}, "
              f"{share}/generator from {sources})")


if __name__ == "__main__":
    main()
