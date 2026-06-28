"""Pull a HF dataset (the modern fake-review corpus) and write it back to one flat
CSV (category,rating,label,text_) so train_run.py --data / baseline scripts can use
it. train_run re-splits with seed 1998 (the same split used to build the HF splits),
so the protocol is preserved.

  python runs/fetch_hf_dataset.py --repo Flowerly/modern-fake-reviews \
         --out dataset/modern_reviews_deepseek.csv
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_dataset, concatenate_datasets
    dd = load_dataset(args.repo)
    full = concatenate_datasets([dd[s] for s in dd])
    cols = ["category", "rating", "label", "text_"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in full:
            w.writerow({c: r[c] for c in cols})
    print(f"wrote {len(full)} rows -> {out}")


if __name__ == "__main__":
    main()
