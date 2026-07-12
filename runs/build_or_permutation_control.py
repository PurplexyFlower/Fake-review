"""Build the human-only label-permutation negative-control dataset.

A literal all-OR dataset cannot exercise the binary OR-vs-CG training pipeline:
the classifier sees only one class and metrics such as ROC-AUC are undefined.
Instead, this control keeps only genuine human (OR) reviews, removes duplicate
texts, and assigns deterministic balanced pseudo-labels.  Because the labels are
independent of the text, a sound end-to-end pipeline should score at chance.

The OR/CG strings in the output are *pseudo-labels used only for harness
compatibility*; no review in this dataset is computer-generated.

  python runs/build_or_permutation_control.py
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from build_modern_dataset import normalize_text

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset" / "fake reviews dataset.csv"
DEFAULT_SEED = 1998
OUT = ROOT / "dataset" / f"control_or_permutation_s{DEFAULT_SEED}.csv"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--label-seed", type=int, default=DEFAULT_SEED)
    return ap.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)
    out = Path(args.out)

    humans, seen = [], set()
    n_or = n_duplicates = 0
    with source.open(encoding="utf-8", newline="") as f:
        for source_row, row in enumerate(csv.DictReader(f)):
            if row["label"] != "OR":
                continue
            n_or += 1
            text = normalize_text(row["text_"])
            key = text.casefold()
            if not text or key in seen:
                n_duplicates += 1
                continue
            seen.add(key)
            humans.append({
                "category": row["category"],
                "rating": row["rating"],
                "label": "",  # assigned below
                "text_": text,
                "source_row": source_row,
            })

    rng = random.Random(args.label_seed)
    dropped_for_balance = None
    if len(humans) % 2:
        dropped_for_balance = humans.pop(rng.randrange(len(humans)))

    labels = ["OR"] * (len(humans) // 2) + ["CG"] * (len(humans) // 2)
    rng.shuffle(labels)
    for row, label in zip(humans, labels):
        row["label"] = label
    rng.shuffle(humans)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["category", "rating", "label", "text_"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in humans)

    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    output_sha = hashlib.sha256(out.read_bytes()).hexdigest()
    metadata = {
        "control": "human-only balanced label permutation",
        "label_semantics": "OR/CG are random pseudo-labels; every text is human OR",
        "source": str(source),
        "source_sha256": source_sha,
        "output": str(out),
        "output_sha256": output_sha,
        "label_seed": args.label_seed,
        "source_or_rows": n_or,
        "duplicate_or_empty_texts_removed": n_duplicates,
        "source_row_dropped_for_exact_balance": (
            dropped_for_balance["source_row"] if dropped_for_balance else None),
        "output_rows": len(humans),
        "pseudo_label_counts": dict(Counter(row["label"] for row in humans)),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"human OR rows: {n_or}")
    print(f"deduplicated/empty removed: {n_duplicates}")
    print(f"wrote {len(humans)} rows -> {out}")
    print("pseudo-labels:", metadata["pseudo_label_counts"])
    print(f"metadata -> {meta_path}")


if __name__ == "__main__":
    main()
