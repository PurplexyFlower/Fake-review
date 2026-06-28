"""Assemble the NEW modern-fake-review dataset for the redirected study:
   real human reviews (OR, from the Kaggle set)  vs  modern AI fakes (CG, from
   runs/gen_fakes_sota.py with a SOTA model).

Output: dataset/modern_reviews_dataset.csv  (category, rating, label, text_)
— same schema as the original, so `train_run.py --data` trains on it unchanged.
"label" stays OR/CG (OR=0 human, CG=1 modern-AI-fake) for harness compatibility.

By default balanced 1:1; if fewer fakes were generated than humans, the human
side is subsampled (seed 1998) to match, keeping category proportions.

  python runs/build_modern_dataset.py
"""
import argparse
import csv
import random
import re
import statistics as st
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "dataset" / "fake reviews dataset.csv"
FAKES = ROOT / "dataset" / "sota_fakes.csv"
OUT = ROOT / "dataset" / "modern_reviews_dataset.csv"

# The real reviews are 100% ASCII; the LLM emits smart quotes/em-dashes in ~64%
# of fakes. Left alone, that punctuation is a trivial encoding tell (em-dash =>
# fake) rather than a linguistic one. Normalise BOTH halves to the same ASCII
# convention so the detector must learn content, not Unicode. (No-op on the
# already-ASCII human side.)
# Map smart quotes -> straight; em/en-dash -> comma (NOT " - ", which itself
# becomes a glaring fake-only token); ellipsis char -> "...". The goal is that
# the punctuation distribution of the (Unicode-heavy) LLM output blends into the
# (ASCII) human convention rather than acquiring a new signature.
_SMART = {0x2018: "'", 0x2019: "'", 0x201a: "'", 0x201b: "'",
          0x201c: '"', 0x201d: '"', 0x201e: '"', 0x2032: "'", 0x2033: '"',
          0x2013: ", ", 0x2014: ", ", 0x2026: "...", 0x00a0: " "}


def normalize_text(t):
    t = t.translate(_SMART)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s*,\s*,", ", ", t)          # collapse doubled commas
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)    # no space before punctuation
    t = re.sub(r",(\s*[.!?])", r"\1", t)      # drop comma before sentence end
    return t.strip().strip(",").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-balance", action="store_true",
                    help="keep ALL humans even if more than fakes (imbalanced)")
    ap.add_argument("--fakes", default=str(FAKES),
                    help="generated-fakes CSV (e.g. dataset/sota_fakes_deepseek.csv)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_path = Path(args.out)

    humans = [{"category": r["category"], "rating": r["rating"], "label": "OR",
               "text_": normalize_text(r["text_"])}
              for r in csv.DictReader(open(ORIG, encoding="utf-8"))
              if r["label"] == "OR"]
    human_texts = {h["text_"].lower() for h in humans}

    seen, fakes = set(), []
    for r in csv.DictReader(open(args.fakes, encoding="utf-8")):
        if r.get("label", "CG") != "CG":   # fakes file may carry both classes
            continue                       # (e.g. the original Kaggle set -> GPT-2 CG only)
        txt = normalize_text(r["text_"])
        key = txt.lower()
        if txt and key not in seen and key not in human_texts:  # dedup within + cross-class
            seen.add(key)
            fakes.append({"category": r["category"], "rating": r["rating"],
                          "label": "CG", "text_": txt})

    print(f"humans (OR) = {len(humans)}   modern fakes (CG, deduped) = {len(fakes)}")

    if not args.no_balance and len(humans) > len(fakes):
        rng = random.Random(1998)
        rng.shuffle(humans)
        humans = humans[:len(fakes)]
        print(f"balanced: subsampled humans -> {len(humans)}")

    rows = humans + fakes
    random.Random(1998).shuffle(rows)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["category", "rating", "label", "text_"])
        w.writeheader(); w.writerows(rows)

    print(f"\nwrote {len(rows)} rows -> {out_path}")
    print("labels:", dict(Counter(r["label"] for r in rows)))
    for lab in ("OR", "CG"):
        wl = [len(r["text_"].split()) for r in rows if r["label"] == lab]
        if wl:
            print(f"  {lab}: n={len(wl)}  word-len mean={st.mean(wl):.1f} "
                  f"median={sorted(wl)[len(wl)//2]}  cats={len(set(r['category'] for r in rows if r['label']==lab))}")


if __name__ == "__main__":
    main()
