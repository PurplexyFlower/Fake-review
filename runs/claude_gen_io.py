"""I/O helper for the `gen-fake-reviews` Claude Code skill — keeps the harness out
of fragile CSV/normalisation handling. Two subcommands:

  next  N [--out FILE]   -> prints a JSON worklist of the next N un-generated real
                            reviews to ground on: [{src_idx, category, rating,
                            target_len, real_text}, ...]  (skips src_idx already in OUT)
  append --in JSON [--out FILE] [--model NAME]
                         -> reads [{src_idx, text}, ...], normalises text (same
                            ASCII rule as the dataset), looks up category/rating,
                            appends rows to OUT (cols: src_idx,category,rating,
                            label,text_,gen_model). Resumable + dedup-safe.

Default OUT = dataset/sota_fakes_claude.csv (build_modern_dataset.py --fakes it to
add a Claude-generated generator to the cross-gen matrix).
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_modern_dataset import normalize_text  # same normalisation as the datasets

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "fake reviews dataset.csv"
DEFAULT_OUT = ROOT / "dataset" / "sota_fakes_claude.csv"
COLS = ["src_idx", "category", "rating", "label", "text_", "gen_model"]


def real_or():
    rows = []
    for i, r in enumerate(csv.DictReader(open(DATA, encoding="utf-8"))):
        if r["label"] == "OR":
            rows.append((i, r["category"], r["rating"], r["text_"]))
    return rows


def done(out):
    if not out.exists():
        return set()
    return {int(r["src_idx"]) for r in csv.DictReader(open(out, encoding="utf-8"))}


def cmd_next(args):
    out = Path(args.out)
    have = done(out)
    # deterministic shuffle so any partial run is category-balanced (the dataset
    # is grouped by category; sequential order would do all of one category first)
    rows = real_or()
    random.Random(1998).shuffle(rows)
    work = []
    for idx, cat, rating, text in rows:
        if idx in have:
            continue
        work.append({"src_idx": idx, "category": cat, "rating": rating,
                     "target_len": min(len(text.split()), 120),
                     "real_text": text[:400]})
        if len(work) >= args.n:
            break
    print(json.dumps(work, ensure_ascii=False, indent=2))


def cmd_append(args):
    out = Path(args.out)
    items = json.loads(Path(args.in_).read_text(encoding="utf-8"))
    by_idx = {i: (c, r) for i, c, r, _ in real_or()}
    have = done(out)
    new_file = not out.exists()
    n = 0
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new_file:
            w.writeheader()
        for it in items:
            idx = int(it["src_idx"])
            txt = normalize_text(it["text"])
            if idx in have or idx not in by_idx or len(txt.split()) < 3:
                continue
            cat, rating = by_idx[idx]
            w.writerow({"src_idx": idx, "category": cat, "rating": rating,
                        "label": "CG", "text_": txt, "gen_model": args.model})
            have.add(idx); n += 1
    print(f"appended {n} -> {out}  (total now {len(done(out))})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pn = sub.add_parser("next"); pn.add_argument("n", type=int)
    pn.add_argument("--out", default=str(DEFAULT_OUT)); pn.set_defaults(fn=cmd_next)
    pa = sub.add_parser("append"); pa.add_argument("--in", dest="in_", required=True)
    pa.add_argument("--out", default=str(DEFAULT_OUT))
    pa.add_argument("--model", default="claude-code"); pa.set_defaults(fn=cmd_append)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
