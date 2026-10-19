"""Normalize independently collected human-vs-LLM review corpora for zero-shot
evaluation of the source-grouped Rs-QLoRA adapters.

Raw files (downloaded at pinned revisions, see dataset/external_benchmark/sources.json):
  raw/maideup/all_data.csv                  MAiDE-up (Ignat et al., 2025), MIT
  raw/detectairev/Testing_generated_review.csv  DetectAIRev (Agrahari et al., 2025)
  raw/mage/test.csv                         MAGE test split (Li et al., 2024), Apache-2.0

Outputs dataset/external_benchmark/{maideup_en,detectairev,mage_yelp}.csv with
columns row_id, dataset, domain, generator, label (OR/CG), text_, plus
manifest.json recording every filtering count.

Both classes go through the same cleaning: HTML tags removed, leading
prompt-template labels ("Fake Review:", "Review:", "1.") removed, then the
training-time normalize_text() used to build the source-grouped datasets.
Texts with fewer than three words, exact duplicates, texts present in both
classes, and texts present verbatim in any source-grouped training file are
dropped.

  python runs/prepare_external_benchmark.py --root .
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with python -I
from build_modern_dataset import normalize_text  # noqa: E402

SEED = 1998
HTML = re.compile(r"<[^>]+>")
# Template labels that some generators echo before the review body. They name
# the class outright ("Fake Review:") or are list enumerators, so they are
# removed from both classes before scoring.
PREFIX = re.compile(
    r"^\s*(?:\d+\s*[.)]\s*)?(?:\*\*)?\s*"
    r"(?:(?:fake|true|real|genuine|authentic|generated|sample|positive|negative|"
    r"neutral|my|customer|product|hotel|movie)\s+)?review\s*\d*\s*:\s*(?:\*\*)?\s*",
    re.I,
)
ENUM = re.compile(r"^\s*\d+\s*\.\s+(?=\S)")


def clean(text: str) -> tuple[str, bool]:
    raw = text
    text = HTML.sub(" ", text)
    text = PREFIX.sub("", text, count=1)
    text = ENUM.sub("", text, count=1)
    stripped = text.strip() != HTML.sub(" ", raw).strip()
    return normalize_text(text), stripped


def load_training_texts(root: Path) -> set[str]:
    texts: set[str] = set()
    for path in sorted((root / "dataset" / "source_grouped").glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                texts.add(row["text_"].lower())
    return texts


def finalize(name: str, rows: list[dict], train_texts: set[str], manifest: dict) -> list[dict]:
    counts: dict[str, Counter] = {k: Counter() for k in ("input", "prefix_stripped",
                                                           "lt3_words", "duplicate",
                                                           "cross_label", "in_training",
                                                           "kept")}
    for row in rows:
        counts["input"][row["label"]] += 1
        if row.pop("_stripped"):
            counts["prefix_stripped"][row["label"]] += 1
    long_enough = []
    for row in rows:
        if len(row["text_"].split()) >= 3:
            long_enough.append(row)
        else:
            counts["lt3_words"][row["label"]] += 1
    rows = long_enough
    by_text: dict[str, set[str]] = {}
    for row in rows:
        by_text.setdefault(row["text_"].lower(), set()).add(row["label"])
    seen: set[str] = set()
    kept = []
    for row in rows:
        key = row["text_"].lower()
        if len(by_text[key]) > 1:
            counts["cross_label"][row["label"]] += 1
            continue
        if key in seen:
            counts["duplicate"][row["label"]] += 1
            continue
        seen.add(key)
        if key in train_texts:
            counts["in_training"][row["label"]] += 1
            continue
        kept.append(row)
        counts["kept"][row["label"]] += 1
    manifest[name] = {"filter_counts": {k: dict(v) for k, v in counts.items()}}
    return kept


def prepare_maideup(raw: Path, train_texts: set[str], manifest: dict) -> list[dict]:
    rows = []
    with (raw / "maideup" / "all_data.csv").open(encoding="utf-8-sig", newline="") as handle:
        for rec in csv.DictReader(handle):
            if rec["Review_Language"] != "English":
                continue
            parts = []
            for field, missing in (("Upside_Review", "na_up_review"),
                                   ("Downside_Review", "na_down_review")):
                value = rec[field].strip()
                if rec[missing] != "True" and value and value.lower() != "nan":
                    parts.append(value)
            text, stripped = clean(" ".join(parts))
            rows.append({
                "source_row": rec[""],
                "dataset": "maideup_en",
                "domain": "hotel (Booking.com)",
                "generator": "gpt-4" if rec["source"] == "1" else "human",
                "label": "CG" if rec["source"] == "1" else "OR",
                "sentiment": rec["Sentiment"],
                "city": rec["City Name"],
                "text_": text,
                "_stripped": stripped,
            })
    rows = finalize("maideup_en", rows, train_texts, manifest)
    manifest["maideup_en"]["selection"] = "all English rows; text = Upside_Review + ' ' + Downside_Review"
    return rows


def prepare_detectairev(raw: Path, train_texts: set[str], manifest: dict, per_class: int) -> list[dict]:
    domains = {"E-Commerce": "e-commerce clothing", "Movie": "movie (IMDb)", "Tourist": "tourist attraction"}
    pool: dict[tuple[str, str], list[dict]] = {}
    with (raw / "detectairev" / "Testing_generated_review.csv").open(encoding="utf-8-sig", newline="") as handle:
        for index, rec in enumerate(csv.DictReader(handle)):
            domain = rec["domain"].strip().title()
            if domain not in domains:
                continue  # Hotel human reviews are a stop-word-stripped, lowercased release
            model = rec["model"].strip().lower()
            label = "OR" if model == "human" else "CG"
            if (label == "OR") != (rec["label"] == "0"):
                raise RuntimeError(f"label/model mismatch at row {index}")
            text, stripped = clean(rec["review_text"])
            pool.setdefault((domain, model), []).append({
                "source_row": str(index),
                "dataset": "detectairev",
                "domain": domains[domain],
                "generator": model,
                "label": label,
                "sentiment": "",
                "city": "",
                "text_": text,
                "_stripped": stripped,
            })
    # Clean and deduplicate the whole pool first, then sample a balanced test set
    # per domain: per_class human reviews and per_class generated reviews split
    # equally among that domain's generators.
    flat = [row for rows in pool.values() for row in rows]
    flat = finalize("detectairev", flat, train_texts, manifest)
    cells: dict[tuple[str, str], list[dict]] = {}
    for row in flat:
        cells.setdefault((row["domain"], row["generator"]), []).append(row)
    rng = random.Random(SEED)
    selected = []
    sample_counts = {}
    for domain in domains.values():
        gens = sorted(g for (d, g) in cells if d == domain and g != "human")
        quota = {"human": per_class}
        base, extra = divmod(per_class, len(gens))
        for i, gen in enumerate(gens):
            quota[gen] = base + (1 if i < extra else 0)
        for gen, n in quota.items():
            cell = sorted(cells[(domain, gen)], key=lambda r: int(r["source_row"]))
            if len(cell) < n:
                raise RuntimeError(f"{domain}/{gen}: only {len(cell)} rows for quota {n}")
            picked = rng.sample(cell, n)
            selected.extend(sorted(picked, key=lambda r: int(r["source_row"])))
            sample_counts[f"{domain}|{gen}"] = n
    manifest["detectairev"]["selection"] = {
        "excluded_domains": {"Hotel": "human TripAdvisor text is lowercased with stop words removed "
                                      "(stop-word rate 0.009 vs 0.30-0.37 for generated text)"},
        "per_domain_per_class": per_class,
        "sample_seed": SEED,
        "sampled": sample_counts,
    }
    return selected


def prepare_mage(raw: Path, train_texts: set[str], manifest: dict) -> list[dict]:
    rows = []
    with (raw / "mage" / "test.csv").open(encoding="utf-8-sig", newline="") as handle:
        for index, rec in enumerate(csv.DictReader(handle)):
            src = rec["src"]
            if not src.startswith("yelp_"):
                continue
            human = src == "yelp_human"
            if human != (rec["label"] == "1"):  # MAGE: 1 = human, 0 = machine
                raise RuntimeError(f"MAGE label/src mismatch at row {index}")
            text, stripped = clean(rec["text"])
            rows.append({
                "source_row": str(index),
                "dataset": "mage_yelp",
                "domain": "restaurant/business (Yelp)",
                "generator": "human" if human else src.removeprefix("yelp_machine_continuation_"),
                "label": "OR" if human else "CG",
                "sentiment": "",
                "city": "",
                "text_": text,
                "_stripped": stripped,
            })
    rows = finalize("mage_yelp", rows, train_texts, manifest)
    manifest["mage_yelp"]["selection"] = "all Yelp rows of the MAGE test split (continuation prompts)"
    return rows


def write(path: Path, rows: list[dict]) -> str:
    fields = ["row_id", "dataset", "domain", "generator", "label", "sentiment", "city",
              "source_row", "text_"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"row_id": i, **row})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--detectairev-per-class", type=int, default=1500)
    args = ap.parse_args()
    root = Path(args.root)
    out = root / "dataset" / "external_benchmark"
    raw = out / "raw"
    train_texts = load_training_texts(root)
    manifest: dict = {"normalization": "HTML tags and leading template labels removed, then "
                                       "build_modern_dataset.normalize_text (both classes)",
                      "training_texts_checked": len(train_texts)}
    datasets = {
        "maideup_en": prepare_maideup(raw, train_texts, manifest),
        "detectairev": prepare_detectairev(raw, train_texts, manifest, args.detectairev_per_class),
        "mage_yelp": prepare_mage(raw, train_texts, manifest),
    }
    for name, rows in datasets.items():
        manifest[name]["sha256"] = write(out / f"{name}.csv", rows)
        manifest[name]["n"] = dict(Counter(r["label"] for r in rows))
        manifest[name]["generators"] = dict(Counter(r["generator"] for r in rows))
        manifest[name]["mean_words"] = {
            lab: round(sum(len(r["text_"].split()) for r in rows if r["label"] == lab)
                       / max(1, sum(r["label"] == lab for r in rows)), 3)
            for lab in ("OR", "CG")
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k]["n"] for k in datasets}, indent=2))


if __name__ == "__main__":
    main()
