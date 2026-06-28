"""Build results/test_exclusions.csv: test-set original-row indices that have an
exact or near duplicate in train/val (from dedup_pairs.csv). Analyses report
metrics with and without these rows."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pairs = ROOT / "results" / "dedup_pairs.csv"
out = ROOT / "results" / "test_exclusions.csv"

excl = {}
with open(pairs, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        si, sj = row["split_i"], row["split_j"]
        if {si, sj} == {"test"} or si == sj:
            continue
        if si == "test":
            excl.setdefault(int(row["i"]), (row["sim"], sj))
        if sj == "test":
            excl.setdefault(int(row["j"]), (row["sim"], si))

with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["orig_idx", "sim", "leaks_from"])
    for idx in sorted(excl):
        w.writerow([idx, *excl[idx]])
print(f"{len(excl)} test rows flagged -> {out}")
