"""Audit source-review overlap in the modern cross-generator experiments.

Each modern generated review retains ``src_idx`` in its raw generation file,
but the assembled training CSV intentionally has the common four-column schema.
This script joins the normalized generated text back to ``src_idx`` and uses the
saved validation/test probability files to recover each model's frozen split.
It reports how often a target generator's test prompt source occurred in another
generator's training split.  This distinguishes generator holdout from the
stronger (not implemented here) joint generator-and-prompt-source holdout.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from build_modern_dataset import normalize_text


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "LFM": {
        "raw": ROOT / "dataset/sota_fakes.csv",
        "final": ROOT / "dataset/modern_reviews_lfm.csv",
        "run": ROOT / "results/xl_rs_r64_s7_07120532",
    },
    "DeepSeek": {
        "raw": ROOT / "dataset/sota_fakes_deepseek.csv",
        "final": ROOT / "dataset/modern_reviews_deepseek.csv",
        "run": ROOT / "results/xd_rs_r64_s7_07120556",
    },
    "GLM": {
        "raw": ROOT / "dataset/sota_fakes_glm.csv",
        "final": ROOT / "dataset/modern_reviews_glm.csv",
        "run": ROOT / "results/xglm_rs_r64_s7_07120621",
    },
}


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def cg_texts(path: Path, text_column: str) -> set[str]:
    return {
        row[text_column]
        for row in rows(path)
        if row.get("true", row.get("label")) == "CG"
    }


def recover_split(config: dict[str, Path]) -> tuple[set[int], set[int]]:
    source_by_text = {
        normalize_text(row["text_"]): int(row["src_idx"])
        for row in rows(config["raw"])
        if row.get("label", "CG") == "CG"
    }
    held_out_texts = cg_texts(config["run"] / "val_probs.csv", "text")
    held_out_texts |= cg_texts(config["run"] / "test_probs.csv", "text")

    train_sources = {
        source_by_text[row["text_"]]
        for row in rows(config["final"])
        if row["label"] == "CG"
        and row["text_"] not in held_out_texts
        and row["text_"] in source_by_text
    }
    test_sources = {
        source_by_text[row["text"]]
        for row in rows(config["run"] / "test_probs.csv")
        if row["true"] == "CG" and row["text"] in source_by_text
    }
    return train_sources, test_sources


def main() -> None:
    splits = {name: recover_split(config) for name, config in CONFIG.items()}
    pairwise = []
    pooled = []

    for target, (_, target_test) in splits.items():
        other_train = []
        for train, (train_sources, _) in splits.items():
            if train == target:
                continue
            overlap = len(train_sources & target_test)
            pairwise.append(
                {
                    "train_generator": train,
                    "test_generator": target,
                    "overlap": overlap,
                    "test_sources": len(target_test),
                    "fraction": overlap / len(target_test),
                }
            )
            other_train.append(train_sources)
        union = set().union(*other_train)
        overlap = len(union & target_test)
        pooled.append(
            {
                "test_generator": target,
                "overlap": overlap,
                "test_sources": len(target_test),
                "fraction": overlap / len(target_test),
            }
        )

    result = {
        "scope": "modern generators only; GPT-2 raw data has no source index",
        "interpretation": (
            "Generator identity is held out, but prompt-source identity is not. "
            "The experiment is not a joint generator-and-source OOD test."
        ),
        "split_sizes": {
            name: {"train_sources": len(train), "test_sources": len(test)}
            for name, (train, test) in splits.items()
        },
        "pairwise": pairwise,
        "pooled_other_generators": pooled,
    }
    out = ROOT / "results/source_overlap.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
