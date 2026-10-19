"""Evaluate one cloud shard and retain strict source-grouped probabilities.

Directional adapters are applied to all three contemporary test partitions.
Each LOGO adapter is applied only to its genuinely held-out generator.  Human
test predictions are computed once per adapter because the source-grouped OR
test partition is identical across generator files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from source_grouped_common import (
    DIRECTIONAL_TAGS,
    GENERATORS,
    LOGO_TAGS,
    ROOT,
    assert_no_source_leakage,
    data_path,
    find_completed_run,
    grid_lines,
    parse_options,
    read_csv,
    sha256,
    target_dataset,
    test_rows,
)


def split_target_rows(generator: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = test_rows(generator)
    humans = sorted(
        (row for row in rows if row["label"] == "OR"),
        key=lambda row: (int(row["source_group"]), int(row["source_id"])),
    )
    generated = sorted(
        (row for row in rows if row["label"] == "CG"),
        key=lambda row: (int(row["source_group"]), int(row["source_id"])),
    )
    return humans, generated


def verify_common_humans() -> list[dict[str, str]]:
    reference, _ = split_target_rows(GENERATORS[0])
    reference_keys = [
        (row["source_id"], row["source_group"], row["category"], row["rating"], row["text_"])
        for row in reference
    ]
    for generator in GENERATORS[1:]:
        humans, _ = split_target_rows(generator)
        keys = [
            (row["source_id"], row["source_group"], row["category"], row["rating"], row["text_"])
            for row in humans
        ]
        if keys != reference_keys:
            raise RuntimeError(f"human test partition differs for {generator}")
    return reference


def metrics(rows: list[dict[str, str]], probabilities: list[float]) -> dict[str, float | int]:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels = np.asarray([1 if row["label"] == "CG" else 0 for row in rows])
    probs = np.asarray(probabilities)
    predictions = (probs >= 0.5).astype(int)
    humans = labels == 0
    generated = labels == 1
    return {
        "n": int(len(labels)),
        "n_OR": int(humans.sum()),
        "n_CG": int(generated.sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_CG": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall_CG": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "f1_OR": float(f1_score(labels, predictions, pos_label=0, zero_division=0)),
        "f1_CG": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "fpr_OR": float(predictions[humans].mean()),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
    }


def write_probabilities(
    path: Path,
    rows: list[dict[str, str]],
    probabilities: list[float],
) -> None:
    columns = [
        "row",
        "true",
        "p_CG",
        "category",
        "rating",
        "text",
        "source_id",
        "source_group",
        "generator",
        "split",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, (row, probability) in enumerate(zip(rows, probabilities, strict=True)):
            writer.writerow(
                {
                    "row": index,
                    "true": row["label"],
                    "p_CG": f"{probability:.9f}",
                    "category": row["category"],
                    "rating": row["rating"],
                    "text": row["text_"],
                    "source_id": row["source_id"],
                    "source_group": row["source_group"],
                    "generator": row["generator"],
                    "split": row["split"],
                }
            )


def evaluate_line(line: str, human_rows: list[dict[str, str]], batch_size: int) -> Path:
    import pyarrow  # noqa: F401  must precede unsloth/torch on Windows
    import torch
    from unsloth import FastModel
    from transformers import AutoModelForSequenceClassification

    options = parse_options(line)
    tag = str(options["tag"])
    if tag in DIRECTIONAL_TAGS:
        protocol = "directional"
        train_generator = DIRECTIONAL_TAGS[tag]
        targets = list(GENERATORS)
        held_out = None
    elif tag in LOGO_TAGS:
        protocol = "logo"
        train_generator = None
        held_out = LOGO_TAGS[tag]
        targets = [held_out]
    elif tag == "sgfr":
        protocol = "fixed_scale_rank"
        train_generator = "deepseek"
        held_out = None
        targets = ["deepseek"]
    else:
        raise RuntimeError(f"unsupported source-grouped tag: {tag}")

    run_dir = find_completed_run(line)
    adapter = run_dir / "adapter"
    training_data = data_path(options)
    leakage = {
        target: assert_no_source_leakage(training_data, target) for target in targets
    }
    if held_out is not None:
        training_generators = {
            row["generator"]
            for row in read_csv(training_data)
            if row["split"] == "train" and row["label"] == "CG"
        }
        if held_out in training_generators:
            raise RuntimeError(f"LOGO training data contains held-out generator {held_out}")

    model, tokenizer = FastModel.from_pretrained(
        model_name=str(adapter),
        auto_model=AutoModelForSequenceClassification,
        max_seq_length=512,
        dtype=None,
        num_labels=2,
        load_in_4bit=True,
        full_finetuning=False,
    )
    FastModel.for_inference(model)
    model.eval()
    device = next(model.parameters()).device

    def predict(rows: list[dict[str, str]]) -> list[float]:
        values: list[float] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer(
                [row["text_"] for row in batch],
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(**encoded).logits.float(), dim=-1)[:, 1]
            values.extend(float(value) for value in probs.cpu().tolist())
        return values

    human_probabilities = predict(human_rows)
    evaluations = {}
    for target in targets:
        _, generated_rows = split_target_rows(target)
        generated_probabilities = predict(generated_rows)
        combined_rows = human_rows + generated_rows
        combined_probabilities = human_probabilities + generated_probabilities
        probability_path = run_dir / f"source_grouped_eval_{target}.csv"
        write_probabilities(probability_path, combined_rows, combined_probabilities)
        evaluations[target] = {
            "dataset": target_dataset(target).relative_to(ROOT).as_posix(),
            "dataset_sha256": sha256(target_dataset(target)),
            "probabilities": probability_path.name,
            "source_isolation": leakage[target],
            "metrics": metrics(combined_rows, combined_probabilities),
        }

    del model
    torch.cuda.empty_cache()
    summary = {
        "schema_version": 1,
        "protocol": protocol,
        "run_id": run_dir.name,
        "tag": tag,
        "seed": int(str(options["seed"])),
        "train_generator": train_generator,
        "held_out_generator": held_out,
        "rank": int(str(options["rank"])),
        "alpha": int(str(options.get("alpha", 2 * int(str(options["rank"]))))),
        "effective_rs_scale": (
            int(str(options.get("alpha", 2 * int(str(options["rank"])))))
            / (int(str(options["rank"])) ** 0.5)
        ),
        "training_data": training_data.relative_to(ROOT).as_posix(),
        "training_data_sha256": sha256(training_data),
        "threshold": 0.5,
        "targets": evaluations,
    }
    (run_dir / "source_grouped_eval.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"evaluated {run_dir.name}: {', '.join(targets)}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    grid = Path(args.grid)
    if not grid.is_absolute():
        grid = ROOT / grid
    human_rows = verify_common_humans()
    for line in grid_lines(grid):
        evaluate_line(line, human_rows, args.batch_size)


if __name__ == "__main__":
    main()
