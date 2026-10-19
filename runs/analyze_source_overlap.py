"""Audit source support in the modern cross-generator experiments.

The modern CG rows retain their grounding-review ``src_idx`` only in the raw
generation files.  Recreate build_modern_dataset.py's *first-retained* dedup
mapping, recover the exact saved train splits by multiset subtraction, and
measure two distinct kinds of support for each target CG test row:

1. another trained CG row has the same ``src_idx``; and
2. the exact normalized OR grounding review itself occurs in training.

The LOGO datasets are split again by train_run.py.  Therefore, the union of the
other generators' pre-pooling train sources is only a non-training upper bound;
it must not be reported as overlap with the neural model's actual train rows.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_modern_dataset import normalize_text


ROOT = Path(__file__).resolve().parents[1]
HUMAN_SOURCE = ROOT / "dataset/fake reviews dataset.csv"
GPT2_DATA = ROOT / "dataset/modern_reviews_gpt2.csv"
OUTPUT = ROOT / "results/source_overlap.json"

CONFIG = {
    "LFM": {
        "raw": ROOT / "dataset/sota_fakes.csv",
        "assembled": ROOT / "dataset/modern_reviews_lfm.csv",
        "matched_run": ROOT / "results/xl_rs_r64_s7_07120532",
        "logo_data": ROOT / "dataset/pooled_wo_lfm.csv",
        "logo_run": ROOT / "results/pwolfm_rs_r64_s1998_07120820",
    },
    "DeepSeek": {
        "raw": ROOT / "dataset/sota_fakes_deepseek.csv",
        "assembled": ROOT / "dataset/modern_reviews_deepseek.csv",
        "matched_run": ROOT / "results/xd_rs_r64_s7_07120556",
        "logo_data": ROOT / "dataset/pooled_wo_deepseek.csv",
        "logo_run": ROOT / "results/pwodeepseek_rs_r64_s1998_07120841",
    },
    "GLM": {
        "raw": ROOT / "dataset/sota_fakes_glm.csv",
        "assembled": ROOT / "dataset/modern_reviews_glm.csv",
        "matched_run": ROOT / "results/xglm_rs_r64_s7_07120621",
        "logo_data": ROOT / "dataset/pooled_wo_glm.csv",
        "logo_run": ROOT / "results/pwoglm_rs_r64_s1998_07120910",
    },
}

MATCHED_CONFIG = {
    "GPT-2": {
        "assembled": GPT2_DATA,
        "matched_run": ROOT / "results/xg_rs_r64_s7_07120504",
    },
    **{
        name: {
            "assembled": config["assembled"],
            "matched_run": config["matched_run"],
        }
        for name, config in CONFIG.items()
    },
}

RowKey = tuple[str, str, str, str]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_key(row: dict[str, str]) -> RowKey:
    return row["category"], row["rating"], row["label"], row["text_"]


def probability_key(row: dict[str, str]) -> RowKey:
    return row["category"], row["rating"], row["true"], row["text"]


def count_metric(count: int, total: int) -> dict[str, int | float]:
    return {"count": count, "fraction": count / total}


def source_reviews() -> tuple[dict[int, str], set[str]]:
    """Return normalized OR text by original CSV row index and its lower keys."""
    by_idx = {}
    lower_texts = set()
    for idx, row in enumerate(rows(HUMAN_SOURCE)):
        if row["label"] != "OR":
            continue
        text = normalize_text(row["text_"])
        by_idx[idx] = text
        lower_texts.add(text.lower())
    return by_idx, lower_texts


def retained_source_map(
    raw_path: Path,
    assembled_path: Path,
    human_lower_texts: set[str],
) -> tuple[dict[RowKey, int], dict[str, int | bool]]:
    """Reproduce build_modern_dataset.py's ordered CG retention exactly."""
    retained: dict[RowKey, int] = {}
    seen: set[str] = set()
    raw_cg_rows = 0
    dropped_empty = 0
    dropped_duplicate = 0
    duplicate_lower_texts: set[str] = set()
    dropped_cross_class = 0

    for row in rows(raw_path):
        if row.get("label", "CG") != "CG":
            continue
        raw_cg_rows += 1
        text = normalize_text(row["text_"])
        lower = text.lower()
        if not text:
            dropped_empty += 1
        elif lower in seen:
            dropped_duplicate += 1
            duplicate_lower_texts.add(lower)
        elif lower in human_lower_texts:
            dropped_cross_class += 1
        else:
            # build_modern_dataset keeps the first raw occurrence of a
            # normalized, lower-cased text and rejects every later occurrence.
            seen.add(lower)
            key = (row["category"], row["rating"], "CG", text)
            if key in retained:
                raise RuntimeError(f"duplicate retained row key in {raw_path}: {key!r}")
            retained[key] = int(row["src_idx"])

    assembled_cg = Counter(
        dataset_key(row) for row in rows(assembled_path) if row["label"] == "CG"
    )
    retained_rows = Counter(retained.keys())
    if assembled_cg != retained_rows:
        missing = assembled_cg - retained_rows
        extra = retained_rows - assembled_cg
        raise RuntimeError(
            f"first-retained mapping does not reproduce {assembled_path}: "
            f"missing={sum(missing.values())}, extra={sum(extra.values())}"
        )
    unique_sources = len(set(retained.values()))
    if unique_sources != len(retained):
        raise RuntimeError(
            f"retained rows in {raw_path} are not one-to-one with src_idx "
            f"({len(retained)} rows, {unique_sources} sources)"
        )

    total_discarded = raw_cg_rows - len(retained)
    classified_discarded = dropped_empty + dropped_duplicate + dropped_cross_class
    if total_discarded != classified_discarded:
        raise RuntimeError(
            f"discard accounting failure for {raw_path}: "
            f"total={total_discarded}, classified={classified_discarded}"
        )

    stats = {
        "raw_cg_rows": raw_cg_rows,
        "retained_cg_rows": len(retained),
        "retained_unique_src_idx": unique_sources,
        "total_discarded_rows": total_discarded,
        "discarded_empty": dropped_empty,
        "discarded_normalized_duplicate_after_first": dropped_duplicate,
        "within_cg_normalized_duplicate_groups": len(duplicate_lower_texts),
        "discarded_normalized_cross_class": dropped_cross_class,
        "all_discarded_rows_are_within_cg_normalized_duplicates": (
            total_discarded == dropped_duplicate
        ),
    }
    return retained, stats


def reconstruct_splits(
    data_path: Path,
    run_path: Path,
    *,
    verify_config_hash: bool,
) -> tuple[dict[str, Counter[RowKey]], dict[str, object]]:
    """Recover exact train/validation/test multisets from saved predictions."""
    config_path = run_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured_data = Path(config["data"])
    if not configured_data.is_absolute():
        configured_data = ROOT / configured_data
    if configured_data.resolve() != data_path.resolve():
        raise RuntimeError(
            f"run {run_path} names {configured_data}, expected {data_path}"
        )
    if config.get("holdout_category") is not None:
        raise RuntimeError(f"run {run_path} is not an outer 80/20 split")

    actual_hash = sha256(data_path)
    expected_hash = config.get("data_sha256")
    hash_verified = expected_hash is not None and actual_hash == expected_hash
    if verify_config_hash and not hash_verified:
        raise RuntimeError(
            f"SHA-256 mismatch for {data_path}: run config has {expected_hash}, "
            f"current file has {actual_hash}"
        )

    full = Counter(dataset_key(row) for row in rows(data_path))
    val = Counter(probability_key(row) for row in rows(run_path / "val_probs.csv"))
    test = Counter(probability_key(row) for row in rows(run_path / "test_probs.csv"))
    held_out = val + test
    missing = held_out - full
    if missing:
        example = next(iter(missing))
        raise RuntimeError(
            f"saved val/test rows are not a sub-multiset of {data_path}: "
            f"missing={sum(missing.values())}, example={example!r}"
        )
    train = full - held_out
    reconstructed = train + val + test
    if reconstructed != full:
        raise RuntimeError(f"multiset reconstruction count failure for {run_path}")

    label_counts = Counter(key[2] for key in train.elements())
    ambiguous_saved_rows = sum(
        count for key, count in held_out.items() if full[key] != 1 or count != 1
    )
    validation: dict[str, object] = {
        "data_path": relative(data_path),
        "run_path": relative(run_path),
        "config_path": relative(config_path),
        "val_probs_path": relative(run_path / "val_probs.csv"),
        "test_probs_path": relative(run_path / "test_probs.csv"),
        "data_rows": sum(full.values()),
        "val_rows_subtracted": sum(val.values()),
        "test_rows_subtracted": sum(test.values()),
        "actual_train_rows": sum(train.values()),
        "actual_train_label_counts": dict(sorted(label_counts.items())),
        "config_data_path_verified": True,
        "config_split_seed": config.get("split_seed"),
        "config_protocol": config.get("protocol"),
        "saved_prediction_rows_with_nonunique_full_row_key": ambiguous_saved_rows,
        "saved_prediction_rows_one_to_one_verified": ambiguous_saved_rows == 0,
        "exact_multiset_subtraction_verified": True,
        "exact_train_validation_test_multiset_identity_verified": True,
    }
    if verify_config_hash:
        validation.update(
            {
                "data_sha256": actual_hash,
                "config_data_sha256": expected_hash,
                "config_sha256_verified": hash_verified,
            }
        )
    return {"train": train, "validation": val, "test": test}, validation


def reconstruct_train(
    data_path: Path,
    run_path: Path,
    *,
    verify_config_hash: bool,
) -> tuple[Counter[RowKey], dict[str, object]]:
    splits, validation = reconstruct_splits(
        data_path, run_path, verify_config_hash=verify_config_hash
    )
    return splits["train"], validation


def exact_text_split_audit(
    splits: dict[str, Counter[RowKey]],
    reconstruction_validation: dict[str, object],
) -> dict[str, object]:
    """Describe literal text duplicates and their reconstructed membership."""
    split_names = ("train", "validation", "test")
    full: Counter[RowKey] = Counter()
    for split_name in split_names:
        full.update(splits[split_name])

    split_sizes = {}
    for split_name in split_names:
        label_counts = Counter()
        for key, count in splits[split_name].items():
            label_counts[key[2]] += count
        split_sizes[split_name] = {
            "rows": sum(splits[split_name].values()),
            "label_counts": dict(sorted(label_counts.items())),
        }

    duplicate_report = {}
    duplicate_details_by_label: dict[str, list[dict[str, object]]] = {}
    for label in ("OR", "CG"):
        text_counts: Counter[str] = Counter()
        label_rows = 0
        for key, count in full.items():
            if key[2] == label:
                text_counts[key[3]] += count
                label_rows += count
        duplicate_texts = sorted(text for text, count in text_counts.items() if count > 1)
        details = []
        for text in duplicate_texts:
            variants = sorted(
                (key for key in full if key[2] == label and key[3] == text),
                key=lambda key: (key[0], key[1], key[3]),
            )
            occurrences_by_split = {
                split_name: sum(
                    count
                    for key, count in splits[split_name].items()
                    if key[2] == label and key[3] == text
                )
                for split_name in split_names
            }
            occurrences = text_counts[text]
            details.append(
                {
                    "text": text,
                    "occurrences": occurrences,
                    "exact_composite_row_key_variants": len(variants),
                    "composite_row_keys": [
                        {
                            "category": key[0],
                            "rating": key[1],
                            "label": key[2],
                            "text": key[3],
                            "occurrences": full[key],
                        }
                        for key in variants
                    ],
                    "occurrences_by_split": occurrences_by_split,
                    "all_occurrences_in_train": (
                        occurrences_by_split["train"] == occurrences
                        and occurrences_by_split["validation"] == 0
                        and occurrences_by_split["test"] == 0
                    ),
                }
            )
        duplicate_details_by_label[label] = details
        duplicate_report[label] = {
            "rows": label_rows,
            "unique_exact_texts": len(text_counts),
            "duplicate_exact_text_groups": len(duplicate_texts),
            "rows_in_duplicate_groups": sum(text_counts[text] for text in duplicate_texts),
            "excess_duplicate_rows": sum(
                text_counts[text] - 1 for text in duplicate_texts
            ),
            "groups": details,
        }

    split_texts = {
        split_name: {key[3] for key in splits[split_name]}
        for split_name in split_names
    }
    split_texts_by_label = {
        label: {
            split_name: {
                key[3] for key in splits[split_name] if key[2] == label
            }
            for split_name in split_names
        }
        for label in ("OR", "CG")
    }
    overlap_report = {}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        exact_texts = split_texts[left] & split_texts[right]
        exact_row_keys = set(splits[left]) & set(splits[right])
        overlap_report[f"{left}__{right}"] = {
            "unique_exact_texts_all_labels": len(exact_texts),
            "unique_exact_texts_by_label": {
                label: len(
                    split_texts_by_label[label][left]
                    & split_texts_by_label[label][right]
                )
                for label in ("OR", "CG")
            },
            "overlapping_exact_texts_all_labels": sorted(exact_texts),
            "unique_exact_composite_row_keys": len(exact_row_keys),
        }

    triple_texts = set.intersection(*(split_texts[name] for name in split_names))
    or_details = duplicate_details_by_label["OR"]
    repeated_or_pair = (
        len(or_details) == 1 and or_details[0]["occurrences"] == 2
    )
    repeated_or_pair_in_train = (
        repeated_or_pair and bool(or_details[0]["all_occurrences_in_train"])
    )
    or_texts = {key[3] for key in full if key[2] == "OR"}
    cg_texts = {key[3] for key in full if key[2] == "CG"}

    return {
        "data_path": reconstruction_validation["data_path"],
        "run_path": reconstruction_validation["run_path"],
        "config_path": reconstruction_validation["config_path"],
        "val_probs_path": reconstruction_validation["val_probs_path"],
        "test_probs_path": reconstruction_validation["test_probs_path"],
        "split_sizes": split_sizes,
        "within_dataset_duplicate_groups_by_label": duplicate_report,
        "repeated_or_pair_summary": {
            "exactly_one_repeated_or_pair": repeated_or_pair,
            "both_repeated_or_rows_are_in_train": repeated_or_pair_in_train,
        },
        "or_cg_exact_text_overlap_in_full_dataset": {
            "unique_exact_texts": len(or_texts & cg_texts),
            "texts": sorted(or_texts & cg_texts),
        },
        "exact_text_overlap_across_splits": overlap_report,
        "train_validation_test_triple_exact_text_overlap": {
            "unique_exact_texts_all_labels": len(triple_texts),
            "texts": sorted(triple_texts),
        },
        "validation": {
            "config_data_path_verified": reconstruction_validation[
                "config_data_path_verified"
            ],
            "config_split_seed": reconstruction_validation["config_split_seed"],
            "config_protocol": reconstruction_validation["config_protocol"],
            "saved_prediction_rows_with_nonunique_full_row_key": (
                reconstruction_validation[
                    "saved_prediction_rows_with_nonunique_full_row_key"
                ]
            ),
            "saved_prediction_rows_one_to_one_verified": (
                reconstruction_validation[
                    "saved_prediction_rows_one_to_one_verified"
                ]
            ),
            "exact_train_validation_test_multiset_identity_verified": (
                reconstruction_validation[
                    "exact_train_validation_test_multiset_identity_verified"
                ]
            ),
            "exactly_one_repeated_or_pair_verified": repeated_or_pair,
            "both_repeated_or_rows_in_train_verified": repeated_or_pair_in_train,
            "all_cross_split_exact_text_overlaps_zero": (
                all(
                    report["unique_exact_texts_all_labels"] == 0
                    for report in overlap_report.values()
                )
                and not triple_texts
            ),
        },
    }


def main() -> None:
    source_text_by_idx, human_lower_texts = source_reviews()

    retained_maps: dict[str, dict[RowKey, int]] = {}
    retention_validation = {}
    for name, config in CONFIG.items():
        retained, stats = retained_source_map(
            config["raw"], config["assembled"], human_lower_texts
        )
        retained_maps[name] = retained
        retention_validation[name] = stats

    matched_splits: dict[str, dict[str, Counter[RowKey]]] = {}
    matched_validation = {}
    for name, config in MATCHED_CONFIG.items():
        splits, validation = reconstruct_splits(
            config["assembled"], config["matched_run"], verify_config_hash=False
        )
        matched_splits[name] = splits
        matched_validation[name] = validation

    matched_exact_text_audits = {
        name: exact_text_split_audit(matched_splits[name], matched_validation[name])
        for name in MATCHED_CONFIG
    }
    exact_text_checks = [
        audit["validation"] for audit in matched_exact_text_audits.values()
    ]
    all_matched_prediction_rows_one_to_one = all(
        check["saved_prediction_rows_one_to_one_verified"]
        for check in exact_text_checks
    )
    all_matched_multisets_verified = all(
        check["exact_train_validation_test_multiset_identity_verified"]
        for check in exact_text_checks
    )
    all_exactly_one_repeated_or_pair = all(
        check["exactly_one_repeated_or_pair_verified"]
        for check in exact_text_checks
    )
    all_repeated_or_pairs_in_train = all(
        check["both_repeated_or_rows_in_train_verified"]
        for check in exact_text_checks
    )
    all_cross_split_exact_text_overlaps_zero = all(
        check["all_cross_split_exact_text_overlaps_zero"]
        for check in exact_text_checks
    )
    if not (
        all_matched_prediction_rows_one_to_one
        and all_matched_multisets_verified
        and all_exactly_one_repeated_or_pair
        and all_repeated_or_pairs_in_train
        and all_cross_split_exact_text_overlaps_zero
    ):
        raise RuntimeError("matched exact-text split audit validation failed")

    matched_trains: dict[str, Counter[RowKey]] = {}
    matched_train_sources: dict[str, set[int]] = {}
    target_test_sources: dict[str, list[int]] = {}
    target_mapping_validation = {}

    for name, config in CONFIG.items():
        train = matched_splits[name]["train"]
        matched_trains[name] = train

        train_cg_keys = [key for key in train.elements() if key[2] == "CG"]
        missing_train = [key for key in train_cg_keys if key not in retained_maps[name]]
        if missing_train:
            raise RuntimeError(
                f"{name} has {len(missing_train)} unmapped CG training rows"
            )
        train_sources = [retained_maps[name][key] for key in train_cg_keys]
        if len(set(train_sources)) != len(train_sources):
            raise RuntimeError(f"{name} CG training rows are not one-to-one with src_idx")
        matched_train_sources[name] = set(train_sources)

        target_rows = [
            probability_key(row)
            for row in rows(config["matched_run"] / "test_probs.csv")
            if row["true"] == "CG"
        ]
        mapped = [retained_maps[name].get(key) for key in target_rows]
        unmapped = sum(src_idx is None for src_idx in mapped)
        mapped_sources = [src_idx for src_idx in mapped if src_idx is not None]
        duplicate_source_rows = len(mapped_sources) - len(set(mapped_sources))
        duplicate_target_rows = len(target_rows) - len(set(target_rows))
        if unmapped or duplicate_source_rows or duplicate_target_rows:
            raise RuntimeError(
                f"{name} target mapping is not one-to-one: unmapped={unmapped}, "
                f"duplicate rows={duplicate_target_rows}, "
                f"duplicate src_idx={duplicate_source_rows}"
            )
        target_test_sources[name] = mapped_sources
        target_mapping_validation[name] = {
            "target_cg_test_rows": len(target_rows),
            "mapped_rows": len(mapped_sources),
            "unique_target_row_keys": len(set(target_rows)),
            "unique_src_idx": len(set(mapped_sources)),
            "unmapped_rows": unmapped,
            "ambiguous_rows_after_first_retained_dedup": 0,
            "one_to_one_verified": True,
        }

    pairwise = []
    for target, target_source_list in target_test_sources.items():
        target_sources = set(target_source_list)
        for train_generator, train_sources in matched_train_sources.items():
            if train_generator == target:
                continue
            overlap = len(train_sources & target_sources)
            pairwise.append(
                {
                    "train_generator": train_generator,
                    "test_generator": target,
                    "overlap": overlap,
                    "test_sources": len(target_sources),
                    "fraction": overlap / len(target_sources),
                    "actual_train_cg_rows": sum(
                        count
                        for key, count in matched_trains[train_generator].items()
                        if key[2] == "CG"
                    ),
                    "actual_train_unique_sources": len(train_sources),
                    "train_data_path": relative(CONFIG[train_generator]["assembled"]),
                    "train_run_path": relative(CONFIG[train_generator]["matched_run"]),
                    "test_data_path": relative(CONFIG[target]["assembled"]),
                    "test_run_path": relative(CONFIG[target]["matched_run"]),
                }
            )

    # Map every possible pooled CG row to its generator and, for modern rows,
    # its retained source index.  Restrict candidates to the generators that
    # actually supplied a given target's pooled dataset.
    memberships: dict[RowKey, list[tuple[str, int | None]]] = defaultdict(list)
    for name, retained in retained_maps.items():
        for key, src_idx in retained.items():
            memberships[key].append((name, src_idx))
    for row in rows(GPT2_DATA):
        if row["label"] == "CG":
            memberships[dataset_key(row)].append(("GPT-2", None))

    logo_support = []
    logo_validation = {}
    for target, config in CONFIG.items():
        logo_train, validation = reconstruct_train(
            config["logo_data"], config["logo_run"], verify_config_hash=True
        )
        allowed_generators = ({"GPT-2"} | set(CONFIG)) - {target}
        train_source_ids: set[int] = set()
        cg_rows_by_generator: Counter[str] = Counter()
        unmapped_rows = 0
        ambiguous_rows = 0
        train_cg_rows = 0

        for key, count in logo_train.items():
            if key[2] != "CG":
                continue
            train_cg_rows += count
            candidates = [
                candidate
                for candidate in memberships.get(key, [])
                if candidate[0] in allowed_generators
            ]
            if not candidates:
                unmapped_rows += count
                continue
            if len(candidates) != 1:
                ambiguous_rows += count
                continue
            generator, src_idx = candidates[0]
            cg_rows_by_generator[generator] += count
            if src_idx is not None:
                train_source_ids.add(src_idx)

        if unmapped_rows or ambiguous_rows:
            raise RuntimeError(
                f"{target} pooled CG mapping failed: unmapped={unmapped_rows}, "
                f"ambiguous={ambiguous_rows}"
            )
        if sum(cg_rows_by_generator.values()) != train_cg_rows:
            raise RuntimeError(f"{target} pooled CG generator counts do not sum")

        train_or_texts = {
            key[3] for key in logo_train.elements() if key[2] == "OR"
        }
        target_sources = set(target_test_sources[target])
        missing_groundings = target_sources - source_text_by_idx.keys()
        if missing_groundings:
            raise RuntimeError(
                f"{target} has source indices absent from the OR source data: "
                f"{sorted(missing_groundings)[:5]}"
            )

        direct = target_sources & train_source_ids
        grounding = {
            src_idx
            for src_idx in target_sources
            if source_text_by_idx[src_idx] in train_or_texts
        }
        both = direct & grounding
        direct_only = direct - grounding
        grounding_only = grounding - direct
        either = direct | grounding
        neither = target_sources - either
        total = len(target_sources)
        if len(both) + len(direct_only) + len(grounding_only) + len(neither) != total:
            raise RuntimeError(f"{target} support partition does not sum to target")

        train_label_counts = Counter()
        for key, count in logo_train.items():
            train_label_counts[key[2]] += count
        logo_support.append(
            {
                "test_generator": target,
                "target_cg_test_rows": total,
                "target_unique_sources": total,
                "actual_logo_train_rows": sum(logo_train.values()),
                "actual_logo_train_or_rows": train_label_counts["OR"],
                "actual_logo_train_cg_rows": train_label_counts["CG"],
                "actual_logo_train_cg_rows_by_generator": dict(
                    sorted(cg_rows_by_generator.items())
                ),
                "actual_logo_train_cg_rows_with_src_idx": sum(
                    count
                    for generator, count in cg_rows_by_generator.items()
                    if generator != "GPT-2"
                ),
                "actual_logo_train_unique_src_idx": len(train_source_ids),
                "direct_cg_source_overlap": count_metric(len(direct), total),
                "exact_grounding_or_text_in_train": count_metric(
                    len(grounding), total
                ),
                "support_partition": {
                    "both": count_metric(len(both), total),
                    "direct_cg_source_only": count_metric(len(direct_only), total),
                    "exact_grounding_or_text_only": count_metric(
                        len(grounding_only), total
                    ),
                    "either": count_metric(len(either), total),
                    "neither": count_metric(len(neither), total),
                },
                "train_data_path": relative(config["logo_data"]),
                "train_run_path": relative(config["logo_run"]),
                "target_data_path": relative(config["assembled"]),
                "target_test_run_path": relative(config["matched_run"]),
            }
        )

        validation["mapped_actual_train_cg_rows"] = train_cg_rows
        validation["unmapped_actual_train_cg_rows"] = unmapped_rows
        validation["ambiguous_actual_train_cg_rows"] = ambiguous_rows
        validation["cg_row_mapping_verified"] = True
        logo_validation[target] = validation

    upper_bound_values = []
    for target, target_source_list in target_test_sources.items():
        source_generators = [name for name in CONFIG if name != target]
        candidate_sources = set().union(
            *(matched_train_sources[name] for name in source_generators)
        )
        target_sources = set(target_source_list)
        overlap = len(candidate_sources & target_sources)
        upper_bound_values.append(
            {
                "test_generator": target,
                "candidate_modern_generators": source_generators,
                "overlap": overlap,
                "test_sources": len(target_sources),
                "fraction": overlap / len(target_sources),
            }
        )

    artifacts = {
        "human_source_data": relative(HUMAN_SOURCE),
        "gpt2_assembled_data": relative(GPT2_DATA),
        "matched_split_datasets": {
            name: {
                "assembled_data": relative(config["assembled"]),
                "matched_run": relative(config["matched_run"]),
                "config": relative(config["matched_run"] / "config.json"),
                "val_probs": relative(config["matched_run"] / "val_probs.csv"),
                "test_probs": relative(config["matched_run"] / "test_probs.csv"),
            }
            for name, config in MATCHED_CONFIG.items()
        },
        "targets": {
            name: {
                "raw_generation_data": relative(config["raw"]),
                "assembled_data": relative(config["assembled"]),
                "matched_split_run": relative(config["matched_run"]),
                "matched_val_probs": relative(config["matched_run"] / "val_probs.csv"),
                "matched_test_probs": relative(config["matched_run"] / "test_probs.csv"),
                "logo_pooled_data": relative(config["logo_data"]),
                "logo_neural_run": relative(config["logo_run"]),
                "logo_val_probs": relative(config["logo_run"] / "val_probs.csv"),
                "logo_test_probs": relative(config["logo_run"] / "test_probs.csv"),
            }
            for name, config in CONFIG.items()
        },
    }

    result = {
        "schema_version": 3,
        "scope": (
            "Source-support results target LFM, DeepSeek, and GLM because GPT-2 "
            "CG rows have no src_idx; the matched exact-text split audit covers "
            "GPT-2 and all three modern datasets."
        ),
        "interpretation": {
            "headline": (
                "LOGO holds out generator identity, but its actual neural training "
                "rows retain substantial shared grounding support. It is not a "
                "joint generator-and-source OOD test."
            ),
            "direct_cg_source_overlap": (
                "A target CG test row's src_idx also occurs on a modern CG row in "
                "the reconstructed actual pooled LOGO training split."
            ),
            "exact_grounding_or_text_in_train": (
                "The normalized human review used to ground the target CG test row "
                "occurs verbatim as an OR row in the reconstructed actual pooled "
                "LOGO training split."
            ),
            "pairwise": (
                "Pairwise values compare each other modern generator's recovered "
                "actual outer-80% matched-run training sources with the target's "
                "CG test sources."
            ),
            "non_training_upper_bound": (
                "The approximately 94% values union the other modern generators' "
                "pre-pooling candidate train sources. They precede balanced "
                "subsampling and the LOGO run's own 80/20 split, so they are not "
                "overlap with rows seen by the neural model."
            ),
            "matched_exact_text_split_audit": (
                "Every assembled dataset has exactly one duplicated literal OR "
                "text pair. Both occurrences are confined to its reconstructed "
                "training multiset, so no literal text is shared across its "
                "frozen train, validation, and test splits."
            ),
        },
        "artifacts": artifacts,
        "validation": {
            "mapping_rule": (
                "Normalize in raw CSV order and retain the first nonempty, "
                "case-insensitively unique, non-OR-colliding generated text, "
                "matching build_modern_dataset.py."
            ),
            "retained_mapping": retention_validation,
            "target_test_mapping": target_mapping_validation,
            "matched_actual_train_reconstruction": matched_validation,
            "logo_actual_train_reconstruction": logo_validation,
            "all_target_mappings_one_to_one": True,
            "all_pooled_train_cg_rows_unambiguous": True,
            "all_logo_data_sha256_verified_against_neural_run_config": True,
            "all_modern_assembly_drops_are_within_cg_normalized_duplicates": all(
                stats[
                    "all_discarded_rows_are_within_cg_normalized_duplicates"
                ]
                for stats in retention_validation.values()
            ),
        },
        "matched_split_exact_text_audit": {
            "equality_semantics": {
                "row_reconstruction": (
                    "Exact, case-sensitive equality of the four parsed CSV fields "
                    "(category, rating, label, text); no normalization or trimming. "
                    "Saved true/text fields are compared with assembled label/text_ "
                    "fields. Duplicate rows are handled as multisets."
                ),
                "duplicate_and_overlap": (
                    "Exact, case-sensitive Python string equality of parsed text "
                    "values only; no normalization, case folding, or whitespace or "
                    "punctuation changes. Counts are numbers of unique literal text "
                    "strings unless explicitly named as rows or occurrences."
                ),
            },
            "interpretation": (
                "GPT-2, LFM, DeepSeek, and GLM each contain one repeated OR text "
                "pair and no repeated CG text. The two OR occurrences are both in "
                "the reconstructed training multiset for every dataset; all three "
                "pairwise train/validation/test exact-text intersections are zero."
            ),
            "datasets": matched_exact_text_audits,
            "validation": {
                "all_four_saved_prediction_mappings_one_to_one": (
                    all_matched_prediction_rows_one_to_one
                ),
                "all_four_split_multiset_identities_verified": (
                    all_matched_multisets_verified
                ),
                "all_four_have_exactly_one_repeated_or_pair": (
                    all_exactly_one_repeated_or_pair
                ),
                "both_repeated_or_rows_in_training_for_all_four": (
                    all_repeated_or_pairs_in_train
                ),
                "all_four_have_zero_cross_split_exact_text_overlap": (
                    all_cross_split_exact_text_overlaps_zero
                ),
            },
        },
        "pairwise_actual_train_source_overlap": pairwise,
        "per_target_actual_logo_training_support": logo_support,
        "non_training_upper_bound_candidate_pool": {
            "label": "NOT ACTUAL LOGO TRAINING; PRE-SUBSAMPLING UPPER BOUND",
            "values": upper_bound_values,
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
