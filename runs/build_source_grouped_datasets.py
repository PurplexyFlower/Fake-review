"""Build strict source-review-grouped datasets for the source-grouped experiments.

The three contemporary generators were grounded on the same original OR pool.
This builder recovers the retained ``src_idx`` for every generated row, keeps
only sources represented in all three generator artifacts, and assigns source
groups to train/validation/test exactly once.  An OR row and every generated
derivative of that source therefore always have the same split.

Outputs (generated, not committed):

* dataset/source_grouped/modern_reviews_{lfm|deepseek|glm}.csv
* dataset/source_grouped/pooled_wo_{lfm|deepseek|glm}.csv
* dataset/source_grouped/manifest.json

The generator-specific files contain one OR and one CG row per source.  Each
contemporary LOGO file contains one OR and one deterministically selected CG
row per source, balanced between the two non-held-out generators.  This keeps
the LOGO training budget balanced while removing target-generator and target-
source exposure from the training partition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_modern_dataset import normalize_text


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
DEFAULT_OUTPUT = DATASET / "source_grouped"
HUMAN_SOURCE = DATASET / "fake reviews dataset.csv"
SPLIT_SEED = 1998
SPLIT_FRACTIONS = {"train": 0.80, "validation": 0.06, "test": 0.14}
GENERATORS = {
    "lfm": {
        "raw": DATASET / "sota_fakes.csv",
        "assembled": DATASET / "modern_reviews_lfm.csv",
    },
    "deepseek": {
        "raw": DATASET / "sota_fakes_deepseek.csv",
        "assembled": DATASET / "modern_reviews_deepseek.csv",
    },
    "glm": {
        "raw": DATASET / "sota_fakes_glm.csv",
        "assembled": DATASET / "modern_reviews_glm.csv",
    },
}
OUTPUT_COLUMNS = [
    "category",
    "rating",
    "label",
    "text_",
    "source_id",
    "source_group",
    "generator",
    "split",
]
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_order(*parts: object) -> str:
    value = "|".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_rows() -> tuple[dict[int, dict[str, str]], dict[str, list[int]]]:
    sources: dict[int, dict[str, str]] = {}
    ids_by_text: dict[str, list[int]] = defaultdict(list)
    for source_id, row in enumerate(read_rows(HUMAN_SOURCE)):
        if row["label"] != "OR":
            continue
        text = normalize_text(row["text_"])
        if not text:
            raise RuntimeError(f"empty normalized OR source at row {source_id}")
        normalized = {
            "category": row["category"],
            "rating": row["rating"],
            "label": "OR",
            "text_": text,
        }
        sources[source_id] = normalized
        ids_by_text[text.casefold()].append(source_id)
    return sources, ids_by_text


def retained_generated_rows(
    generator: str,
    raw_path: Path,
    assembled_path: Path,
    sources: dict[int, dict[str, str]],
    human_lower_texts: set[str],
) -> tuple[dict[int, dict[str, str]], dict[str, int | bool]]:
    """Reproduce the ordered first-retained assembly mapping exactly."""
    retained: dict[int, dict[str, str]] = {}
    retained_keys: Counter[tuple[str, str, str, str]] = Counter()
    seen: set[str] = set()
    raw_cg_rows = 0
    dropped_empty = 0
    dropped_duplicate = 0
    dropped_cross_class = 0

    for row in read_rows(raw_path):
        if row.get("label", "CG") != "CG":
            continue
        raw_cg_rows += 1
        text = normalize_text(row["text_"])
        lower = text.casefold()
        if not text:
            dropped_empty += 1
            continue
        if lower in seen:
            dropped_duplicate += 1
            continue
        if lower in human_lower_texts:
            dropped_cross_class += 1
            continue
        seen.add(lower)
        source_id = int(row["src_idx"])
        if source_id not in sources:
            raise RuntimeError(
                f"{generator}: src_idx={source_id} is not an OR source row"
            )
        if source_id in retained:
            raise RuntimeError(f"{generator}: multiple retained CG rows for {source_id}")
        source = sources[source_id]
        if row["category"] != source["category"] or row["rating"] != source["rating"]:
            raise RuntimeError(
                f"{generator}: source metadata mismatch for src_idx={source_id}"
            )
        normalized = {
            "category": row["category"],
            "rating": row["rating"],
            "label": "CG",
            "text_": text,
        }
        retained[source_id] = normalized
        retained_keys[(row["category"], row["rating"], "CG", text)] += 1

    assembled_keys = Counter(
        (row["category"], row["rating"], row["label"], row["text_"])
        for row in read_rows(assembled_path)
        if row["label"] == "CG"
    )
    if retained_keys != assembled_keys:
        missing = assembled_keys - retained_keys
        extra = retained_keys - assembled_keys
        raise RuntimeError(
            f"{generator}: recovered CG mapping does not reproduce assembled data "
            f"(missing={sum(missing.values())}, extra={sum(extra.values())})"
        )

    discarded = raw_cg_rows - len(retained)
    classified = dropped_empty + dropped_duplicate + dropped_cross_class
    if discarded != classified:
        raise RuntimeError(
            f"{generator}: discarded-row accounting mismatch "
            f"({discarded} != {classified})"
        )
    return retained, {
        "raw_cg_rows": raw_cg_rows,
        "retained_cg_rows": len(retained),
        "unique_source_ids": len(retained),
        "dropped_empty": dropped_empty,
        "dropped_duplicate": dropped_duplicate,
        "dropped_cross_class": dropped_cross_class,
        "assembled_mapping_verified": True,
    }


def assign_splits(
    common_sources: set[int],
    sources: dict[int, dict[str, str]],
    ids_by_text: dict[str, list[int]],
) -> tuple[dict[int, str], dict[int, int], dict[str, object]]:
    """Assign normalized-text source groups once, stratified by category/rating."""
    common_by_text: dict[str, list[int]] = defaultdict(list)
    for source_id in common_sources:
        common_by_text[sources[source_id]["text_"].casefold()].append(source_id)

    canonical_group: dict[int, int] = {}
    group_members: dict[int, list[int]] = {}
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    duplicate_groups = 0
    for lower, members in common_by_text.items():
        del lower  # the normalized text is represented by the canonical source id
        members = sorted(members)
        group_id = members[0]
        group_members[group_id] = members
        if len(members) > 1:
            duplicate_groups += 1
        metadata = {(sources[idx]["category"], sources[idx]["rating"]) for idx in members}
        if len(metadata) != 1:
            raise RuntimeError(
                f"exact duplicate OR group {members} crosses category/rating strata"
            )
        for source_id in members:
            canonical_group[source_id] = group_id
        strata[next(iter(metadata))].append(group_id)

    # Confirm that every original exact duplicate represented in the common
    # source universe was collapsed into the same split group.
    for ids in ids_by_text.values():
        common_ids = sorted(set(ids) & common_sources)
        if len(common_ids) > 1:
            groups = {canonical_group[source_id] for source_id in common_ids}
            if len(groups) != 1:
                raise RuntimeError(f"duplicate source rows were split: {common_ids}")

    group_split: dict[int, str] = {}
    stratum_manifest: dict[str, dict[str, int]] = {}
    for (category, rating), groups in sorted(strata.items()):
        ordered = sorted(
            groups,
            key=lambda group_id: stable_order(
                SPLIT_SEED, category, rating, group_id
            ),
        )
        n = len(ordered)
        n_validation = round(SPLIT_FRACTIONS["validation"] * n)
        n_test = round(SPLIT_FRACTIONS["test"] * n)
        n_train = n - n_validation - n_test
        for group_id in ordered[:n_train]:
            group_split[group_id] = "train"
        for group_id in ordered[n_train : n_train + n_validation]:
            group_split[group_id] = "validation"
        for group_id in ordered[n_train + n_validation :]:
            group_split[group_id] = "test"
        stratum_manifest[f"{category}|{rating}"] = {
            "groups": n,
            "train": n_train,
            "validation": n_validation,
            "test": n_test,
        }

    source_split = {
        source_id: group_split[canonical_group[source_id]]
        for source_id in common_sources
    }
    source_counts = Counter(source_split.values())
    group_counts = Counter(group_split.values())
    return source_split, canonical_group, {
        "unique_source_ids": len(common_sources),
        "unique_source_groups": len(group_split),
        "exact_duplicate_groups": duplicate_groups,
        "source_counts": dict(sorted(source_counts.items())),
        "group_counts": dict(sorted(group_counts.items())),
        "strata": stratum_manifest,
    }


def output_row(
    row: dict[str, str],
    source_id: int,
    source_group: int,
    generator: str,
    split: str,
) -> dict[str, str | int]:
    return {
        "category": row["category"],
        "rating": row["rating"],
        "label": row["label"],
        "text_": row["text_"],
        "source_id": source_id,
        "source_group": source_group,
        "generator": generator,
        "split": split,
    }


def write_dataset(
    path: Path,
    rows_to_write: list[dict[str, str | int]],
    dataset_name: str,
) -> dict[str, object]:
    ordered = sorted(
        rows_to_write,
        key=lambda row: (
            SPLIT_ORDER[str(row["split"])],
            stable_order(
                SPLIT_SEED,
                dataset_name,
                row["source_id"],
                row["label"],
                row["generator"],
            ),
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)

    labels = Counter(str(row["label"]) for row in ordered)
    split_labels: dict[str, dict[str, int]] = {}
    for split in SPLIT_ORDER:
        split_labels[split] = dict(
            sorted(
                Counter(
                    str(row["label"])
                    for row in ordered
                    if row["split"] == split
                ).items()
            )
        )
        if split_labels[split].get("OR") != split_labels[split].get("CG"):
            raise RuntimeError(f"{path}: {split} partition is not label-balanced")

    group_splits: dict[int, set[str]] = defaultdict(set)
    for row in ordered:
        group_splits[int(row["source_group"])].add(str(row["split"]))
    leaked_groups = [group for group, splits in group_splits.items() if len(splits) != 1]
    if leaked_groups:
        raise RuntimeError(f"{path}: {len(leaked_groups)} source groups cross splits")

    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "rows": len(ordered),
        "sha256": sha256(path),
        "label_counts": dict(sorted(labels.items())),
        "split_label_counts": split_labels,
        "source_groups_crossing_splits": 0,
    }


def build(output_dir: Path) -> dict[str, object]:
    sources, ids_by_text = source_rows()
    human_lower_texts = set(ids_by_text)
    retained: dict[str, dict[int, dict[str, str]]] = {}
    retention_stats: dict[str, dict[str, int | bool]] = {}
    for generator, paths in GENERATORS.items():
        retained[generator], retention_stats[generator] = retained_generated_rows(
            generator,
            paths["raw"],
            paths["assembled"],
            sources,
            human_lower_texts,
        )

    common_sources = set.intersection(*(set(rows) for rows in retained.values()))
    if not common_sources:
        raise RuntimeError("the contemporary generators have no common source ids")
    source_split, canonical_group, split_manifest = assign_splits(
        common_sources, sources, ids_by_text
    )

    artifacts: dict[str, dict[str, object]] = {}
    for generator in GENERATORS:
        output_rows: list[dict[str, str | int]] = []
        for source_id in sorted(common_sources):
            split = source_split[source_id]
            group_id = canonical_group[source_id]
            output_rows.append(
                output_row(sources[source_id], source_id, group_id, "human", split)
            )
            output_rows.append(
                output_row(
                    retained[generator][source_id],
                    source_id,
                    group_id,
                    generator,
                    split,
                )
            )
        path = output_dir / f"modern_reviews_{generator}.csv"
        artifacts[f"matched_{generator}"] = write_dataset(
            path, output_rows, f"matched_{generator}"
        )

    for held_out in GENERATORS:
        training_generators = sorted(set(GENERATORS) - {held_out})
        selected_generator: dict[int, str] = {}
        selection_counts: dict[str, Counter[str]] = {
            split: Counter() for split in SPLIT_ORDER
        }
        for split in SPLIT_ORDER:
            split_sources = sorted(
                (source_id for source_id in common_sources if source_split[source_id] == split),
                key=lambda source_id: stable_order(
                    SPLIT_SEED, "logo", held_out, split, source_id
                ),
            )
            for position, source_id in enumerate(split_sources):
                generator = training_generators[position % len(training_generators)]
                selected_generator[source_id] = generator
                selection_counts[split][generator] += 1

        output_rows = []
        for source_id in sorted(common_sources):
            split = source_split[source_id]
            group_id = canonical_group[source_id]
            generator = selected_generator[source_id]
            output_rows.append(
                output_row(sources[source_id], source_id, group_id, "human", split)
            )
            output_rows.append(
                output_row(
                    retained[generator][source_id],
                    source_id,
                    group_id,
                    generator,
                    split,
                )
            )
        path = output_dir / f"pooled_wo_{held_out}.csv"
        record = write_dataset(path, output_rows, f"pooled_wo_{held_out}")
        training_cg_generators = {
            str(row["generator"])
            for row in output_rows
            if row["split"] == "train" and row["label"] == "CG"
        }
        if held_out in training_cg_generators:
            raise RuntimeError(
                f"{path}: held-out generator {held_out} appears in LOGO training"
            )
        if training_cg_generators != set(training_generators):
            raise RuntimeError(
                f"{path}: LOGO training generators are {sorted(training_cg_generators)}, "
                f"expected {training_generators}"
            )
        record["held_out_generator"] = held_out
        record["training_generators"] = training_generators
        record["cg_selection_counts"] = {
            split: dict(sorted(counts.items()))
            for split, counts in selection_counts.items()
        }
        artifacts[f"logo_wo_{held_out}"] = record

    input_hashes = {
        HUMAN_SOURCE.resolve().relative_to(ROOT.resolve()).as_posix(): sha256(HUMAN_SOURCE)
    }
    for paths in GENERATORS.values():
        for path in paths.values():
            input_hashes[path.resolve().relative_to(ROOT.resolve()).as_posix()] = sha256(path)

    train_groups = {
        canonical_group[source_id]
        for source_id in common_sources
        if source_split[source_id] == "train"
    }
    validation_groups = {
        canonical_group[source_id]
        for source_id in common_sources
        if source_split[source_id] == "validation"
    }
    test_groups = {
        canonical_group[source_id]
        for source_id in common_sources
        if source_split[source_id] == "test"
    }
    if train_groups & validation_groups or train_groups & test_groups or validation_groups & test_groups:
        raise RuntimeError("global source-group partitions overlap")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "protocol": "source-grouped-contemporary-v1",
        "split_seed": SPLIT_SEED,
        "split_fractions": SPLIT_FRACTIONS,
        "scope": ["lfm", "deepseek", "glm"],
        "source_universe": "intersection of retained source ids across all three generators",
        "source_group_rule": (
            "original src_idx, with exact normalized duplicate OR texts merged for splitting"
        ),
        "retention": retention_stats,
        "split": split_manifest,
        "input_sha256": dict(sorted(input_hashes.items())),
        "artifacts": artifacts,
        "validation": {
            "all_assembled_cg_mappings_verified": True,
            "all_outputs_label_balanced_within_split": True,
            "source_groups_crossing_partitions": 0,
            "directional_train_to_any_test_source_group_overlap": 0,
            "logo_train_to_held_out_test_source_group_overlap": 0,
            "held_out_generator_rows_in_logo_training_files": 0,
        },
    }
    manifest_path = output_dir / "manifest.json"
    # The manifest is compared byte-for-byte during cross-platform ingestion.
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    manifest = build(output_dir)
    print(f"source ids: {manifest['split']['unique_source_ids']}")
    print(f"source groups: {manifest['split']['unique_source_groups']}")
    print(f"source counts: {manifest['split']['source_counts']}")
    for name, record in manifest["artifacts"].items():
        print(f"{name}: {record['rows']} rows -> {record['path']}")
    print(f"manifest -> {(output_dir / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()
