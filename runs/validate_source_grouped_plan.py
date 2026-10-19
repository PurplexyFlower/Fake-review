"""Validate the master source-grouped grids and their three-instance shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRIDS = ROOT / "grids"
SHARDS = GRIDS / "source_grouped"
SOURCE_DATA = ROOT / "dataset" / "source_grouped"


def lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def options(line: str) -> dict[str, str | bool]:
    tokens = shlex.split(line)
    parsed: dict[str, str | bool] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"unexpected grid token {token!r} in {line!r}")
        key = token[2:]
        if index + 1 == len(tokens) or tokens[index + 1].startswith("--"):
            parsed[key] = True
            index += 1
        else:
            parsed[key] = tokens[index + 1]
            index += 2
    return parsed


def validate_line(line: str) -> None:
    parsed = options(line)
    required = {"tag", "scaling", "rank", "seed", "data", "predefined-splits"}
    missing = required - parsed.keys()
    if missing:
        raise RuntimeError(f"grid line is missing {sorted(missing)}: {line}")
    if parsed["predefined-splits"] is not True:
        raise RuntimeError(f"--predefined-splits must be a flag: {line}")
    if parsed["scaling"] != "rs":
        raise RuntimeError(f"source-grouped plan expects Rs scaling: {line}")


def validate_completed(grid: Path) -> None:
    from run_queue import completed_rows, is_completed

    ledger = completed_rows()
    missing = [line for line in lines(grid) if not is_completed(line, ledger, True)]
    if missing:
        joined = "\n  ".join(missing)
        raise RuntimeError(f"{len(missing)} required runs are incomplete:\n  {joined}")
    print(f"completed grid verified: {grid}")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_datasets() -> None:
    manifest_path = SOURCE_DATA / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            "source-grouped manifest is missing; run runs/build_source_grouped_datasets.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "source-grouped-contemporary-v1":
        raise RuntimeError("unexpected source-grouped protocol in manifest")
    expected_artifacts = {
        "matched_lfm",
        "matched_deepseek",
        "matched_glm",
        "logo_wo_lfm",
        "logo_wo_deepseek",
        "logo_wo_glm",
    }
    artifacts = manifest.get("artifacts", {})
    if set(artifacts) != expected_artifacts:
        raise RuntimeError("source-grouped manifest does not contain the six core datasets")

    global_group_split: dict[str, str] = {}
    artifact_rows: dict[str, list[dict[str, str]]] = {}
    required_columns = {
        "category",
        "rating",
        "label",
        "text_",
        "source_id",
        "source_group",
        "generator",
        "split",
    }
    for name, record in sorted(artifacts.items()):
        path = ROOT / record["path"]
        if not path.exists() or file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"missing or hash-mismatched source-grouped artifact: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if set(reader.fieldnames or []) != required_columns:
                raise RuntimeError(f"unexpected columns in {path}")
            rows = list(reader)
        artifact_rows[name] = rows
        if len(rows) != int(record["rows"]):
            raise RuntimeError(f"row-count mismatch in {path}")
        split_labels: dict[str, Counter[str]] = defaultdict(Counter)
        local_groups: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            split_labels[row["split"]][row["label"]] += 1
            local_groups[row["source_group"]].add(row["split"])
            previous = global_group_split.setdefault(row["source_group"], row["split"])
            if previous != row["split"]:
                raise RuntimeError(
                    f"source group {row['source_group']} has inconsistent global splits"
                )
        if set(split_labels) != {"train", "validation", "test"}:
            raise RuntimeError(f"incomplete split set in {path}")
        for split, labels in split_labels.items():
            if labels["OR"] != labels["CG"] or set(labels) != {"OR", "CG"}:
                raise RuntimeError(f"{path}: {split} is not exactly OR/CG balanced")
        if any(len(splits) != 1 for splits in local_groups.values()):
            raise RuntimeError(f"source groups cross partitions in {path}")
        train_groups = {
            row["source_group"] for row in rows if row["split"] == "train"
        }
        test_groups = {row["source_group"] for row in rows if row["split"] == "test"}
        if train_groups & test_groups:
            raise RuntimeError(f"training/test source leakage in {path}")

    for generator in ("lfm", "deepseek", "glm"):
        rows = artifact_rows[f"matched_{generator}"]
        if any(
            row["generator"] != ("human" if row["label"] == "OR" else generator)
            for row in rows
        ):
            raise RuntimeError(f"matched_{generator} has incorrect generator labels")

        logo_rows = artifact_rows[f"logo_wo_{generator}"]
        training_generators = {
            row["generator"]
            for row in logo_rows
            if row["split"] == "train" and row["label"] == "CG"
        }
        expected_training = {"lfm", "deepseek", "glm"} - {generator}
        if training_generators != expected_training:
            raise RuntimeError(
                f"logo_wo_{generator} trains on {training_generators}, "
                f"expected {expected_training}"
            )

    train_groups = {group for group, split in global_group_split.items() if split == "train"}
    test_groups = {group for group, split in global_group_split.items() if split == "test"}
    if train_groups & test_groups:
        raise RuntimeError("global directional train/test source overlap is nonzero")
    print(
        "datasets: 6 hash-verified balanced artifacts; "
        f"{len(global_group_split)} source groups; zero train/test overlap"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completed-grid", default=None)
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="validate only the 18 directional/LOGO cells, not the 27-run plan",
    )
    args = parser.parse_args()

    validate_datasets()

    directional = lines(GRIDS / "source_grouped_directional.txt")
    logo = lines(GRIDS / "source_grouped_logo.txt")
    if (len(directional), len(logo)) != (9, 9):
        raise RuntimeError("core grids must contain 9 directional and 9 LOGO runs")
    for line in directional + logo:
        validate_line(line)

    expected_core = set(directional + logo)
    for mode, expected, count in (("core", expected_core, 6),):
        shard_lines = []
        for instance_id in (1, 2, 3):
            path = SHARDS / f"instance_{instance_id}_{mode}.txt"
            current = lines(path)
            if len(current) != count:
                raise RuntimeError(f"{path} has {len(current)} runs, expected {count}")
            shard_lines.extend(current)
        duplicates = [line for line, n in Counter(shard_lines).items() if n != 1]
        if duplicates:
            raise RuntimeError(f"{mode} shards contain duplicate runs: {duplicates}")
        if set(shard_lines) != expected:
            missing = expected - set(shard_lines)
            extra = set(shard_lines) - expected
            raise RuntimeError(
                f"{mode} shard union mismatch: missing={len(missing)}, extra={len(extra)}"
            )
        print(f"{mode}: {len(shard_lines)} unique runs across 3 balanced shards")

    for path in sorted(SHARDS.glob("instance_*_core.txt")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.relative_to(ROOT)}  sha256={digest}")

    if not args.core_only:
        rank = lines(GRIDS / "source_grouped_rank_fixed_scale.txt")
        if len(rank) != 9:
            raise RuntimeError("controlled-rank grid must contain 9 runs")
        for line in rank:
            validate_line(line)
        rank_cells = Counter()
        for line in rank:
            parsed = options(line)
            rank_value = int(str(parsed["rank"]))
            alpha = int(str(parsed.get("alpha", 2 * rank_value)))
            scale = alpha / (rank_value ** 0.5)
            if abs(scale - 16.0) > 1e-12:
                raise RuntimeError(f"fixed-scale rank line has scale {scale}: {line}")
            rank_cells[(rank_value, alpha)] += 1
        if rank_cells != Counter({(16, 64): 3, (36, 96): 3, (64, 128): 3}):
            raise RuntimeError(f"unexpected fixed-scale rank cells: {rank_cells}")
        expected_full = expected_core | set(rank)
        full_lines = []
        for instance_id in (1, 2, 3):
            path = SHARDS / f"instance_{instance_id}_full.txt"
            current = lines(path)
            if len(current) != 9:
                raise RuntimeError(f"{path} has {len(current)} runs, expected 9")
            full_lines.extend(current)
        if len(set(full_lines)) != 27 or set(full_lines) != expected_full:
            raise RuntimeError("optional full shard union does not match the 27-run plan")
        print("full plan: 27 unique runs across 3 balanced nine-run shards")

    if args.completed_grid:
        completed_grid = Path(args.completed_grid)
        if not completed_grid.is_absolute():
            completed_grid = ROOT / completed_grid
        validate_completed(completed_grid)


if __name__ == "__main__":
    main()
