"""Shared helpers for the strict contemporary source-grouped experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE_GROUPED = ROOT / "dataset" / "source_grouped"
GENERATORS = ("lfm", "deepseek", "glm")

DIRECTIONAL_TAGS = {
    "sgxl": "lfm",
    "sgxd": "deepseek",
    "sgxglm": "glm",
}
LOGO_TAGS = {
    "sgpwolfm": "lfm",
    "sgpwodeepseek": "deepseek",
    "sgpwoglm": "glm",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_options(line: str) -> dict[str, str | bool]:
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


def data_path(options: dict[str, str | bool]) -> Path:
    path = Path(str(options["data"]))
    return path if path.is_absolute() else ROOT / path


def run_prefix(options: dict[str, str | bool]) -> str:
    return (
        f"{options['tag']}_{options['scaling']}_r{options['rank']}_s{options['seed']}"
    )


def find_completed_run(line: str, *, require_evaluation: bool = False) -> Path:
    """Find the newest complete run matching one exact grid line.

    This intentionally verifies the config and dataset bytes without relying on
    the shared ledger, because each cloud instance has its own appended ledger.
    """
    options = parse_options(line)
    path = data_path(options)
    expected_hash = sha256(path)
    rank = int(str(options["rank"]))
    expected = {
        "tag": str(options["tag"]),
        "scaling": str(options["scaling"]),
        "rank": rank,
        "alpha": int(str(options.get("alpha", 2 * rank))),
        "seed": int(str(options["seed"])),
        "scheduler": str(options.get("scheduler", "cosine_with_restarts")),
        "label_smoothing": float(str(options.get("label-smoothing", 0.01))),
        "holdout_category": options.get("holdout-category"),
        "predefined_splits": options.get("predefined-splits") is True,
    }

    matches: list[Path] = []
    for run_dir in sorted(RESULTS.glob(f"{run_prefix(options)}_*")):
        config_path = run_dir / "config.json"
        required = [
            config_path,
            run_dir / "metrics.json",
            run_dir / "test_probs.csv",
            run_dir / "adapter" / "adapter_config.json",
            run_dir / "adapter" / "adapter_model.safetensors",
        ]
        if require_evaluation:
            required.append(run_dir / "source_grouped_eval.json")
        if not all(item.exists() for item in required):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not all(config.get(key) == value for key, value in expected.items()):
            continue
        if config.get("data_sha256") != expected_hash:
            continue
        matches.append(run_dir)
    if not matches:
        raise RuntimeError(
            f"no complete exact run found for {line}; expected prefix "
            f"{run_prefix(options)}"
        )
    return matches[-1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def target_dataset(generator: str) -> Path:
    if generator not in GENERATORS:
        raise ValueError(f"unknown contemporary generator: {generator}")
    return SOURCE_GROUPED / f"modern_reviews_{generator}.csv"


def test_rows(generator: str) -> list[dict[str, str]]:
    rows = [row for row in read_csv(target_dataset(generator)) if row["split"] == "test"]
    if not rows or {row["label"] for row in rows} != {"OR", "CG"}:
        raise RuntimeError(f"invalid source-grouped test partition for {generator}")
    return rows


def assert_no_source_leakage(training_data: Path, target: str) -> dict[str, int]:
    train_rows = [row for row in read_csv(training_data) if row["split"] == "train"]
    target_rows = test_rows(target)
    train_groups = {row["source_group"] for row in train_rows}
    test_groups = {row["source_group"] for row in target_rows}
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(
            f"source leakage from {training_data} into {target}: {len(overlap)} groups"
        )
    return {
        "training_source_groups": len(train_groups),
        "target_test_source_groups": len(test_groups),
        "overlapping_source_groups": 0,
    }
