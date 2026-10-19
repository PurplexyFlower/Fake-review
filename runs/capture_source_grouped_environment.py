"""Capture non-secret environment and artifact identity for one cloud shard."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "unsloth",
    "unsloth_zoo",
    "datasets",
    "accelerate",
    "trl",
    "pyarrow",
    "numpy",
    "scikit-learn",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            args, check=True, capture_output=True, text=True, timeout=20
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--offer-id", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    grid = Path(args.grid)
    if not grid.is_absolute():
        grid = ROOT / grid
    source_manifest = ROOT / "dataset/source_grouped/manifest.json"
    output = Path(args.output) if args.output else (
        ROOT / "results" / f"source_grouped_instance_{args.instance_id}_environment.json"
    )
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    record = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "instance_id": args.instance_id,
        "provider_offer_id": args.offer_id,
        "grid": grid.resolve().relative_to(ROOT.resolve()).as_posix(),
        "grid_sha256": sha256(grid),
        "source_grouped_manifest_sha256": sha256(source_manifest),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": versions,
        "nvidia_smi": command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "code_sha256": {
            path: sha256(ROOT / path)
            for path in (
                "runs/train_run.py",
                "runs/run_queue.py",
                "runs/build_source_grouped_datasets.py",
                "runs/validate_source_grouped_plan.py",
                "runs/evaluate_source_grouped.py",
                "runs/source_grouped_common.py",
                "runs/package_source_grouped_results.py",
                "runs/capture_source_grouped_environment.py",
            )
        },
    }
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"environment -> {output}")


if __name__ == "__main__":
    main()
