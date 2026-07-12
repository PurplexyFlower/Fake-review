"""Build an immutable publication manifest for saved PEFT adapters.

The manifest keeps historical run records untouched. It hashes the frozen
dataset currently named by each run config and every file in the saved adapter,
and reports ledger rows whose weights are unavailable.

  python runs/build_adapter_manifest.py \
    --tags m1,m1s,xg,xl,xd,xglm,pwogpt2,pwolfm,pwodeepseek,pwoglm,ctrl_orperm \
    --output results/adapter_manifest.json
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_data_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def configuration_key(record: dict) -> tuple:
    """Identity of a reproducible training configuration, independent of run ID."""
    return (record["tag"], record["scaling"], int(record["rank"]),
            int(record["seed"]), record.get("holdout_category") or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", required=True,
                        help="comma-separated ledger tags in publication scope")
    parser.add_argument("--output", default=str(RESULTS / "adapter_manifest.json"))
    args = parser.parse_args()
    tags = {tag.strip() for tag in args.tags.split(",") if tag.strip()}

    with (RESULTS / "runs.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["tag"] in tags]
    ledger = {row["run_id"]: row for row in rows}

    entries = []
    for adapter_dir in sorted(RESULTS.glob("*/adapter")):
        run_dir = adapter_dir.parent
        if run_dir.name not in ledger:
            continue
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        required = [config_path, metrics_path,
                    adapter_dir / "adapter_config.json",
                    adapter_dir / "adapter_model.safetensors"]
        missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"incomplete adapter {run_dir.name}: {missing}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        data_path = resolve_data_path(config.get("data"))
        dataset = None
        if data_path and data_path.is_file():
            dataset = {
                "path": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT)
                        else str(data_path),
                "sha256": sha256(data_path),
                "recorded_sha256": config.get("data_sha256"),
            }

        files = []
        for path in sorted(p for p in adapter_dir.rglob("*") if p.is_file()):
            files.append({
                "path": str(path.relative_to(RESULTS)).replace("\\", "/"),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            })
        entries.append({
            "run_id": run_dir.name,
            "tag": config["tag"],
            "scaling": config["scaling"],
            "rank": config["rank"],
            "alpha": config["alpha"],
            "seed": config["seed"],
            "dataset": dataset,
            "metrics": metrics,
            "files": files,
        })

    saved_ids = {entry["run_id"] for entry in entries}
    saved_keys = {configuration_key(entry) for entry in entries}
    historical_fields = ("run_id", "tag", "scaling", "rank", "seed")
    unavailable = [
        {key: row[key] for key in historical_fields}
        for row in rows
        if row["run_id"] not in saved_ids and configuration_key(row) not in saved_keys
    ]
    superseded = [
        {key: row[key] for key in historical_fields}
        for row in rows
        if row["run_id"] not in saved_ids and configuration_key(row) in saved_keys
    ]
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_model": "unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit",
        "publication_tags": sorted(tags),
        "saved_adapter_count": len(entries),
        "unavailable_weight_count": len(unavailable),
        "unavailable_weights": unavailable,
        "superseded_metric_only_count": len(superseded),
        "superseded_metric_only_runs": superseded,
        "adapters": entries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"saved adapters: {len(entries)}")
    print(f"unavailable weights: {len(unavailable)}")
    print(f"manifest: {output}")


if __name__ == "__main__":
    main()
