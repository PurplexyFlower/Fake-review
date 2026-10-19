"""Create a hash-manifested package for one completed cloud shard."""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from source_grouped_common import ROOT, find_completed_run, grid_lines, sha256


def add_file(
    archive: zipfile.ZipFile,
    source: Path,
    archive_name: str,
    records: list[dict[str, object]],
) -> None:
    archive.write(source, archive_name)
    records.append(
        {
            "path": archive_name,
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    grid = Path(args.grid)
    if not grid.is_absolute():
        grid = ROOT / grid
    run_dirs = [find_completed_run(line, require_evaluation=True) for line in grid_lines(grid)]
    if len(run_dirs) != 9:
        raise RuntimeError(
            f"full cloud shard must contain nine completed runs, got {len(run_dirs)}"
        )

    output = Path(args.output) if args.output else (
        ROOT / "results" / f"source_grouped_instance_{args.instance_id}_package.zip"
    )
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    environment = ROOT / "results" / f"source_grouped_instance_{args.instance_id}_environment.json"
    required_metadata = [
        environment,
        ROOT / "dataset/source_grouped/manifest.json",
        ROOT / "results/runs.csv",
    ]
    missing = [str(path) for path in required_metadata if not path.exists()]
    if missing:
        raise RuntimeError(f"missing package metadata: {missing}")

    records: list[dict[str, object]] = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        add_file(archive, grid, "metadata/grid.txt", records)
        add_file(archive, environment, "metadata/environment.json", records)
        add_file(
            archive,
            ROOT / "dataset/source_grouped/manifest.json",
            "metadata/source_grouped_manifest.json",
            records,
        )
        add_file(archive, ROOT / "results/runs.csv", "metadata/runs.csv", records)

        run_ids = []
        for run_dir in run_dirs:
            run_ids.append(run_dir.name)
            for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
                relative = path.relative_to(ROOT).as_posix()
                add_file(archive, path, relative, records)

        queue_logs = ROOT / "results/queue_logs"
        if queue_logs.exists():
            prefixes = tuple(run_dir.name.rsplit("_", 1)[0] for run_dir in run_dirs)
            for path in sorted(queue_logs.glob("*.log")):
                if any(prefix in path.name for prefix in prefixes):
                    add_file(
                        archive,
                        path,
                        f"metadata/queue_logs/{path.name}",
                        records,
                    )

        package_manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "instance_id": args.instance_id,
            "grid_sha256": sha256(grid),
            "run_ids": sorted(run_ids),
            "files": records,
        }
        archive.writestr(
            "package_manifest.json",
            json.dumps(package_manifest, indent=2).encode("utf-8") + b"\n",
        )

    package_hash = sha256(output)
    hash_path = output.with_suffix(output.suffix + ".sha256")
    hash_path.write_text(f"{package_hash}  {output.name}\n", encoding="utf-8")
    print(f"package -> {output}")
    print(f"sha256  -> {package_hash}")


if __name__ == "__main__":
    main()
