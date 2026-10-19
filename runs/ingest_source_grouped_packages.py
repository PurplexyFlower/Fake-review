"""Verify and merge three source-grouped cloud packages into local results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

from source_grouped_common import ROOT, sha256


RESULTS = ROOT / "results"


def safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe archive path: {name}")
    return path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_verified(destination: Path, data: bytes) -> None:
    if destination.exists():
        if sha256(destination) != digest(data):
            raise RuntimeError(f"existing file differs from package: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def merge_ledger(data: bytes, allowed_run_ids: set[str]) -> None:
    incoming = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    incoming = [row for row in incoming if row["run_id"] in allowed_run_ids]
    if len(incoming) != len(allowed_run_ids):
        found = {row["run_id"] for row in incoming}
        raise RuntimeError(f"package ledger missing run ids: {sorted(allowed_run_ids - found)}")

    ledger = RESULTS / "runs.csv"
    with ledger.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        current = list(reader)
    if fieldnames is None:
        raise RuntimeError("local results/runs.csv has no header")
    by_id = {row["run_id"]: row for row in current}
    additions = []
    for row in incoming:
        existing = by_id.get(row["run_id"])
        if existing is not None and existing != row:
            raise RuntimeError(f"ledger conflict for {row['run_id']}")
        if existing is None:
            additions.append(row)
    if additions:
        with ledger.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerows(additions)


def ingest(package: Path) -> int:
    companion = package.with_suffix(package.suffix + ".sha256")
    if companion.exists():
        expected_package_hash = companion.read_text(encoding="utf-8").split()[0]
        actual_package_hash = sha256(package)
        if actual_package_hash != expected_package_hash:
            raise RuntimeError(
                f"{package}: package SHA-256 mismatch "
                f"({actual_package_hash} != {expected_package_hash})"
            )
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        if "package_manifest.json" not in names:
            raise RuntimeError(f"{package}: package_manifest.json is missing")
        manifest_data = archive.read("package_manifest.json")
        manifest = json.loads(manifest_data)
        if int(manifest.get("schema_version", -1)) != 1:
            raise RuntimeError(f"{package}: unsupported package schema")
        instance_id = int(manifest["instance_id"])
        run_ids = set(manifest["run_ids"])
        if len(run_ids) != 9:
            raise RuntimeError(f"{package}: expected nine unique run ids")

        recorded = {record["path"]: record for record in manifest["files"]}
        if set(recorded) | {"package_manifest.json"} != names:
            raise RuntimeError(f"{package}: archive contents do not match manifest")
        payload: dict[str, bytes] = {}
        for name, record in recorded.items():
            safe_name(name)
            data = archive.read(name)
            if len(data) != int(record["bytes"]) or digest(data) != record["sha256"]:
                raise RuntimeError(f"{package}: hash or size mismatch for {name}")
            payload[name] = data

    local_grid = ROOT / f"grids/source_grouped/instance_{instance_id}_full.txt"
    if digest(payload["metadata/grid.txt"]) != sha256(local_grid):
        raise RuntimeError(f"{package}: grid differs from the local full shard")
    local_source_manifest = ROOT / "dataset/source_grouped/manifest.json"
    if digest(payload["metadata/source_grouped_manifest.json"]) != sha256(
        local_source_manifest
    ):
        raise RuntimeError(f"{package}: source-grouped manifest differs from local")
    environment = json.loads(payload["metadata/environment.json"])
    for relative, expected_hash in environment.get("code_sha256", {}).items():
        local_code = ROOT / relative
        if not local_code.exists() or sha256(local_code) != expected_hash:
            raise RuntimeError(
                f"{package}: local code differs from cloud evidence for {relative}"
            )

    for name, data in payload.items():
        path = safe_name(name)
        if path.parts[0] == "results" and len(path.parts) >= 3:
            if path.parts[1] not in run_ids:
                raise RuntimeError(f"{package}: unexpected result directory {path.parts[1]}")
            write_verified(ROOT.joinpath(*path.parts), data)

    metadata_dir = RESULTS / "cloud_packages"
    write_verified(
        metadata_dir / f"instance_{instance_id}_environment.json",
        payload["metadata/environment.json"],
    )
    write_verified(
        metadata_dir / f"instance_{instance_id}_package_manifest.json",
        manifest_data,
    )
    write_verified(
        metadata_dir / f"instance_{instance_id}_source_manifest.json",
        payload["metadata/source_grouped_manifest.json"],
    )
    merge_ledger(payload["metadata/runs.csv"], run_ids)
    print(f"verified and ingested instance {instance_id}: {len(run_ids)} runs")
    return instance_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packages", nargs="+")
    args = parser.parse_args()
    packages = [Path(value).resolve() for value in args.packages]
    instances = [ingest(package) for package in packages]
    if len(packages) == 3 and set(instances) != {1, 2, 3}:
        raise RuntimeError(f"expected instance ids 1, 2, 3; got {instances}")


if __name__ == "__main__":
    main()
