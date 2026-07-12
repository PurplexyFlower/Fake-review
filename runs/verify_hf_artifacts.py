"""Verify a published adapter artifact repository against its local manifest."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lfs_value(lfs, key):
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get(key)
    return getattr(lfs, key, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--manifest", default=str(ROOT / "results" / "adapter_manifest.json"))
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    tree = HfApi().list_repo_tree(
        args.repo, repo_type="model", recursive=True, expand=True)
    remote = {item.path: item for item in tree if hasattr(item, "size")}
    required_top = {
        "README.md", "PAPER_FINDINGS.md", "adapter_manifest.json", "runs.csv",
        "cross_gen_results.json", "baselines_transformer.csv", "head_to_head.log",
        "external_eval/m1_on_1p2b.json",
    }
    absent_top = sorted(required_top - remote.keys())
    if absent_top:
        raise RuntimeError(f"missing top-level evidence: {absent_top}")

    verified_lfs = 0
    expected_files = []
    for entry in manifest["adapters"]:
        run_id = entry["run_id"]
        for suffix in ("config.json", "metrics.json"):
            path = f"{run_id}/{suffix}"
            if path not in remote:
                raise RuntimeError(f"missing remote file: {path}")
        for record in entry["files"]:
            path = record["path"]
            expected_files.append(path)
            item = remote.get(path)
            if item is None:
                raise RuntimeError(f"missing remote adapter file: {path}")
            if item.size != record["size"]:
                raise RuntimeError(
                    f"size mismatch for {path}: {item.size} != {record['size']}")
            remote_sha = lfs_value(getattr(item, "lfs", None), "sha256")
            if remote_sha is not None:
                if remote_sha != record["sha256"]:
                    raise RuntimeError(
                        f"LFS SHA-256 mismatch for {path}: "
                        f"{remote_sha} != {record['sha256']}")
                verified_lfs += 1

    downloaded = Path(hf_hub_download(
        args.repo, "adapter_manifest.json", repo_type="model", force_download=True))
    if sha256(downloaded) != sha256(manifest_path):
        raise RuntimeError("downloaded manifest differs from the local publication manifest")

    safetensors = [path for path in expected_files
                   if path.endswith("adapter_model.safetensors")]
    print(f"remote files: {len(remote)}")
    print(f"adapters: {manifest['saved_adapter_count']}")
    print(f"adapter files: {len(expected_files)}")
    print(f"adapter safetensors: {len(safetensors)}")
    print(f"LFS SHA-256 verified: {verified_lfs}")
    print("manifest re-download: exact match")


if __name__ == "__main__":
    main()
