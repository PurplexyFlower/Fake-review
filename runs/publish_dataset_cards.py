"""Publish and remotely verify all provenance-backed Hugging Face dataset cards.

The card text is rendered from dataset/provenance.json through push_to_hf.py, so
the live documentation and future full-dataset uploads share one implementation.
No dataset parquet files are modified by this command.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from push_to_hf import PROVENANCE, load_env, load_provenance, render_card


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(PROVENANCE))
    args = parser.parse_args()

    load_env()
    manifest = load_provenance(args.manifest)
    api = HfApi()
    identity = api.whoami()
    print(f"authenticated as: {identity['name']}")

    for key, record in manifest["datasets"].items():
        repo = record["hub"]["repo"]
        card = render_card(record)
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
            commit_message="Document exact generation provenance",
        )
        downloaded = Path(hf_hub_download(
            repo, "README.md", repo_type="dataset", force_download=True))
        if downloaded.read_text(encoding="utf-8") != card:
            raise RuntimeError(f"remote card verification failed: {repo}")
        print(f"verified {key}: https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
