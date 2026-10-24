"""Stage gated Cosmos weights into a persistent Hugging Face cache."""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", nargs="?", default="nvidia/Cosmos3-Super")
    parser.add_argument("--revision", default=None, help="HF commit hash for reproducible runs")
    parser.add_argument("--local-dir", default=None)
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        parser.error("HF_TOKEN is required; accept the model's Hugging Face access terms first")
    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        token=token,
        local_dir=args.local_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
