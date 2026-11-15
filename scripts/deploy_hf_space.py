"""Publish the current commit to a Hugging Face Space (Docker SDK).

    hf auth login                                   # once; paste a WRITE token, never commit it
    python -m scripts.deploy_hf_space --space <hf-username>/<space-name>

What it does:

1. Collects the files tracked by git (``git ls-files``), so untracked or ignored files
   (``.env``, ``data/uploads``, ``models``, local experiments) are never uploaded.
2. Copies them to a temporary staging folder and puts the Space configuration (YAML front
   matter: ``sdk: docker``, ``app_port: 7860``) on top of README.md there. The repository's
   own README is not modified.
3. Creates the Space if it does not exist and uploads the folder as one commit. Files that
   exist in the Space but no longer in the repo are deleted, so the Space mirrors HEAD.

API keys are NOT uploaded: add GEMINI_API_KEY or ANTHROPIC_API_KEY as a *secret* in the
Space settings (Settings -> Variables and secrets). Without one, answers use the offline
extractive fallback.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.config import PROJECT_ROOT

SPACE_FRONT_MATTER = """---
title: Enterprise Documentation Q&A Assistant
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Grounded answers with citations over enterprise documents
---

"""


def tracked_files(root: Path) -> list[Path]:
    """Files tracked by git (committed or staged), relative to ``root``. Edits are included as they
    are on disk, so commit first if the Space must match a specific commit."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout
    return [Path(p) for p in out.decode("utf-8").split("\0") if p and (root / p).is_file()]


def stage(root: Path, target: Path) -> int:
    """Copy tracked files into ``target`` and prepend the Space front matter to README.md."""
    files = tracked_files(root)
    for rel in files:
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, dest)
    readme = target / "README.md"
    body = readme.read_text(encoding="utf-8") if readme.exists() else ""
    readme.write_text(SPACE_FRONT_MATTER + body, encoding="utf-8")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish this repo to a Hugging Face Space (Docker).")
    parser.add_argument("--space", required=True, help="<hf-username>/<space-name>")
    parser.add_argument("--private", action="store_true", help="create the Space as private")
    parser.add_argument("--dry-run", action="store_true", help="stage files and list them, upload nothing")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="hf-space-") as tmp:
        staging = Path(tmp)
        count = stage(PROJECT_ROOT, staging)
        print(f"staged {count} tracked files")
        if args.dry_run:
            for path in sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file()):
                print("  ", path)
            return

        from huggingface_hub import HfApi  # installed with sentence-transformers

        api = HfApi()
        api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
        commit = api.upload_folder(folder_path=staging, repo_id=args.space, repo_type="space",
                                   delete_patterns="*", commit_message="Deploy from local repository")
        print(f"uploaded: {commit.commit_url}")
        print(f"Space:    https://huggingface.co/spaces/{args.space}  (the build takes a few minutes)")


if __name__ == "__main__":
    main()
