#!/usr/bin/env python3
"""(Maintainers) Upload ./cache to the public Hugging Face weights repo.

    huggingface-cli login   # once
    uv run python scripts/upload_weights.py --repo USER/REPO [--create]

Uploads the artifact tree verbatim (same layout the code reads) and stamps the
repo id into weights_manifest.json so scripts/download_weights.py finds it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
MANIFEST = ROOT / "weights_manifest.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF dataset repo id, e.g. team2s2/music-crs-weights")
    ap.add_argument("--create", action="store_true", help="create the (public) repo first")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    if args.create:
        api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
    # Manifest-driven upload: ONLY files listed in weights_manifest.json are shipped, so
    # runtime byproducts under cache/ (e.g. new response-cache entries) can never leak.
    manifest_files = list(json.loads(MANIFEST.read_text())["files"])
    print(f"uploading {len(manifest_files)} manifest files from {CACHE} → "
          f"https://huggingface.co/datasets/{args.repo} …", flush=True)
    api.upload_folder(folder_path=str(CACHE), repo_id=args.repo, repo_type="dataset",
                      allow_patterns=manifest_files)

    manifest = json.loads(MANIFEST.read_text())
    manifest["weights_repo"] = args.repo
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"done; weights_repo={args.repo} stamped into weights_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
