#!/usr/bin/env python3
"""Download the inference weights/artifact cache (~1 GB) into ./cache and verify it.

Default source is the public Hugging Face weights repo (see README). A direct
zip URL can be used as fallback with --url.

    uv run python scripts/download_weights.py                  # from the HF repo in the manifest
    uv run python scripts/download_weights.py --repo USER/REPO # explicit HF repo
    uv run python scripts/download_weights.py --url https://…/weights.zip

Every downloaded file is checked against the sha256 in weights_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
MANIFEST = ROOT / "weights_manifest.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify() -> int:
    manifest = json.loads(MANIFEST.read_text())
    bad, missing = [], []
    for rel, meta in manifest["files"].items():
        p = CACHE / rel
        if not p.exists():
            missing.append(rel)
        elif sha256_of(p) != meta["sha256"]:
            bad.append(rel)
    if missing or bad:
        for r in missing:
            print(f"MISSING  cache/{r}")
        for r in bad:
            print(f"BAD SHA  cache/{r}")
        return 1
    print(f"OK: {len(manifest['files'])} files verified against weights_manifest.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="HF repo id (default: manifest's weights_repo)")
    ap.add_argument("--url", default=None, help="direct zip URL fallback")
    ap.add_argument("--verify-only", action="store_true", help="only verify an existing cache/")
    args = ap.parse_args()

    if args.verify_only:
        return verify()

    if args.url:
        import urllib.request
        tmp = ROOT / "weights.zip"
        print(f"downloading {args.url} → {tmp} …", flush=True)
        urllib.request.urlretrieve(args.url, tmp)
        print("extracting …", flush=True)
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(ROOT)   # archive stores cache/-prefixed paths
        tmp.unlink()
        return verify()

    repo = args.repo or json.loads(MANIFEST.read_text()).get("weights_repo")
    if not repo:
        print("no --repo/--url given and no weights_repo in the manifest", file=sys.stderr)
        return 2
    from huggingface_hub import snapshot_download
    print(f"downloading weights from https://huggingface.co/datasets/{repo} → {CACHE} …", flush=True)
    # workspace.tar.gz is a convenience snapshot of the whole repo for the validators;
    # the artifact tree itself is what inference needs, so skip the tarball here.
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=CACHE,
                      ignore_patterns=["workspace.tar.gz"])
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
