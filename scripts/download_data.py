#!/usr/bin/env python3
"""Pre-fetch every EXTERNAL dataset/checkpoint the pipeline needs (all public, no auth).

Everything here is also downloaded automatically on first use; this script just makes the
external material explicit and lets you populate the HuggingFace cache up front (e.g. to
run on an air-gapped GPU node afterwards, or to audit exactly what is fetched).

    uv run python scripts/download_data.py               # inference-time datasets (~150 MB)
    uv run python scripts/download_data.py --training    # + training data & base encoders (~3.5 GB)

Inference (deliverable 1):
  - talkpl-ai/TalkPlayData-Challenge-Track-Metadata   (catalog; all_tracks split)
  - talkpl-ai/TalkPlayData-Challenge-Blind-B          (the sessions to score)

Training / from-scratch reproduction (deliverable 2, --training):
  - talkpl-ai/TalkPlayData-Challenge-Dataset          (train conversations + the 8000 dev cases)
  - talkpl-ai/TalkPlayData-Challenge-Track-Embeddings (precomputed qwen3 / cf-bpr / CLAP columns)
  - talkpl-ai/TalkPlayData-Challenge-Blind-A          (adversarial-weights reference split)
  - BAAI/bge-base-en-v1.5, BAAI/bge-large-en-v1.5     (base encoders the trainers fine-tune)

The model weights / artifact cache of THIS repo is separate — see scripts/download_weights.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INFERENCE_DATASETS = [
    "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    "talkpl-ai/TalkPlayData-Challenge-Blind-B",
]
TRAINING_DATASETS = [
    "talkpl-ai/TalkPlayData-Challenge-Dataset",
    "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
    "talkpl-ai/TalkPlayData-Challenge-Blind-A",
]
TRAINING_CHECKPOINTS = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training", action="store_true",
                    help="also fetch the training datasets and base encoder checkpoints")
    args = ap.parse_args()

    from datasets import load_dataset
    names = INFERENCE_DATASETS + (TRAINING_DATASETS if args.training else [])
    for name in names:
        print(f"[dataset] {name} …", flush=True)
        ds = load_dataset(name)   # downloads into the HF cache; no-op when already cached
        print(f"          splits: {list(ds)}")

    if args.training:
        from huggingface_hub import snapshot_download
        for name in TRAINING_CHECKPOINTS:
            print(f"[checkpoint] {name} …", flush=True)
            snapshot_download(repo_id=name)

    print("done — everything cached under the HuggingFace cache "
          "(HF_HOME to relocate it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
