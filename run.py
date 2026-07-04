#!/usr/bin/env python3
"""Single entrypoint for the RecSys-2026 Music-CRS inference pipeline.

  uv run python run.py infer [--blind blind_b] [--limit N] [--out submission.zip]
  uv run python run.py train-all [--smoke] [--force]

`infer`     : blind sessions → recommender → response_gen → submission.zip.
`train-all` : rebuild every cached artifact from scratch (see TRAINING.md).

The recommendation half (predicted_track_ids) is fully deterministic given the shipped
artifacts; verify it against the reference payload with `scripts/verify_inference.py`
(no API key needed). The response half calls the Gemini API and requires
GEMINI_API_KEY / GOOGLE_API_KEY in the environment (see README).
"""
# NOTE: the module docstring above must stay the VERY FIRST statement, i.e. ABOVE the
# `from __future__` line below. A __future__ import is only legal at the top of a module
# (after at most the docstring), so swapping the order would make this file fail to import.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap sys.path so the sibling packages (pipeline, recommender, response_gen) import
# as top-level modules regardless of the caller's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("infer", help="blind sessions → recommender → responses → submission.zip")
    p.add_argument("--blind", default="blind_b")
    p.add_argument("--out", default="submission.zip")
    # --limit caps the number of sessions processed (quick smoke run); None = all.
    p.add_argument("--limit", type=int, default=None)
    # Response-half (Level 7) knobs: which generator class, model, and prompt variant.
    # "refiner" = Gemini Pro draft -> Flash polish (the shipped configuration).
    p.add_argument("--gemini-class", default="refiner")
    p.add_argument("--model", default="gemini-3.1-pro-preview")
    p.add_argument("--prompt", default="blindbnat")
    # Track-fact enrichment is ON by default (part of the shipped response configuration).
    p.add_argument("--enrich-items", dest="enrich_items", action="store_true", default=True,
                   help="enable track-fact enrichment (default ON)")
    p.add_argument("--no-enrich-items", dest="enrich_items", action="store_false",
                   help="disable track-fact enrichment")

    pt = sub.add_parser("train-all", help="rebuild every cached artifact from scratch (TRAINING.md)")
    # --smoke = tiny fast pass to validate the wiring (not a real artifact); --force =
    # rebuild even when a cached artifact already exists. Both map to train_all(...) kwargs.
    pt.add_argument("--smoke", action="store_true")
    pt.add_argument("--force", action="store_true")

    args = ap.parse_args()

    if args.cmd == "infer":
        from pipeline import run_pipeline
        res = run_pipeline(
            blind_name=args.blind, out_zip=args.out, limit=args.limit,
            gemini_class=args.gemini_class, model_name=args.model, prompt_variant=args.prompt,
            enrich_items=args.enrich_items,
        )
        print("RESULT:", res)
    elif args.cmd == "train-all":
        from training.train_all import train_all
        train_all(smoke=args.smoke, force=args.force)


if __name__ == "__main__":
    main()
