#!/usr/bin/env python3
"""Verify the recommendation half against the submitted reference payload.

Runs the shipped Recommender over all Blind-B sessions and checks that every
session's `predicted_track_ids` (top-20) EXACTLY matches the reference
`reference/prediction_827190.json` (the payload of our validated Codabench
submission). The recommendation half is fully deterministic, so this must
report 80/80 — no Gemini API key is needed (responses are not generated here).

    uv run python scripts/verify_inference.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    reference = json.loads((ROOT / "reference" / "prediction_827190.json").read_text())
    ref_by_sid = {r["session_id"]: r["predicted_track_ids"] for r in reference}

    from recommender import Recommender
    from recommender.data import load_blind_sessions

    sessions = load_blind_sessions("blind_b")
    print(f"loaded {len(sessions)} blind_b sessions; running the recommender…", flush=True)
    rec = Recommender()
    out = rec.batch_recommend(sessions)

    ok, bad = 0, []
    for o in out:
        sid = o["session_id"]
        if o["predicted_track_ids"] == ref_by_sid.get(sid):
            ok += 1
        else:
            bad.append(sid)
    print(f"\nmatched {ok}/{len(out)} sessions against reference/prediction_827190.json")
    if bad:
        print("MISMATCHED session_ids:", bad)
        return 1
    print("OK: recommendation half reproduces the submitted payload exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
