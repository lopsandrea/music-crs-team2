"""recommender — conversation → top-20 track recommendations (RecSys Challenge 2026).

The full retrieval+ranking architecture, ported as a clean, integrated module:
candidate sources (BM25 + qwen3 + CF-BPR + ALS + text_retriever + struct_base +
struct_large) → weighted RRF pool → base/large LambdaRank with selective routing → top-20.

    from recommender import Recommender
    from recommender.data import load_blind_sessions

    rec = Recommender()
    out = rec.batch_recommend(load_blind_sessions("blind_b"))
    # out[i] = {session_id, turn_number, predicted_track_ids[20], recommend_item, ...}

From-scratch training of the shipped artifacts arrives in the follow-up
reproducibility commits (see README).
"""
from __future__ import annotations

# Re-export the package facade so callers can `from recommender import Recommender`
# without reaching into the recommender.recommender submodule. `Recommender` wires up the
# L1-L6 pipeline (sources -> RRF fusion -> features -> routed LambdaRank -> CatBoost ensemble).
from .recommender import Recommender

__all__ = ["Recommender"]
