"""Weighted Reciprocal Rank Fusion (Level 2).

Merge the per-source ranked candidate lists (Level 1) into one recall pool. RRF
combines lists using only each item's RANK position, never its raw score — which
is why sources with wildly incomparable score scales (BM25 tf-idf sums, cosine in
[-1, 1], ALS dot products) can be fused without any re-calibration, and why a new
modality like the acoustic clap_recent can join the pool for free.
"""
from __future__ import annotations


def weighted_rrf(sources: dict[str, list[str]], weights: dict[str, float],
                 topk: int, k: int = 20) -> list[str]:
    """Weighted Reciprocal Rank Fusion of several ranked track-id lists.

    Each item accumulates `score = sum over sources  weight / (k + rank)` across
    every source list it appears in (rank is 1-based). Returns the `topk` ids by
    descending fused score — the recall pool (POOL_K=300 here).

    The constant `k` (RRF_K=20 in config) damps the influence of the very top
    ranks: with k=20, rank 1 contributes 1/21 and rank 2 1/22 — close together,
    so no single source can unilaterally force an item to the top; agreement
    ACROSS sources is what accumulates. The per-source `weight` scales that
    contribution (see config.SW_BASELINE for the shipped weights).

    Args:
        sources: {source_name: ranked_track_id_list}.
        weights: {source_name: fusion_weight}; sources absent here default to 0.
        topk:    size of the returned pool.
        k:       the RRF damping constant.
    """
    scores: dict[str, float] = {}
    for name, ranked in sources.items():
        w = weights.get(name, 0.0)
        # Skip zero-weighted (e.g. disabled clap_recent) or empty source lists.
        if w == 0 or not ranked:
            continue
        for rank, tid in enumerate(ranked, start=1):
            # Accumulate this source's reciprocal-rank contribution for the item.
            scores[tid] = scores.get(tid, 0.0) + w / (k + rank)
    # Descending by fused score; truncate to the pool size. (Python's sort is stable,
    # so ties keep first-seen order — deterministic for a fixed source iteration order.)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:topk]
