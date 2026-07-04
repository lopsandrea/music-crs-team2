"""LambdaRank scoring + selective routing between the base and large rankers.

The base ranker's top1-top2 margin decides which LambdaRank booster orders the pool
(provenance: ported from the validated blind-replay pipeline).

This module covers Levels 4-6 of the pipeline (see README, "What this system is"):
  L4  score the pool with both LightGBM LambdaRank boosters and pick one by a
      margin-based "selective routing" rule.
  L5  optionally blend the routed LightGBM scores with the CatBoost-YetiRank
      ensemble in z-score space (the winning lever; equal weight CB_W=1.0).
  L6  argsort the final scores (stable mergesort), skip already-played tracks,
      return the top-TOP_K (=20) track_ids.
The two boosters differ ONLY in the last-3 feature columns (struct_base vs
struct_large triple), so they learn slightly different orderings of the same pool.

Naming note (locked legacy tokens): r54 == struct_base and r84 == struct_large.
The "base"/"large" pair here refers to those two fine-tuned BGE retrievers whose
per-source feature triples occupy the last 3 columns of feats_base / feats_large
respectively. The tokens are retained because they are baked into shipped on-disk
artifacts (LightGBM feature_name vectors, pickle keys, path strings) — renaming
would force a multi-hour GPU retrain.

Why the CatBoost half is a separate, decorrelated ranker (the L5 "winning lever"):
the CatBoost-YetiRank model is a gradient-boosted-tree ranker like the LightGBM
boosters, but trained with a different listwise loss (YetiRank vs LambdaMART) on a
different GBDT implementation (ordered/oblivious trees). Its ranking errors are only
weakly correlated with LightGBM's, so the equal-weight z-space blend reduces
score variance near the top and lifts nDCG@20 — a structural gain that transfers to
the blind set. The CatBoost model is trained on
feats_large (the 37-column struct_large variant); at serve time its scores reach this
module via the `ce_scores` argument of `score_route_top20`.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def load_rankers():
    """Load the two shipped LightGBM LambdaRank boosters from disk: (lgbm_base,
    lgbm_large). Same recipe; they differ only in their last-3 feature columns."""
    import lightgbm as lgb
    return (lgb.Booster(model_file=str(C.LGBM_BASE)),
            lgb.Booster(model_file=str(C.LGBM_LARGE)))


def _z(a):
    """Z-score standardization: (a - mean) / std -> mean 0, std 1. Puts scores on a
    common scale before blending. Returns all-zeros if std is ~0 (a constant array)."""
    a = np.asarray(a, dtype=np.float64)
    s = a.std()
    return (a - a.mean()) / s if s > 1e-9 else np.zeros_like(a)


def _znan(a):
    """zscore over non-NaN entries; NaN -> 0 (neutral).

    Like `_z` but NaN-robust: only finite entries are standardized (mean/std taken
    over them), and any NaN stays 0 so it neither pulls the blend up nor down.

    Why NaN-robust at all: this is the standardizer applied to the CatBoost scores
    (`ce_scores`). In the normal serving path those scores are dense — CatBoost's
    predict() scores the same full pool as LightGBM (recommender.py feeds it the whole
    feats_large matrix), so no candidate is "missing". The NaN guard is purely
    defensive: should any candidate's CatBoost score be NaN, mapping it to a neutral 0
    in z-space leaves that candidate's final score determined by the LightGBM half
    alone, instead of letting a NaN poison the argsort. `_z` (used for the routed
    LightGBM scores) omits this guard because those scores are always finite."""
    a = np.asarray(a, dtype=np.float64)
    out = np.zeros_like(a)
    m = ~np.isnan(a)
    if not m.any():
        return out
    v = a[m]
    s = v.std()
    out[m] = (v - v.mean()) / s if s > 1e-9 else 0.0
    return out


def score_route_full(lgbm_base, lgbm_large, feats_base, feats_large, pool, played_set,
                     route_low=C.ROUTE_LOW, route_high=C.ROUTE_HIGH,
                     ce_scores=None, ce_w=0.0):
    """Return (ranked_full, used_large, margin): the FULL ranked track_id list
    (played filtered, NOT cut to C.TOP_K).

    Performs Levels 4-6 exactly as score_route_top20 does, but collects ALL
    non-played candidates in descending score order instead of stopping at TOP_K.
    Used by the diagnostic rank_pool seam and any caller that needs the full ordering
    (e.g., to compute gold_rank beyond position 20).

    `used_large` / `margin` are the same routing diagnostics as score_route_top20.
    """
    # L4: score the pool with both boosters (one score per candidate).
    s_base = lgbm_base.predict(feats_base)
    s_large = lgbm_large.predict(feats_large)
    # Confidence of the base ranker's #1 pick = gap between its top-2 scores.
    s_sorted = np.sort(s_base)[::-1]
    margin = float(s_sorted[0] - s_sorted[1]) if len(s_sorted) >= 2 else 0.0
    # Selective routing: switch to lgbm_large when the base margin is UNINFORMATIVE —
    # either too small (no confident #1) or implausibly large (likely over-confident);
    # otherwise keep lgbm_base. Thresholds 0.25 / 1.5 from config.
    use_large = (margin < route_low) or (margin >= route_high)
    s = s_large if use_large else s_base
    # L5: equal-weight (in z-space) blend of the routed LightGBM scores with the
    # CatBoost ensemble. Standardizing both first makes ce_w=1.0 a true 50/50 blend
    # regardless of the two models' raw score scales.
    if ce_w and ce_scores is not None:
        s = _z(s) + ce_w * _znan(ce_scores)
    # L6: stable (mergesort -> deterministic) descending sort, then walk in order
    # skipping already-played tracks, collecting ALL remaining ids (no top_k cutoff).
    order = np.argsort(-s, kind="mergesort")
    ranked_full = []
    for idx in order:
        tid = pool[int(idx)]
        if tid in played_set:
            continue
        ranked_full.append(tid)
    return ranked_full, use_large, margin


def score_route_top20(lgbm_base, lgbm_large, feats_base, feats_large, pool, played_set,
                      route_low=C.ROUTE_LOW, route_high=C.ROUTE_HIGH, top_k=C.TOP_K,
                      ce_scores=None, ce_w=0.0):
    """Return (top20, used_large, margin). With ce_w>0, re-rank the routed pool by
    final = zscore(routed_LGB) + ce_w*zscore(CatBoost) (the CatBoost ensemble blend;
    ce_scores carries the CatBoost-YetiRank scores over the same pool).

    `used_large` is the routing decision (True if lgbm_large was selected) and
    `margin` is the base ranker's top1-top2 score gap — both returned as
    diagnostics. Routing thresholds default to config's ROUTE_LOW/ROUTE_HIGH.

    Delegates to score_route_full then slices to top_k — output is byte-identical
    to the previous implementation.
    """
    ranked_full, use_large, margin = score_route_full(
        lgbm_base, lgbm_large, feats_base, feats_large, pool, played_set,
        route_low=route_low, route_high=route_high, ce_scores=ce_scores, ce_w=ce_w)
    return ranked_full[:top_k], use_large, margin
