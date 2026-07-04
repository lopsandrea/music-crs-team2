"""Train the champion's two LightGBM LambdaRank rankers the recommender routes between.

The live ``recommender.ranker.score_route_top20`` scores every candidate pool with two
LambdaRank boosters and picks one per case by the base ranker's top1-top2 margin:

  * ``lgbm_base``  (``C.LGBM_BASE``) — the default ranker; trained on ``feats_r54`` (base +
    album + the **struct_base** rank/presence/cosine triple).
  * ``lgbm_large`` (``C.LGBM_LARGE``) — the routed-to ranker; trained on ``feats_r84_only`` (the
    same 34 base columns with the last 3 swapped struct_base → **struct_large**).

Both train on the SAME per-case candidate pools — the **nine-source** RRF pool that unions
struct_large into the recall pool (the live-recommender default), i.e.
``C.CASE_FEATURES_LARGE_POOL``. The two rankers differ only in which feature matrix they read
off each case (``feats_r54`` vs ``feats_r84_only``); the pool, the labels, and the groups are
identical. This exactly reproduces the shipped LightGBM boosters (verified: 1.0000 top-20
agreement on all 8000 dev cases).

  Pool note. The CatBoost ensemble (``training.catboost_ensemble``) trains on the EIGHT-source
  pool (``C.CASE_FEATURES``, struct_large a ranker feature only); these LightGBM rankers train
  on the NINE-source pool. That pool divergence (the "large_pool" variant) is the only thing
  that differs between the two ranker families' recall pools. ``case_features.build`` produces
  whichever via its ``large_in_pool`` switch.

Labels / groups (per case, == one ranking group): label is 1.0 for the candidate at the
case's ``gt_pos`` (the gold's pool index), 0.0 otherwise; the group size is the pool length.

Inputs:  the large-pool case-features artifact at ``C.CASE_FEATURES_LARGE_POOL`` (built on demand
         via ``case_features.build(large_in_pool=True)`` if absent).
Output:  ``C.LGBM_BASE`` (lgbm_base) and ``C.LGBM_LARGE`` (lgbm_large). Idempotent: skips when both
         models exist unless ``force=True``. Deterministic (fixed seed → predictions match the
         shipped boosters to fp round-off, max |Δ| ≲ 1e-13; the serialized text differs only in
         non-semantic header bytes, so re-trained models are not byte-identical but rank
         identically).
"""
from __future__ import annotations

import os
import pickle

# Cap OpenMP threads BEFORE importing lightgbm: LightGBM reads OMP_NUM_THREADS at import to
# size its thread pool. Pinning it (setdefault — only if the caller hasn't set it) bounds
# nondeterminism from thread-count-dependent floating-point reductions and avoids oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import lightgbm as lgb
import numpy as np

from recommender import config as C
from recommender.features import FEAT_R39_ALL

# Champion LambdaRank params (verbatim from the validated production LambdaRank trainers).
LGB_PARAMS = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
              "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
              "verbose": -1, "seed": 0}
NUM_BOOST_ROUND = 300

# The two rankers, each = (name, config output-path attribute, case-features key, last-3
# feature names). The case-features keys (``feats_r54`` / ``feats_r84_only``) and the last-3
# feature-name strings are LOCKED: both are baked into the shipped on-disk artifacts
# (``case_features.pkl`` dict schema; LightGBM ``feature_name`` in lgbm_base/large.txt), so
# renaming either needs the GPU retrain. The output path is resolved off ``C`` at build time
# (late-bound, so an override of ``C.LGBM_BASE`` / ``C.LGBM_LARGE`` — e.g. the verification
# harness's scratch paths — is honoured).
_RANKERS = (
    ("lgbm_base", "LGBM_BASE", "feats_r54", ["r54_rank_inv", "r54_presence", "r54_cosine"]),
    ("lgbm_large", "LGBM_LARGE", "feats_r84_only", ["r84_rank_inv", "r84_presence", "r84_cosine"]),
)


def _dataset(case_features: dict, feat_key: str, feature_name: list[str]) -> lgb.Dataset:
    """Pooled LambdaRank dataset: one ranking group per case, gold candidate labelled 1.

    For each case (in sorted index order, == one group) stack its ``feat_key`` rows; label is
    1.0 for the candidate at ``gt_pos`` else 0.0; the group size is the pool length — exactly
    as the validated production LambdaRank trainer assembles it.
    """
    X, y, group = [], [], []
    # sorted(case_features) fixes the case (group) order deterministically; LightGBM reads
    # `group` as consecutive block sizes over the row-concatenated X, so the rows of each case
    # must be appended contiguously (they are, one case per iteration).
    for idx in sorted(case_features):
        case = case_features[idx]
        n = len(case["pool"])
        X.append(np.asarray(case[feat_key], np.float64))
        # Binary relevance: exactly the gold candidate (at pool index gt_pos) is relevant.
        y.extend(1.0 if k == case["gt_pos"] else 0.0 for k in range(n))
        group.append(n)
    # feature_name = the 34 shared columns (FEAT_R39_ALL) + the 3 struct triple names for this
    # ranker, giving the 37-name vector baked into the saved model (the r54_*/r84_* last-3
    # names are locked — see module docstring).
    return lgb.Dataset(np.concatenate(X), label=np.asarray(y, np.float64),
                       group=group,
                       feature_name=list(FEAT_R39_ALL) + feature_name)


def build(force: bool = False, smoke: bool = False) -> None:
    """Train and save both LambdaRank rankers (``C.LGBM_BASE`` and ``C.LGBM_LARGE``).

    No-op when both models already exist unless ``force=True``. With ``smoke=True`` the boost
    rounds are reduced for a fast end-to-end check; all other params are identical.
    """
    if C.LGBM_BASE.exists() and C.LGBM_LARGE.exists() and not force:
        print("  [skip] LightGBM rankers present"); return

    # Ensure the large-in-pool training matrix exists (build it if a prior stage hasn't);
    # no-op when C.CASE_FEATURES_LARGE_POOL is already present.
    from training import case_features
    case_features.build(force=False, smoke=smoke, large_in_pool=True)

    with open(C.CASE_FEATURES_LARGE_POOL, "rb") as f:
        case_features_dev = pickle.load(f)
    print(f"[lgbm_rankers] training 2 LambdaRank rankers on {len(case_features_dev)} dev "
          f"cases (large-in-pool, smoke={smoke}) …", flush=True)

    params = dict(LGB_PARAMS)
    num_boost_round = 50 if smoke else NUM_BOOST_ROUND

    # Train both rankers from the SAME pools/labels/groups, differing only in which feature
    # matrix (feat_key: struct_base vs struct_large triple) each reads off every case.
    for name, attr, feat_key, feat_names in _RANKERS:
        # Resolve the output path off C at build time (late-bound) so a test harness can
        # redirect C.LGBM_BASE / C.LGBM_LARGE to scratch paths.
        out = getattr(C, attr)
        ds = _dataset(case_features_dev, feat_key, feat_names)
        booster = lgb.train(params, ds, num_boost_round=num_boost_round)
        out.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(out))
        print(f"  {name} ({num_boost_round} rounds, {feat_key}) -> {out}", flush=True)
