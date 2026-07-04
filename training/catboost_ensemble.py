"""Train the champion CatBoost-YetiRank ensemble ranker → cache/rankers/catboost_ensemble.cbm.

This is the champion's winning lever. A `CatBoostRanker` (YetiRank, 600 trees) is
fit on the per-case candidate pools using the struct_large feature variant (the 37-column
`feats_r84_only` matrix), with one ranking group per dev case. At inference it is
ensembled with the routed LightGBM score by z-scoring both and adding them:

    final = z(routed_LGB) + C.CB_W * z(catboost)          # C.CB_W = 1.0

so the CatBoost model only needs to be a *decorrelated* re-ranker over the same
pool — the standardisation makes the blend weight scale-free.

Inputs:  the case-features artifact at `C.CASE_FEATURES` (8000 dev cases; each a
dict with `pool`, `gt_pos`, and the precomputed `feats_r84_only` (n_cand, 37) matrix).
The `feats_r84_only` dict key is the locked on-disk schema of `case_features.pkl`.
Output:  `C.CB_MODEL` (CatBoost `.cbm`). Idempotent: skips when the model exists
unless `force=True`. GPU (device 0); deterministic seed.
"""
from __future__ import annotations

import pickle

import numpy as np
from catboost import CatBoostRanker, Pool

from recommender import config as C

# Champion params (verbatim from the validated train_prod_catboost.cb_train).
# YetiRank is a listwise ranking loss (optimises the order within each group); 600 trees of
# depth 8 at lr 0.05. task_type="GPU"/devices="0" pin the fit to CUDA device 0 (this stage is
# GPU-only); random_seed=0 makes the fit reproducible.
CB_PARAMS = dict(loss_function="YetiRank", iterations=600, depth=8, learning_rate=0.05,
                 random_seed=0, task_type="GPU", devices="0", verbose=False)

# Feature variant the champion ranks on: the 37-column struct_large matrix.
# r84 == struct_large; name locked: "feats_r84_only" is the on-disk dict key baked into the
# shipped case_features.pkl schema, so it cannot be renamed without the deferred 14-17h retrain.
# It selects the matrix whose last 3 columns are the struct_large triple (r84_rank_inv /
# r84_presence / r84_cosine), as opposed to the feats_r54 (struct_base) variant the LightGBM
# lgbm_base ranks on. CatBoost trains only on this struct_large variant.
FEAT_KEY = "feats_r84_only"


def _assemble(case_features: dict) -> Pool:
    """Pooled training matrix: one ranking group per case, the gold candidate labelled 1.

    For each case (in sorted index order, == one group) stack its `feats_r84_only`
    rows; label is 1.0 for the candidate at `gt_pos`, else 0.0; group_id is the
    case's running index — exactly as the validated `cb_train` builds it.
    """
    X, y, group_id = [], [], []
    # `group` is a dense 0..n_cases-1 index; unlike LightGBM (which takes per-group block sizes),
    # CatBoost groups rows by EQUAL group_id, so every row of a case is tagged with that case's
    # running index. sorted(case_features) fixes the order deterministically.
    for group, idx in enumerate(sorted(case_features)):
        case = case_features[idx]
        n = len(case["pool"])
        X.append(np.asarray(case[FEAT_KEY], np.float64))
        # Binary relevance: only the gold candidate (pool index gt_pos) is labelled 1.
        # YetiRank only needs the in-group order, so a single positive per group is enough.
        y.extend(1.0 if k == case["gt_pos"] else 0.0 for k in range(n))
        # One repeated id per pool row tags the whole case as one ranking group.
        group_id.extend([group] * n)
    # `X` is a list of (n_cand_i, 37) blocks; concatenate -> (sum n_cand, 37) single matrix.
    # CatBoost requires rows of the same group_id to be contiguous, which they are here because
    # each case's rows are appended in one go in group order. Returns ONE Pool covering all cases
    # (contrast LightGBM, which takes an explicit per-group block-size list instead of group ids).
    return Pool(np.concatenate(X), label=y, group_id=group_id)


def build(force: bool = False, smoke: bool = False,
          feat_cache=None, out=None) -> None:
    """Train and save the champion CatBoost-YetiRank ensemble model.

    No-op when `C.CB_MODEL` already exists unless `force=True`. With `smoke=True`
    the tree count is reduced for a fast end-to-end check; all other params are
    identical. `feat_cache`/`out` override the config paths (for testability);
    the public contract is `build(force, smoke)`.
    """
    out = C.CB_MODEL if out is None else out
    if out.exists() and not force:
        print("  [skip] CatBoost ensemble model present"); return

    feat_cache = C.CASE_FEATURES if feat_cache is None else feat_cache
    print(f"[catboost_ensemble] fitting YetiRank ranker on {FEAT_KEY} "
          f"(smoke={smoke}) …", flush=True)
    with open(feat_cache, "rb") as f:
        case_features = pickle.load(f)

    # Copy so the smoke override never mutates the module-level CB_PARAMS template.
    params = dict(CB_PARAMS)
    if smoke:
        # Fast end-to-end wiring check only: 50 trees instead of 600 (NOT the champion model).
        params["iterations"] = 50

    model = CatBoostRanker(**params)
    model.fit(_assemble(case_features))

    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))
    print(f"  CatBoost ensemble ({params['iterations']} trees, "
          f"{len(case_features)} groups) -> {out}", flush=True)
