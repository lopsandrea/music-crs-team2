"""Final ranker training: importance-weighted retrain for the dev→blind covariate shift.

The dev split the rankers train on and the blind evaluation sessions are not identically
distributed (session depth, query shape, coverage of the played tracks by the embedding
indices, played-track popularity all shift). The shipped rankers therefore use a standard
two-step domain-adaptation recipe on top of the plain training protocol:

Step 1 — ``build_weights`` (adversarial validation): a classifier separates dev cases
(label 0) from blind sessions (label 1) on GOLD-FREE case descriptors only (history
depth, query shape, index coverage of the played tracks, played popularity — never the
target track). Every dev case then gets an importance weight ``w = p(blind|x) / mean(p)``,
so dev cases that look like blind sessions dominate the training loss. 5-fold CV keeps
the reported AUC and the dev-side probabilities out-of-fold.

Step 2 — ``train`` : the EXACT plain training protocol (same matrices, parameters and
pools as ``training.lgbm_rankers`` / ``training.catboost_ensemble``) re-run with
per-group weights ``w ** TRANSFER_ALPHA`` (alpha tempers the reweighting; the shipped
rankers use ``C.TRANSFER_ALPHA = 0.25``). Outputs OVERWRITE ``cache/rankers/`` — these
weighted models are the shipped serving rankers.

All inputs are official challenge data (the dev cases and the public blind sessions);
no labels of any evaluation set are used anywhere — only input-side session descriptors.
"""
from __future__ import annotations

import json
import pickle

import numpy as np

from recommender import config as C

# Gold-free case descriptors for the adversarial classifier (order is the feature order).
FEATS = ["n_played", "n_user_turns", "query_toks", "turn_number",
         "cov_qwen", "cov_cfbpr", "cov_als", "played_pop_mean",
         "played_pop_max", "n_artists"]

# The adversarial-classifier reference blind split. The shipped weights were fit against
# the Blind-A sessions (the public blind split available at training time).
_ADV_BLIND_SPLIT = "blind_a"


def _case_feats(uq, history, played, idx_sets, track_pop, meta):
    """One gold-free descriptor row for a session/case (turn_number filled by caller)."""
    users = [h for h in history if h.get("role") == "user"]
    pops = [float(track_pop.get(t, 0)) for t in played]
    arts = {str((meta.get(t, {}).get("artist_name") or [""])[0]) for t in played if t in meta}
    cov = [(sum(1 for t in played if t in s) / len(played)) if played else 0.0
           for s in idx_sets]
    return [len(played), len(users), len(str(uq).split()),
            0,  # turn_number filled by caller
            *cov, float(np.mean(pops)) if pops else 0.0,
            float(np.max(pops)) if pops else 0.0, len(arts)]


def build_weights(force: bool = False):
    """Adversarial validation → per-dev-case importance weights (C.TRANSFER_WEIGHTS)."""
    if C.TRANSFER_WEIGHTS.exists() and not force:
        print("  [skip] transfer case weights present")
        return
    import lightgbm as lgb
    import training.case_features as cf
    from recommender.data import (load_blind_sessions, load_supporting_maps,
                                  load_track_metadata)

    meta = load_track_metadata()
    _maps, track_pop, _alb = load_supporting_maps()
    # Coverage sets: which played tracks each embedding/collaborative index knows about.
    idx_sets = []
    for d in (C.QWEN_DIR, C.CFBPR_DIR):
        idx_sets.append(set(json.loads((d / "track_ids.json").read_text())))
    als = np.load(C.ALS_NPZ, allow_pickle=True)
    idx_sets.append(set(als["track_ids"].tolist()))

    dev_cases = cf._load_dev_payload()["cases"]
    blind = load_blind_sessions(_ADV_BLIND_SPLIT)

    def row(uq, hist, played, turn):
        r = _case_feats(uq, hist, played, idx_sets, track_pop, meta)
        r[3] = turn
        return r

    Xd = np.array([row(c["user_query"], c["history"], c["music_turns"],
                       c.get("turn_number", 0)) for c in dev_cases])
    Xb = np.array([row(s["user_query"], s["history"], s["music_turns"],
                       s.get("turn_number", 0)) for s in blind])
    X = np.vstack([Xd, Xb])
    y = np.array([0.0] * len(Xd) + [1.0] * len(Xb))

    # 5-fold CV for an honest AUC + out-of-fold p(blind|x) on the dev rows.
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    p = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        m = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                               is_unbalance=True, random_state=0, verbose=-1)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, p)
    full = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                              is_unbalance=True, random_state=0, verbose=-1).fit(X, y)
    imp = sorted(zip(FEATS, full.feature_importances_), key=lambda kv: -kv[1])
    print(f"adversarial AUC (dev vs {_ADV_BLIND_SPLIT}) = {auc:.4f}")
    print("top shift drivers:", [(f, int(v)) for f, v in imp[:6]])

    pd_dev = p[:len(Xd)]
    w = pd_dev / pd_dev.mean()                    # mean 1.0 over dev cases
    C.TRANSFER_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    C.TRANSFER_WEIGHTS.write_bytes(pickle.dumps({"w": w, "auc": auc,
                                                 "p_blind_dev": pd_dev,
                                                 "importances": imp}))
    q = np.quantile(w, [0.1, 0.5, 0.9, 0.99])
    print(f"dev case weights: q10 {q[0]:.3f} med {q[1]:.3f} q90 {q[2]:.3f} q99 {q[3]:.3f}"
          f" -> {C.TRANSFER_WEIGHTS.name}")


def _weighted_arm(cf_dict, train_idx, w_case, smoke=False):
    """The plain trainers with per-group weights ``w_case[i]`` (row-replicated for
    LightGBM, group weights for CatBoost) over the 37-column matrices."""
    import lightgbm as lgb
    from catboost import CatBoostRanker, Pool
    from recommender.features import FEAT_R39_ALL
    from training.catboost_ensemble import CB_PARAMS
    from training.lgbm_rankers import LGB_PARAMS, NUM_BOOST_ROUND, _RANKERS

    sub = [i for i in sorted(cf_dict) if i in train_idx]
    boosters = {}
    for name, _a, feat_key, last3 in _RANKERS:
        X, y, group, wrow = [], [], [], []
        for i in sub:
            case = cf_dict[i]
            n = len(case["pool"])
            X.append(np.asarray(case[feat_key], np.float64))
            y.extend(1.0 if k == case["gt_pos"] else 0.0 for k in range(n))
            group.append(n)
            wrow.extend([w_case[i]] * n)
        ds = lgb.Dataset(np.concatenate(X), label=np.asarray(y, np.float64),
                         group=group, weight=np.asarray(wrow, np.float64),
                         feature_name=list(FEAT_R39_ALL) + last3)
        boosters[name] = lgb.train(dict(LGB_PARAMS), ds,
                                   num_boost_round=50 if smoke else NUM_BOOST_ROUND)
        print(f"  trained {name}", flush=True)
    X, y, gid, gw = [], [], [], []
    for g, i in enumerate(sub):
        case = cf_dict[i]
        n = len(case["pool"])
        X.append(np.asarray(case["feats_r84_only"], np.float64))
        y.extend(1.0 if k == case["gt_pos"] else 0.0 for k in range(n))
        gid.extend([g] * n)
        gw.extend([w_case[i]] * n)
    params = dict(CB_PARAMS)
    if smoke:
        params["iterations"] = 50
    cb = CatBoostRanker(**params)
    cb.fit(Pool(np.concatenate(X), label=y, group_id=gid, weight=gw))
    print("  trained catboost", flush=True)
    return boosters["lgbm_base"], boosters["lgbm_large"], cb


def _alpha_weights(alpha: float) -> np.ndarray:
    w = pickle.loads(C.TRANSFER_WEIGHTS.read_bytes())["w"] ** alpha
    return w / w.mean()


def train(alpha: float, smoke: bool = False):
    """Weighted full-train on ALL dev cases → the final serving rankers (cache/rankers/).

    Exactly the plain protocol, only the per-group weights differ: the LightGBM pair
    trains on the 9-source matrix (C.CASE_FEATURES_LARGE_POOL), the CatBoost ensemble
    on the 8-source one (C.CASE_FEATURES).
    """
    lp = pickle.loads(C.CASE_FEATURES_LARGE_POOL.read_bytes())   # LGBM matrices
    cbp = pickle.loads(C.CASE_FEATURES.read_bytes())             # CatBoost (8-src pool)
    w = _alpha_weights(alpha)
    out = C.LGBM_BASE.parent
    out.mkdir(parents=True, exist_ok=True)
    lb, ll, _cb_unused = _weighted_arm(lp, set(lp), w, smoke=smoke)
    lb.save_model(str(C.LGBM_BASE))
    ll.save_model(str(C.LGBM_LARGE))
    _lb2, _ll2, cb = _weighted_arm(cbp, set(cbp), w, smoke=smoke)
    cb.save_model(str(C.CB_MODEL))
    stamp = {"alpha": alpha, "weights": C.TRANSFER_WEIGHTS.name, "smoke": bool(smoke)}
    (out / "transfer_stamp.json").write_text(json.dumps(stamp, indent=1))
    print(f"saved weighted rankers (alpha={alpha:g}) -> {out}")


def build(force: bool = False, smoke: bool = False) -> None:
    """train_all stage: adversarial weights + weighted retrain → cache/rankers/.

    The sentinel is ``cache/rankers/transfer_stamp.json`` (NOT the model files: the
    plain stages 9-10 write the same paths first; this stage must still run after them
    to produce the final weighted models).
    """
    stamp = C.LGBM_BASE.parent / "transfer_stamp.json"
    if stamp.exists() and not force:
        print("  [skip] transfer-weighted rankers present (transfer_stamp.json)")
        return
    build_weights(force=force)
    train(C.TRANSFER_ALPHA, smoke=smoke)
