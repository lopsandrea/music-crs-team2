"""Build the per-dev-case feature matrices the LGBM rankers and CatBoost ensemble train on.

The dict KEYS emitted per case (``feats_r54`` / ``feats_r84_only`` / ``feats_combined``) are
the on-disk schema of the shipped ``case_features.pkl`` artifact — kept verbatim so the
shipped rankers read it unchanged (they are NOT renamed; renaming needs the GPU retrain).

For each of the 8000 dev cases this fuses the eight pool sources (qwen/bm25/cf-bpr/ALS +
text_retriever + struct_base; struct_large is a ranker feature, not a pool source — see below)
into a 300-track weighted-RRF pool and emits, for that pool:

  * ``feats_r54``       — the 37-column matrix from ``recommender.features.featurize``
                          (base + album + the struct_base rank/presence/cosine triple).
  * ``feats_r84_only``  — a copy of ``feats_r54`` whose last three columns are replaced by
                          the struct_large rank/presence/score triple (the substitution
                          inference's ``recommend_session`` performs to build ``feats_large``).
                          struct_large is a *ranker feature only*; it never enters the RRF pool.
  * ``feats_combined``  — the 40-column ``[feats_r54 | struct_large-triple]`` stack, kept for
                          parity with the shipped artifact; the LGBM/CatBoost rankers only
                          consume ``feats_r54`` + ``feats_r84_only``.

plus ``pool`` (300 ids), ``gt``, and ``gt_pos`` (pool index of the gold, or -1).

The per-track featurisation reuses the SAME ``recommender.features.featurize`` (and
``recommender.fusion.weighted_rrf``) the live recommender uses, so columns are computed
identically to serving. The offline-specific wiring is the candidate sourcing: rather than
the blind/live retrievers, each case is fed its *out-of-fold* text_retriever/struct_base/
struct_large lists (so the rankers never train on cases a retriever already saw), struct_large
is selected per case from the fold-model that held it out, and the RRF pool is fused over eight
sources (see below).

OOF source provenance (all consumed from clean ``recommender.config`` paths):

  ===================  =====================  ==================================================
  source (RRF key)     config path            format
  ===================  =====================  ==================================================
  qwen_recent          DEV_PAYLOAD["src_a"]   [n_cases][track_id]
  bm25_lastmusic       DEV_PAYLOAD["src_b"]   [n_cases][track_id]
  bm25_convo           DEV_PAYLOAD["src_c"]   [n_cases][track_id]
  qwen_neighbors       DEV_PAYLOAD["src_d"]   [n_cases][track_id]
  cfbpr_recent         DEV_PAYLOAD["src_f"]   [n_cases][track_id]
  als_session          derived from ALS_NPZ   per-case top-200 list + session vector (below)
  text_retriever       TEXT_RETRIEVER_OOF     [n_cases][track_id]
  struct_base          STRUCT_BASE_OOF        {"lists": [n_cases][[track_id, score]]}
  struct_large         STRUCT_LARGE_OOF_DIRS[fold]  {case_idx_str: [[track_id, score]]} per fold,
                                              fold chosen via the computed fold map
                                              (folds.grouped_session_folds; case_idx -> fold_idx)
  ===================  =====================  ==================================================

Pool composition — two shipped matrices (8-source for CatBoost, 9-source for the LGBM rankers)
---------------------------------------------------------------------------------------------
The two shipped ranker families were fit on RRF pools that differ ONLY in whether struct_large
is unioned into the recall pool. Selected via ``build(large_in_pool=...)``:

* ``large_in_pool=False`` → EIGHT-source pool (qwen/bm25/cf-bpr + ALS + text_retriever +
  struct_base); struct_large is held out of the fusion and contributes only its three
  substituted ranker columns. This is the matrix the shipped **CatBoost** ensemble was fit on
  (``training.catboost_ensemble``).
* ``large_in_pool=True`` → NINE-source pool (the above + struct_large), matching the live
  ``recommender.recommend_session`` (whose ``SW_BASELINE`` weights ``struct_large=1.0``). This
  is the matrix the shipped **LightGBM** LambdaRank rankers (``lgbm_base``/``lgbm_large``) were
  fit on (``training.lgbm_rankers``).

In both pools struct_large still contributes the same ranker-feature triple (last 3 cols of
``feats_r84_only``); only pool membership — and hence which 300 candidates each case has —
changes. ``_POOL_SOURCES`` / ``_POOL_SOURCES_LARGE`` below are the two offline source sets.

ALS session vector — note on the shipped weighting
---------------------------------------------------
The shipped artifact was built with a *recency-decayed, newest-weighted* session vector
(``w[j] = 0.8 ** (n-1-j)`` over the played tracks in order, normalised), and that same
vector feeds the ``als_dot`` column. This module reproduces that exact weighting so the
matrices match the artifact byte-for-byte. It deliberately does NOT reuse
``recommender.sources.Sources.src_als``: that clean source uses the inverse
(oldest-weighted ``w[i] = 0.8 ** i``) decay — a known divergence — which would shift the
ALS source list and the ``als_dot`` feature away from the artifact.

Inputs are consumed from the clean ``recommender.config`` paths; wiring the earlier
training stages to *produce* those clean-named files is a separate task.
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np

from recommender import config as C
from recommender.data import load_als, load_supporting_maps
from recommender.features import N_R39, featurize
from recommender.fusion import weighted_rrf
from training.folds import grouped_session_folds

# Offline RRF source order maps the candidate retrievers onto the inference-side source keys
# `featurize`/`weighted_rrf` expect (verbatim from the live recommender's `src_lists`).
# struct_large is appended below; the other eight are dict-keyed here.
_PAYLOAD_KEYS = {"qwen_recent": "src_a", "bm25_lastmusic": "src_b", "bm25_convo": "src_c",
                 "qwen_neighbors": "src_d", "cfbpr_recent": "src_f"}
# The eight RRF pool sources for the CatBoost training matrix (struct_large is a ranker
# feature, not a pool source — this 8-source pool reproduces the shipped CatBoost).
_POOL_SOURCES = ["qwen_recent", "bm25_lastmusic", "bm25_convo", "qwen_neighbors",
                 "cfbpr_recent", "als_session", "text_retriever", "struct_base"]
# The LightGBM LambdaRank rankers (`lgbm_base`/`lgbm_large`) shipped trained on a NINE-source
# pool that ALSO unions struct_large into the RRF recall pool — the live recommender's default
# (see config.SW_BASELINE weights struct_large=1.0). Selected via `build(large_in_pool=True)`;
# reproduces the shipped `lgbm_base`/`lgbm_large` rankers 1.0000. struct_large still also
# contributes its ranker-feature triple (feats_r84_only) exactly as in the 8-source pool —
# only pool membership changes.
_POOL_SOURCES_LARGE = _POOL_SOURCES + ["struct_large"]
_N_FOLDS = 5
_ALS_TOPN = 200
_ALS_DECAY = 0.8


# ---------------- input loaders (clean config paths) ----------------

def _load_dev_payload() -> dict:
    """The pickled dev payload: ``cases`` (per-case session/query/gt) + the precomputed
    source lists ``src_a``..``src_f`` (qwen/bm25/cf-bpr). Built by ``training.dev_payload``."""
    with open(C.DEV_PAYLOAD, "rb") as f:
        return pickle.load(f)


def _load_text_oof() -> list[list[str]]:
    """Per-case OOF text-retriever lists: [n_cases][track_id]."""
    with open(C.TEXT_RETRIEVER_OOF) as f:
        return json.load(f)


def _load_struct_base_oof() -> tuple[list[list[str]], list[dict[str, float]]]:
    """Per-case OOF struct_base lists + cosine-score maps.

    On disk each case is ``[[track_id, score], ...]``; this splits that into the ordered
    id list (for RRF / rank features) and a ``{track_id: cosine}`` map (the struct_base
    cosine feature column). r54 == struct_base; name locked into the shipped artifact.
    """
    with open(C.STRUCT_BASE_OOF) as f:
        data = json.load(f)
    lists, scores = [], []
    for case_lists in data["lists"]:
        lists.append([tid for tid, _ in case_lists])
        scores.append({tid: float(s) for tid, s in case_lists})
    return lists, scores


def _load_struct_large_oof_per_fold(missing_ok: bool = False) -> dict[int, dict[int, dict]]:
    """Per-fold OOF struct_large lists, keyed fold -> case_idx -> {ranks, scores, tids}.

    r84 == struct_large; name locked (baked into the shipped artifact). Unlike struct_base,
    struct_large is stored as one JSON file PER FOLD (``C.STRUCT_LARGE_OOF_DIRS[fold]``),
    each a ``{case_idx_str: [[track_id, score], ...]}`` map of only that fold's held-out cases.
    For each case the pairs are pre-digested into:
      * ``ranks``  — {track_id: 1-based rank} for the rank-inverse feature,
      * ``scores`` — {track_id: score} for the score feature column,
      * ``tids``   — the ordered id list (for RRF when struct_large is in the pool).
    The caller selects the right fold per case via the computed fold map (``_compute_fold_map``).
    """
    per_fold: dict[int, dict[int, dict]] = {}
    for fold in range(_N_FOLDS):
        if not C.STRUCT_LARGE_OOF_DIRS[fold].exists():
            # Under smoke the struct_large stage trains a single tiny fold, so the other
            # OOF files legitimately do not exist yet; a full build must fail fast instead.
            if missing_ok:
                print(f"  [smoke] struct_large OOF fold {fold} missing — its cases are skipped")
                continue
            raise FileNotFoundError(f"missing struct_large OOF file: {C.STRUCT_LARGE_OOF_DIRS[fold]}")
        with open(C.STRUCT_LARGE_OOF_DIRS[fold]) as f:
            raw = json.load(f)
        out: dict[int, dict] = {}
        for case_idx_str, pairs in raw.items():
            # JSON object keys are strings; convert back to the integer case index.
            tids = [tid for tid, _ in pairs]
            out[int(case_idx_str)] = {
                "ranks": {tid: r + 1 for r, tid in enumerate(tids)},
                "scores": {tid: float(s) for tid, s in pairs},
                "tids": tids,
            }
        per_fold[fold] = out
    return per_fold


def _compute_fold_map(cases: list[dict]) -> list[int]:
    """case_idx -> fold_idx that held the case out (so struct_large is scored OOF per case).

    Computed from the pipeline-wide shared split (``training.folds.grouped_session_folds``
    with ``seed=0, k=5``) over the dev-payload case order, then inverted to a per-case map.
    This replaces a read of a precomputed fold artifact so a from-scratch rebuild needs no
    external fold file; the shared split is the same one every retriever (text_retriever/
    struct_base/struct_large) holds cases out by, and it is byte-identical to the old artifact
    (8000/8000 ``fold_idx`` match).
    """
    sessions = [c["session_id"] for c in cases]
    folds = grouped_session_folds(sessions, seed=0, k=5)
    case_fold = [-1] * len(cases)
    for fold_idx, held in enumerate(folds):
        for case_idx in held.tolist():
            case_fold[case_idx] = fold_idx
    return case_fold


# ---------------- ALS session source (artifact-matching weighting) ----------------

def _als_session(played: list[str], als_factors: np.ndarray, als_to_idx: dict[str, int]):
    """Per-case ALS source list (top-200, played tracks excluded) + session vector.

    Newest-weighted recency decay, matching the shipped artifact (see module docstring).
    """
    # Map played tracks to ALS item rows, preserving play order; skip any not in the ALS vocab.
    anchors = [als_to_idx[t] for t in played if t in als_to_idx]
    if not anchors:
        # No ALS-known history => no session vector (this case gets an empty ALS source).
        return [], None
    n = len(anchors)
    # Newest-weighted recency decay: the LAST played track gets weight 0.8**0 = 1, the
    # one before 0.8**1, etc. (j runs oldest->newest, exponent n-1-j). Normalised to sum 1.
    # This deliberately mirrors the SHIPPED artifact weighting, NOT recommender.sources
    # (which uses the inverse oldest-weighted decay) — see module docstring.
    w = np.array([_ALS_DECAY ** (n - 1 - j) for j in range(n)], dtype=np.float32)
    w /= w.sum()
    # Weighted average of the anchor item factors => one (dim,) session embedding.
    session_vec = np.zeros(als_factors.shape[1], dtype=np.float32)
    for j, idx in enumerate(anchors):
        session_vec += w[j] * als_factors[idx]
    # Score every catalog item by dot with the session vector; shape (n_items,).
    scores = als_factors @ session_vec
    # Exclude already-played tracks from the recommendation list.
    for t in played:
        idx = als_to_idx.get(t)
        if idx is not None:
            scores[idx] = -np.inf
    # Top-200 via argpartition (unordered top-N, O(n_items)) then argsort just those N.
    topn = min(_ALS_TOPN, len(scores))
    top = np.argpartition(-scores, topn - 1)[:topn]
    top = top[np.argsort(-scores[top])]
    # Return BOTH the id-index list (the ALS source) and the session vector (feeds als_dot feature).
    return top, session_vec


# ---------------- build ----------------

def build(force: bool = False, smoke: bool = False,
          large_in_pool: bool = False, out=None) -> None:
    """Build the case-features artifact.

    No-op when the target artifact already exists unless ``force=True``. With ``smoke=True``
    only the first ~200 cases are built (fast end-to-end check). Writes a dict
    ``{case_idx: {pool, gt, gt_pos, feats_r54, feats_r84_only, feats_combined}}`` (the
    ``feats_*`` keys are the locked shipped-artifact schema — see module docstring).

    Pool composition is selected by ``large_in_pool`` (the only thing that differs between the
    two shipped training matrices):

    * ``large_in_pool=False`` (default) → 8-source RRF pool (struct_large excluded from the
      pool); written to ``C.CASE_FEATURES``. This is the matrix the shipped CatBoost trained on.
    * ``large_in_pool=True`` → 9-source RRF pool (struct_large unioned into the pool, the
      live-recommender default); written to ``C.CASE_FEATURES_LARGE_POOL``. This is the matrix
      the shipped LightGBM rankers (``lgbm_base``/``lgbm_large``) trained on;
      ``training.lgbm_rankers`` consumes it.

    ``out`` overrides the output path (and the skip check); otherwise it defaults to
    ``C.CASE_FEATURES_LARGE_POOL`` when ``large_in_pool`` else ``C.CASE_FEATURES``.
    """
    pool_sources = _POOL_SOURCES_LARGE if large_in_pool else _POOL_SOURCES
    if out is None:
        out = C.CASE_FEATURES_LARGE_POOL if large_in_pool else C.CASE_FEATURES
    if out.exists() and not force:
        print("  [skip] case-features artifact present"); return

    print("[case_features] loading dev payload + OOF lists + ALS/maps …", flush=True)
    payload = _load_dev_payload()
    cases = payload["cases"]
    n = len(cases)

    text_oof = _load_text_oof()
    struct_base_oof, struct_base_scores = _load_struct_base_oof()
    struct_large_per_fold = _load_struct_large_oof_per_fold(missing_ok=smoke)
    case_fold = _compute_fold_map(cases)

    als_factors, als_ids, als_to_idx = load_als()
    # Supporting per-track maps shared with the live featuriser. max_pop normalises the
    # popularity feature to [0,1]; the five unpacked maps are: track->artist, track->tag set,
    # track->title tokens, track->artist tokens, track->all-metadata tokens.
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1
    ta, tt, ttl, tat, tmt = (maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
                             maps["track_artist_toks"], maps["track_meta_toks"])

    # smoke mode caps the build at the first 200 cases for a fast end-to-end wiring check;
    # the resulting artifact is partial and must NOT be shipped (recall stats below are noisy).
    n_build = min(200, n) if smoke else n
    print(f"[case_features] building {n_build}/{n} cases (smoke={smoke}) …", flush=True)

    case_features: dict[int, dict] = {}
    t0 = time.time()
    for i in range(n_build):
        # Smoke: skip cases held out by a fold whose (tiny) struct_large OOF wasn't built.
        if case_fold[i] not in struct_large_per_fold or i not in struct_large_per_fold[case_fold[i]]:
            continue
        case = cases[i]
        played = case["music_turns"]

        # --- ALS source list (top-200) + session vector, artifact-matching weighting ---
        # `_als_session` returns item-ROW indices into als_factors; map them back to track_ids
        # via als_ids so the ALS source speaks the same id space as the other eight sources.
        # An empty history (als_vec is None) yields an empty ALS contribution for this case.
        als_top_idx, als_vec = _als_session(played, als_factors, als_to_idx)
        als_list = [als_ids[int(j)] for j in als_top_idx] if als_vec is not None else []

        # --- nine candidate sources, keyed as the live recommender's `src_lists` ---
        src_lists = {key: payload[pk][i] for key, pk in _PAYLOAD_KEYS.items()}
        src_lists["als_session"] = als_list
        src_lists["text_retriever"] = text_oof[i]
        src_lists["struct_base"] = struct_base_oof[i]
        src_lists["struct_large"] = struct_large_per_fold[case_fold[i]][i]["tids"]

        # --- weighted-RRF pool over the selected pool sources (8-source, or 9 with struct_large) ---
        pool_lists = {k: src_lists[k] for k in pool_sources}
        pool = weighted_rrf(pool_lists, C.SW_BASELINE, topk=C.POOL_K, k=C.RRF_K)
        gt = case["gt"]
        # Pool index of the gold next track, or -1 if it fell outside the 300-candidate pool.
        # The ranker training label is derived from this position downstream; -1 == not recoverable.
        gt_pos = pool.index(gt) if gt in pool else -1

        # --- feats_base: the 37-column inference featurisation (struct_base triple) ---
        # 1-based rank maps for the two retriever rank-inverse features (r21==text_retriever,
        # r54==struct_base; both names locked into the shipped LightGBM feature vector). Truncated
        # to POOL_K so ranks match what the live featuriser sees.
        text_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["text_retriever"][:C.POOL_K])}
        base_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["struct_base"][:C.POOL_K])}
        # Same positional contract as the live recommender's featurize call: pool + all nine
        # source lists, the two retriever rank maps (text_retriever / struct_base) and the
        # struct_base cosine map, the textual context (query/history/played + played as a set
        # for O(1) "already-played" lookups), the five per-track token/metadata maps, and the
        # ALS factors/index/session vector + popularity/album maps. Returns the (n_pool, 37)
        # base matrix whose last 3 cols are the struct_base triple.
        feats_base = featurize(
            pool, src_lists, text_rank_map, base_rank_map, struct_base_scores[i],
            case["user_query"], case["history"], played, set(played),
            ta, tt, ttl, tat, tmt,
            als_factors, als_to_idx, als_vec, track_pop, max_pop, track_album)

        # --- struct_large rank/presence/score triple over the same pool ---
        # Pick the struct_large lists from the fold that held THIS case out (OOF), so the
        # ranker never sees a struct_large score from a model that trained on this case.
        large = struct_large_per_fold[case_fold[i]][i]
        large_ranks = large["ranks"]
        # One (n_pool, 3) block aligned to `pool`: [rank_inv, presence, score]. Tracks absent
        # from struct_large's list get rank_inv=0, presence=0, score=0 (the missing-source default).
        large_cols = np.zeros((len(pool), 3), dtype=np.float64)
        for k, tid in enumerate(pool):
            large_cols[k, 0] = (1.0 / large_ranks[tid]) if tid in large_ranks else 0.0
            large_cols[k, 1] = 1.0 if tid in large_ranks else 0.0
            large_cols[k, 2] = large["scores"].get(tid, 0.0)

        # feats_large: copy feats_base, substitute the last-3 (struct_base) cols with struct_large —
        # exactly the feats_base -> feats_large step in recommender.recommend_session.
        feats_large = feats_base.copy()
        feats_large[:, N_R39:N_R39 + 3] = large_cols
        feats_combined = np.concatenate([feats_base, large_cols], axis=1)  # 40 cols (37 + 3 struct_large)

        # dict KEYS are the locked on-disk artifact schema (case_features.pkl) — do NOT rename.
        case_features[i] = {"pool": pool, "gt": gt, "gt_pos": gt_pos,
                            "feats_r54": feats_base, "feats_r84_only": feats_large,
                            "feats_combined": feats_combined}
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_build} ({time.time() - t0:.0f}s)", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    # HIGHEST_PROTOCOL keeps the (potentially large) numpy feature matrices compact on disk;
    # the downstream ranker-training stages read this same pickle back.
    with open(out, "wb") as f:
        pickle.dump(case_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Pool recall@POOL_K: fraction of cases whose gold track made it into the 300-candidate
    # pool (gt_pos >= 0). The ceiling on how well any ranker over this pool can score; logged only.
    recall = sum(1 for cf in case_features.values() if cf["gt_pos"] >= 0) / len(case_features)
    print(f"[case_features] wrote {len(case_features)} cases -> {out} "
          f"(large_in_pool={large_in_pool}, pool_recall@{C.POOL_K}={recall:.4f}, "
          f"{time.time() - t0:.0f}s)", flush=True)
