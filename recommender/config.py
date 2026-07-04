"""Central configuration: paths to the shipped artifacts and the (frozen) pipeline constants.

Everything the inference pipeline reads lives under ``cache/`` at the repository root
(populated by ``scripts/download_weights.py``; see README). All candidate retrieval is
performed over the ENTIRE ``all_tracks`` catalog (47,071 tracks) — no split filtering of
any kind is applied at any stage.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repository root (this file lives at <root>/recommender/config.py). RECSYS_ROOT
# overrides it for callers that relocate the artifact tree.
ROOT = Path(os.environ.get("RECSYS_ROOT", Path(__file__).resolve().parents[1]))
CACHE = ROOT / "cache"

# --- HuggingFace datasets (downloaded on first run, cached afterwards) ---
# DS_TRACK_META: the official per-track catalog (title/artist/album/tags), `all_tracks` split.
DS_TRACK_META = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
# DS_TRACK_EMB: the organizer-provided per-track embedding columns (qwen3-metadata / cf-bpr /
# CLAP); repackaged by training.base_caches into the vectors.npy indices under cache/track_sim/.
DS_TRACK_EMB = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
BLIND_DATASETS = {
    "blind_a": "talkpl-ai/TalkPlayData-Challenge-Blind-A",
    "blind_b": "talkpl-ai/TalkPlayData-Challenge-Blind-B",
}
# Which blind split the precomputed struct_base/struct_large list files resolve to.
# This release targets Blind-B (the split of our validated submission).
RECSYS_BLIND_NAME = os.environ.get("RECSYS_BLIND_NAME", "blind_b")
_BLIND_SUFFIX = "" if RECSYS_BLIND_NAME == "blind_a" else f"_{RECSYS_BLIND_NAME}"

# --- fusion (Level 2) ---
# Per-source weighted-RRF fusion weights. qwen_recent / qwen_neighbors are kept in the
# source set as ranker-feature providers (their rank/presence feed Level-3 features) but
# carry zero fusion weight; clap_recent joins the pool at half weight (acoustic recall).
SW_BASELINE = {"qwen_recent": 0.0, "bm25_lastmusic": 1.0, "bm25_convo": 1.0,
               "qwen_neighbors": 0.0, "cfbpr_recent": 1.0, "als_session": 1.0,
               "text_retriever": 1.0, "struct_base": 1.0, "struct_large": 2.0,
               "clap_recent": 0.5}
RRF_K = 20          # RRF damping constant: rank r contributes weight / (RRF_K + r)
POOL_K = 300        # fused recall-pool size handed to the rankers
TOP_K = 20          # final recommendation list length (nDCG@20)

# --- ranking (Levels 4-6) ---
ROUTE_LOW = 0.25    # base-ranker top1-top2 margin below which lgbm_large is routed in
ROUTE_HIGH = 1.5    # ... and at/above which lgbm_large is routed in (over-confidence guard)
CB_W = 1.0          # z-space blend weight of the CatBoost-YetiRank ensemble scores

# --- source hyperparameters (Level 1) ---
A_PRIME_RECENT_K = 3   # qwen_recent / clap_recent: max-cosine over the last K played
CFBPR_RECENT_K = 3     # cfbpr_recent: same, in the CF-BPR space
TEXT_QUERY_TURNS = 3   # text_retriever: last K user turns form the query

# --- shipped artifacts (all read-only at inference) ---
# Dense embedding indices (vectors.npy + track_ids.json, L2-normalised rows).
QWEN_DIR = CACHE / "track_sim" / "metadata-qwen3_embedding_0.6b"  # semantic metadata space
CFBPR_DIR = CACHE / "track_sim" / "cf-bpr"                        # collaborative item space
CLAP_DIR = CACHE / "track_sim" / "audio-laion_clap"               # acoustic (CLAP) space
# Lexical BM25 index over track name + artist + album + tags.
BM25_INDEX = CACHE / "bm25" / "track_name_artist_name_album_name_tag_list"
# ALS matrix-factorization item factors (als_session source + the als_dot feature).
ALS_NPZ = CACHE / "als_factors.npz"
# Featurization support artifacts.
TRACK_POP = CACHE / "track_popularity.json"      # global popularity prior
PAYLOAD_MAPS = CACHE / "metadata_maps.pkl"       # per-track artist/tags/token lookup maps
# text_retriever (fine-tuned bi-encoder): model runs LIVE at inference; the catalog side
# is pre-encoded (track_embeddings.npy).
TEXT_RETRIEVER_DIR = CACHE / "retrievers" / "text_retriever"
# struct_base / struct_large: 5-fold fine-tuned BGE retrievers. No model runs at
# inference — their per-session blind outputs are precomputed offline (see the
# from-scratch reproduction section of the README) and read as JSON here.
STRUCT_BASE_BLIND_LISTS = CACHE / "retrievers" / "struct_base" / f"blind_lists{_BLIND_SUFFIX}.json"
STRUCT_LARGE_ENSEMBLE = CACHE / "retrievers" / "struct_large" / f"blind_lists{_BLIND_SUFFIX}.json"
# GBDT rankers: two LightGBM LambdaRank boosters + the CatBoost-YetiRank ensemble,
# trained on 37-column feature matrices over unfiltered all_tracks candidate pools.
LGBM_BASE = CACHE / "rankers" / "lgbm_base.txt"
LGBM_LARGE = CACHE / "rankers" / "lgbm_large.txt"
CB_MODEL = CACHE / "rankers" / "catboost_ensemble.cbm"

# --- training-side paths (read/written ONLY by the `training` package; see TRAINING.md) ---
# The conversation sessions dataset (train split = ranker/retriever training data; its held-out
# split provides the 8000 dev cases dev_payload.py parses).
DS_CONVO = "talkpl-ai/TalkPlayData-Challenge-Dataset"
# Canonical parsed dev cases (the ~8000 sessions each split into context + held-out next track);
# every downstream training stage reads cases from here. Never touched at inference.
DEV_PAYLOAD = CACHE / "eval" / "dev_payload.pkl"
# Per-case candidate-pool feature matrices. Two variants differing only in pool composition
# (see training/case_features.py): the 8-source pool trains the CatBoost ensemble; the 9-source
# pool (struct_large also unioned into the recall pool — the live serving pool) trains the
# LightGBM LambdaRank rankers.
CASE_FEATURES = CACHE / "training" / "case_features.pkl"
CASE_FEATURES_LARGE_POOL = CACHE / "training" / "case_features_r84pool.pkl"
# Per-case out-of-fold retriever lists (offline featurisation MUST use OOF lists, not the
# blind lists, so the dev cases the rankers train on are never scored by a model that saw them).
TEXT_RETRIEVER_OOF = TEXT_RETRIEVER_DIR / "dev_oof_lists.json"     # [n_cases][track_id]
STRUCT_BASE_OOF = CACHE / "retrievers" / "struct_base" / "oof_lists.json"  # {"lists": [[ [tid, score] ]]}
STRUCT_LARGE_OOF_DIRS = [CACHE / "retrievers" / "struct_large" / f"oof_fold_{k}.json"
                         for k in range(5)]                        # {case_idx_str: [[tid, score]]}
# Per-fold working dirs (model/ + embeddings) written by the retriever trainers and reused by
# their blind-list encodes. Legacy layout tokens (r54 == struct_base) are locked, see README.
STRUCT_BASE_FOLD_DIRS = [CACHE / ("r54/phase3_smoke/fold_0" if k == 0 else f"r54/phase3_full/fold_{k}")
                         for k in range(5)]
STRUCT_LARGE_FOLD_DIRS = [CACHE / "retrievers" / "struct_large" / f"fold_{k}" for k in range(5)]
# Blind base-source cache: per-blind-session precomputed source lists + encode inputs, built by
# training/blind_source_cache and consumed by the struct_large blind-ensemble encode.
BLIND_SRC_CACHE = CACHE / RECSYS_BLIND_NAME / "source_cache.pkl"
# Transfer-weighted final ranker training (training/transfer_weighting.py): adversarial
# dev-vs-blind case weights and the tempering exponent used for the shipped rankers.
TRANSFER_WEIGHTS = CACHE / "training" / "transfer_case_weights.pkl"
TRANSFER_ALPHA = 0.25
