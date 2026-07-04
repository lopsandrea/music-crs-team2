"""Data layer: TalkPlayData catalog/metadata, feature support maps, ALS cache,
session parsing, and the recommend-item renderer for response_gen.

Loaders that read shipped artifacts (maps/pop/als) and the HF catalog. All
ported faithfully from the validated pipeline.

Role in the pipeline: this module backs Level 0 ("parse session") and supplies the
read-only lookup structures the later levels need. At Recommender construction time
load_track_metadata / load_supporting_maps / load_als populate the catalog, the
per-track feature-support maps and the ALS item factors once; parse_last_turn /
load_blind_sessions turn each raw HF session into the per-case record L1-L6 consume.
Nothing here is GPU/heavy — that all lives in `training`, which WRITES the artifacts
(metadata_maps.pkl, track_popularity.json, als_factors.npz) that these loaders READ.
"""
from __future__ import annotations

import json
import pickle

import numpy as np

from . import config as C


def _load_hf(name: str, **kwargs):
    """Load a (public TalkPlayData) HF dataset: use the local cache if present — reproducible
    and offline — else download it on a fresh machine's first run."""
    from datasets import DownloadConfig, load_dataset
    try:
        return load_dataset(name, download_config=DownloadConfig(local_files_only=True), **kwargs)
    except Exception:  # not in the local HF cache -> fetch it (colleague's first run)
        return load_dataset(name, **kwargs)


# ---------------- catalog ----------------

def load_track_metadata() -> dict[str, dict]:
    """{track_id: metadata_dict} over all_tracks (local HF cache)."""
    ds = _load_hf(C.DS_TRACK_META)["all_tracks"]
    return {str(item["track_id"]): item for item in ds}


def load_track_albums():
    """Map track_id -> scalar album string (album_id[0] or album_name[0], "" if none).

    The catalog stores album_id/album_name as LISTS (a track can appear on several releases);
    this collapses each to a single key for the "same-album" feature. album_id is preferred over
    album_name because IDs are exact-match-safe (no spelling/casing drift across tracks).
    """
    ds = _load_hf(C.DS_TRACK_META)["all_tracks"]
    track_album = {}
    for item in ds:
        tid = str(item["track_id"])
        alb_id = item.get("album_id", [])
        alb_name = item.get("album_name", [])
        # Prefer the first album_id; fall back to the first album_name; "" when neither exists.
        if isinstance(alb_id, list) and alb_id:
            track_album[tid] = str(alb_id[0])
        else:
            track_album[tid] = str(alb_name[0]) if isinstance(alb_name, list) and alb_name else ""
    return track_album


def render_recommend_item(track_id: str, metadata: dict[str, dict]) -> str:
    """Compact metadata string fed to response_gen as the recommended track.

    Produces a single "key: value, key: value, ..." line (e.g. the top-1 track) that the Gemini
    response generator quotes when writing its natural-language reply. Only non-empty fields are
    emitted; tags are capped at 6 to keep the prompt fragment short.
    """
    m = metadata.get(str(track_id), {})
    # Catalog fields may be lists (multi-valued) or scalars; `first` normalises both to a single
    # display value: first element of a non-empty list, the scalar itself, or "" for an empty list.
    def first(v):
        return v[0] if isinstance(v, list) and v else (v if not isinstance(v, list) else "")
    parts = [f"track_id: {track_id}"]
    for k in ("track_name", "artist_name", "album_name", "release_date"):
        val = first(m.get(k, ""))
        if val:
            parts.append(f"{k}: {str(val)}")
    tags = m.get("tag_list", [])
    if isinstance(tags, list) and tags:
        parts.append("genres: " + ", ".join(str(t) for t in tags[:6]))
    return ", ".join(parts)


# ---------------- feature support (shipped artifacts) ----------------

def load_supporting_maps():
    """Load the shipped feature-support artifacts used during L3 featurisation.

    Returns (maps, track_pop, track_album):
      maps        — pickled lookup dicts built by `training` (e.g. co-occurrence / id maps).
                    Concretely a dict keyed by feature name; the recommender unpacks five
                    per-track entries in a FIXED positional order for L3 featurisation:
                    track_artist, track_tags, track_title_toks, track_artist_toks,
                    track_meta_toks (each {track_id: value}). These drive the text-overlap /
                    same-artist / same-tag features in recommender/features.py.
      track_pop   — {track_id: popularity} JSON (a global-prior feature). The recommender
                    divides by max(track_pop.values()) to normalise the popularity feature
                    into [0, 1] (features.py col 23), so absolute play counts don't dominate.
      track_album — track_id -> album key (see load_track_albums), the same-album signal.
    """
    # Binary pickle: maps may hold object/dict values, so it MUST be opened "rb".
    with open(C.PAYLOAD_MAPS, "rb") as f:
        maps = pickle.load(f)
    with open(C.TRACK_POP) as f:
        track_pop = json.load(f)
    track_album = load_track_albums()
    return maps, track_pop, track_album


def load_als():
    """Load the ALS collaborative-filtering item factors used by the als_session source.

    Returns (factors, ids, to_idx):
      factors — float32 array, shape (n_tracks, n_latent_factors); row i is track ids[i]'s vector.
      ids     — list[str] of track_ids, aligned to `factors` rows (positional).
      to_idx  — {track_id: row index} for O(1) lookup into `factors`.
    allow_pickle=True is required because track_ids is stored as an object array.
    """
    data = np.load(C.ALS_NPZ, allow_pickle=True)
    factors = np.asarray(data["factors"], dtype=np.float32)
    ids = [str(t) for t in data["track_ids"].tolist()]
    to_idx = {t: i for i, t in enumerate(ids)}
    return factors, ids, to_idx


# ---------------- session parsing ----------------

def parse_last_turn(item: dict) -> dict:
    """Parse one blind/dev session into the per-case record the pipeline (L0) consumes.

    The conversation is a mix of 'user', 'music', and 'assistant' turns ordered by turn_number.
    The LAST user turn is what we must answer:
      - user_query   = that final user turn's text (the live request).
      - turn_number  = its turn index.
      - history      = every turn BEFORE it (turn_number < turn_num), strictly the prior context.
      - music_turns  = the prior 'music' turns' content == the TRACK_IDS already played this
                       session (a 'music' turn stores a track_id, not prose); used both as the
                       "already-played" set to skip and as recency seeds for several sources.
    Sorting by turn_number first makes iloc[-1] the chronologically latest user turn regardless
    of the dataset's row order. user_profile is passed through verbatim when present.
    """
    import pandas as pd
    df = pd.DataFrame(item["conversations"]).sort_values("turn_number")
    user_rows = df[df["role"] == "user"]
    # iloc[-1] = the chronologically latest user turn (df is already turn-sorted). Invariant:
    # every challenge session ends on a user request, so user_rows is guaranteed non-empty here.
    last_user = user_rows.iloc[-1]
    turn_num = int(last_user["turn_number"])
    user_query = str(last_user["content"])
    # Everything strictly before the turn we are answering = the visible conversation history.
    prior = df[df["turn_number"] < turn_num]
    history = [{"role": str(r["role"]), "content": r["content"], "turn_number": int(r["turn_number"])}
               for _, r in prior.iterrows()]
    # 'music' turn content == a played track_id; these are the session's already-played tracks.
    music_turns = [str(r["content"]).strip() for _, r in prior.iterrows() if r["role"] == "music"]
    # conversation_goal.listener_goal — an explicit natural-language statement of what the
    # user wants this session (e.g. "find multiple energetic and feel-good pop songs").
    # Surfaced as `listener_goal` for convenience; note it is null in the Blind-B sessions.
    cg = item.get("conversation_goal") or {}
    listener_goal = str(cg.get("listener_goal", "")) if isinstance(cg, dict) else ""
    return {"session_id": str(item["session_id"]), "turn_number": turn_num,
            "user_query": user_query, "history": history, "music_turns": music_turns,
            "user_profile": item.get("user_profile"),
            "conversation_goal": item.get("conversation_goal"),
            "listener_goal": listener_goal}


def load_blind_sessions(blind_name: str) -> list[dict]:
    """Load and parse every session in a blind split (e.g. 'blind_a'/'blind_b') for inference.

    Reads the named HF blind dataset's 'test' split and maps each item through parse_last_turn,
    yielding the list of per-case records that Recommender.batch_recommend iterates over.
    """
    ds = _load_hf(C.BLIND_DATASETS[blind_name], split="test")
    return [parse_last_turn(item) for item in ds]
