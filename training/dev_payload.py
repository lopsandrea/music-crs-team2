"""Reconstruct the dev-case payload the offline featuriser (``training.case_features``)
and the LambdaRank/CatBoost training consume — clean, config-driven, **A/B/C/D/F only**.

Three artifacts are written:

* ``C.DEV_PAYLOAD`` (``cache/eval/dev_payload.pkl``) — the per-dev-case payload::

      {"cases":  [ {session_id,user_id,turn_number,user_query,history,
                    music_turns,gt,n_prior_music,
                    conversation_goal,gold_turn_assessment}, ... ]   # 8000 dev cases (all user turns w/ GT)
       "src_a":  [n_cases][track_id],   # qwen3 max-recent-5 (A')
       "src_b":  [n_cases][track_id],   # BM25 last-music-meta query (B)
       "src_c":  [n_cases][track_id],   # BM25 full-conversation query (C)
       "src_d":  [n_cases][track_id],   # qwen3 last-track neighbours (D)
       "src_f":  [n_cases][track_id],   # cf-bpr max-recent-5 (F)
       "track_artist","track_tags","track_title_toks",
       "track_artist_toks","track_meta_toks": {track_id: ...}}   # the 5 metadata maps

  ``case_features._PAYLOAD_KEYS`` reads ``src_a/b/c/d/f`` here. **The dead ``src_g``
  (session-cooccur) source is dropped** — the champion fuses only A/B/C/D/F as its
  non-supervised candidate sources, so ``src_g`` never enters any RRF pool.

* ``C.PAYLOAD_MAPS`` (``cache/metadata_maps.pkl``) — the same 5 maps, the exact dict
  ``recommender.data.load_supporting_maps`` (inference featuriser) reads:
  ``{track_artist, track_tags, track_title_toks, track_artist_toks, track_meta_toks}``.
  ``track_artist`` -> lowercased ``str(artist_name)``; ``track_tags`` -> set of lowercased
  tags; the ``*_toks`` -> ``recommender.text.tokens`` sets.

* ``C.TRACK_POP`` (``cache/track_popularity.json``) — ``{track_id: train_music_turn_count}``
  (``recommender.data.load_supporting_maps`` reads it via ``json.load``).

Retrieval reuse — A/B/C/D/F are produced with the SAME building blocks the live
recommender uses (``recommender.sources._VecIndex`` / ``_maxrecent_topn`` for qwen A'/D +
cf-bpr F; ``recommender.text.query_parts`` for the BM25 B/C queries), driven with the
*legacy A/B/C/D/F parameters* this dev payload was built with: A'/F over the last **5**
played (vs the live sources' ``A_PRIME_RECENT_K``/``CFBPR_RECENT_K``=3), top-**200** lists,
and BM25 ``topk=500`` (vs the live 100). Those param differences are why the live
``Sources.src_*`` methods are not called directly — but the underlying math is shared, so
this reproduces the shipped ``_R12_all_turns_payload.pkl`` A/B/C/D/F lists (verified:
top-50 of A'/D/F identical; deep tail differs only by ~1e-7 cosine float-drift in the
embeddings). The metadata-map construction is ported verbatim from the validated builder
(``scripts/expR12_bucket_policies.build_payload``), and the popularity counter from
``scripts/expS2_lr_v2.build_popularity_stats``.
"""
from __future__ import annotations

import json
import pickle
import time
from collections import Counter

import numpy as np

from recommender import config as C
from recommender.data import load_track_metadata

def build_assess_map(item: dict) -> dict[int, str]:
    """{turn_number: goal_progress_assessment} for one raw dataset session (None entries dropped)."""
    out: dict[int, str] = {}
    for a in item.get("goal_progress_assessments") or []:
        tn = a.get("turn_number")
        val = a.get("goal_progress_assessment")
        if tn is not None and val is not None:
            out[int(tn)] = val
    return out

# _VecIndex/_maxrecent_topn are the SAME building blocks the live recommender uses for the
# qwen3 (A'/D) and cf-bpr (F) embedding sources — reused here so the dev payload's recall is
# bit-for-bit consistent with inference (differences are only the legacy params; see docstring).
from recommender.sources import _VecIndex, _maxrecent_topn
# query_parts builds the BM25 B/C query strings; tokens() builds the *_toks bag-of-tokens sets.
# meta_text is imported from the shared text toolkit but is NOT called below: the metadata-map
# construction here inlines the raw catalog fields verbatim from the validated builder
# (scripts/expR12_bucket_policies.build_payload) rather than going through meta_text.
from recommender.text import meta_text, query_parts, tokens

# Legacy A/B/C/D/F params this dev payload was built with (see module docstring).
_RECENT_K = 5            # A'/F: max cosine over the last 5 played
_SRC_TOPN = 200          # A'/D/F list length
_BM25_TOPK = 500         # B/C list length


# ---------------- dev-case loading (all user turns with ground truth) ----------------

def _cached_test_arrow_path():
    """Path to the HF-cached dev/test split arrow (the 8000-case devset).

    Ensures the Challenge dataset is present first (downloading it from the Hub on a
    fresh machine), then memory-maps its prepared test-split arrow directly — the
    downstream loader relies on the arrow's exact on-disk row order.
    """
    from pathlib import Path

    from datasets.config import HF_DATASETS_CACHE
    from recommender.data import _load_hf

    # Populate the local HF cache if needed (no-op when already cached).
    _load_hf(C.DS_CONVO)
    hf_cache = Path(HF_DATASETS_CACHE)
    matches = sorted(hf_cache.glob(
        "talkpl-ai___talk_play_data-challenge-dataset/default/*/*/*-test.arrow"))
    if not matches:
        raise FileNotFoundError(
            "No cached *-test.arrow for the Challenge dataset under "
            f"{hf_cache} even after load_dataset — unexpected HF cache layout.")
    return str(matches[-1])


def _build_ground_truth(ds) -> dict:
    """{(session_id[,user_id]) -> {turn_number: gold_track_id}} from the music turns.

    Two parallel indexes are built so ``_lookup_gt`` can prefer the more specific
    ``(session_id, user_id)`` key (when a ``user_id`` is present) and fall back to the
    plain ``session_id`` key — see ``_lookup_gt``. The gold track for a user turn at
    ``turn_number=t`` is the music turn that immediately follows it, so every music turn
    is recorded under its own ``turn_number`` here.
    """
    by_session: dict = {}
    by_user: dict = {}
    for item in ds:
        sid = item["session_id"]
        uid = item.get("user_id")
        by_session[sid] = {}
        if uid is not None:
            by_user[(sid, uid)] = {}
        for conv in item["conversations"]:
            if conv["role"] == "music":
                turn = int(conv["turn_number"])
                tid = conv["content"].strip()
                by_session[sid][turn] = tid
                if uid is not None:
                    by_user[(sid, uid)][turn] = tid
    return {"session": by_session, "session_user": by_user}


def _lookup_gt(gt: dict, sid: str, uid, turn: int):
    # Prefer the user-scoped gold (most specific) and only fall back to the
    # session-scoped map when there is no user_id or no per-user entry for this turn.
    if uid is not None:
        found = gt["session_user"].get((sid, uid), {}).get(turn)
        if found is not None:
            return found
    return gt["session"].get(sid, {}).get(turn)


def load_dev_cases(ds: "Dataset | None" = None) -> list[dict]:
    """Expand the devset into one case per user turn that has a ground-truth next track.

    Ported from ``scripts/expR12_bucket_policies.load_all_turn_cases``: every user turn
    becomes a case whose ``history`` is the prior conversation and whose ``gt`` is the
    gold track for that turn (``music_turns`` = prior played tracks).
    """
    from datasets import Dataset
    if ds is None:
        ds = Dataset.from_file(_cached_test_arrow_path())
    gt_map = _build_ground_truth(ds)
    cases: list[dict] = []
    for item in ds:
        sid = str(item["session_id"])
        uid = item.get("user_id")
        # Conversations may arrive unordered; sort by turn_number so `history` (strictly
        # earlier turns) and `music` (prior played tracks) are reconstructed correctly.
        convs = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))
        assess = build_assess_map(item)
        conversation_goal = item.get("conversation_goal")  # session-level: same for every turn
        for ut in [c for c in convs if c["role"] == "user"]:
            turn = int(ut["turn_number"])
            query = str(ut["content"])
            # history = everything strictly before this user turn; music_turns = the
            # already-played tracks within that history (the retrieval anchors).
            history = [c for c in convs if int(c["turn_number"]) < turn]
            music = [str(c["content"]).strip() for c in history if c["role"] == "music"]
            gt = _lookup_gt(gt_map, sid, uid, turn)
            # Skip user turns with no following music turn — they have no gold target,
            # so they cannot be scored and would only add noise to training.
            if not gt:
                continue
            cases.append({
                "session_id": sid, "user_id": uid, "turn_number": turn,
                "user_query": query, "history": history, "music_turns": music,
                "gt": str(gt), "n_prior_music": len(music),
                "conversation_goal": conversation_goal,
                "gold_turn_assessment": assess.get(turn),
            })
    return cases


# ---------------- candidate sources (A/B/C/D/F) + metadata maps ----------------

def _bm25():
    """Load the BM25 index (the same one ``recommender.sources`` uses)."""
    import bm25s
    model = bm25s.BM25.load(str(C.BM25_INDEX), load_corpus=True)
    ids = json.loads((C.BM25_INDEX / "track_ids.json").read_text())
    return model, ids, bm25s


def _bm25_retrieve_batch(model, ids, bm25s, queries: list[str], topk: int) -> list[list[str]]:
    """Retrieve top-`topk` track_ids for every query in one batched BM25 call.

    Returns a list parallel to `queries`: result[i] is the ranked track_id list for
    queries[i]. `bm25s` retrieves by integer corpus row id, so each returned `item["id"]`
    is mapped back through `ids` (the corpus row -> track_id table) to the catalog id.
    Queries are lowercased before tokenising to match how the index was built.
    """
    if not queries:
        return []
    # tokenise all queries at once (lowercased to match index-time tokenisation).
    toks = bm25s.tokenize([q.lower() for q in queries])
    # res.documents is (n_queries, topk) of {"id": corpus_row, "score": ...} dicts.
    res = model.retrieve(toks, k=topk, return_as="tuple")
    # Translate each corpus row id back to its track_id via the parallel `ids` table.
    return [[ids[item["id"]] for item in res.documents[i]] for i in range(len(queries))]


def build_payload(cases: list[dict]) -> dict:
    """Build the A/B/C/D/F source lists + the 5 metadata maps for ``cases`` (NO G)."""
    t0 = time.time()
    n = len(cases)
    metadata = load_track_metadata()

    # --- B/C: BM25 (last-music-meta query / full-conversation query) ---
    print("  [dev_payload] BM25 batch (B/C) …", flush=True)
    model, bm25_ids, bm25s = _bm25()
    q_b, q_c = [], []
    for c in cases:
        # B query = "last_music_meta" (metadata of the last played track + the request);
        # C query = "full" (the whole conversation). Same `query_parts` builder the live
        # bm25 sources use, so the B/C queries match inference.
        qb = " ".join(query_parts(c["history"], c["user_query"], metadata, "last_music_meta"))
        qc = " ".join(query_parts(c["history"], c["user_query"], metadata, "full"))
        # Fall back to the raw user request when query_parts yields an empty string
        # (e.g. a first turn with no history) so BM25 always gets a non-empty query.
        q_b.append(qb or c["user_query"])
        q_c.append(qc or c["user_query"])
    src_b = _bm25_retrieve_batch(model, bm25_ids, bm25s, q_b, _BM25_TOPK)
    src_c = _bm25_retrieve_batch(model, bm25_ids, bm25s, q_c, _BM25_TOPK)

    # --- A'/D: qwen3 (max-recent-5 / last-track neighbours) ---
    print("  [dev_payload] qwen3 A'/D …", flush=True)
    qwen = _VecIndex(C.QWEN_DIR)
    src_a, src_d = [], []
    for c in cases:
        played = c["music_turns"]
        # A' = max-cosine over the last `_RECENT_K`=5 played (multi-anchor OR of recent
        # interests). Empty `played` (e.g. a first user turn) yields an empty source list.
        src_a.append(_maxrecent_topn(played, _RECENT_K, qwen.vectors, qwen.id_to_idx,
                                     qwen.track_ids, _SRC_TOPN) if played else [])
        # D = item-to-item neighbours of the single most-recent played track (the anchor),
        # so it only uses played[-1]; no anchor -> empty list.
        anchor = played[-1] if played else None
        src_d.append(qwen.neighbors(anchor, _SRC_TOPN) if anchor else [])

    # --- F: cf-bpr (max-recent-5) ---
    print("  [dev_payload] cf-bpr F …", flush=True)
    cfbpr = _VecIndex(C.CFBPR_DIR)
    src_f = []
    for c in cases:
        played = c["music_turns"]
        src_f.append(_maxrecent_topn(played, _RECENT_K, cfbpr.vectors, cfbpr.id_to_idx,
                                     cfbpr.track_ids, _SRC_TOPN) if played else [])

    # --- metadata maps (over the union of retrieved candidates; A/B/C/D/F + played) ---
    print("  [dev_payload] metadata maps …", flush=True)
    track_artist: dict[str, str] = {}
    track_tags: dict[str, set] = {}
    track_title_toks: dict[str, set] = {}
    track_artist_toks: dict[str, set] = {}
    track_meta_toks: dict[str, set] = {}
    all_tids: set[str] = set()
    # Only map tracks that can plausibly reach the fused candidate pool: the top of each
    # source list (BM25 B/C top-100, the embedding sources A'/D/F top-50) plus the played
    # tracks. Mapping the full source tails would be wasteful — those candidates rarely
    # survive fusion, and metadata-map construction is the slow part of this build.
    for i in range(n):
        all_tids.update(src_b[i][:100]); all_tids.update(src_c[i][:100])
        all_tids.update(src_a[i][:50]); all_tids.update(src_d[i][:50]); all_tids.update(src_f[i][:50])
        all_tids.update(cases[i]["music_turns"])
    for tid in all_tids:
        # `all_tids` is a set, so this guard is mostly defensive; the construction below is
        # idempotent per track, so skipping an already-mapped id just avoids redundant work.
        if tid in track_artist:
            continue
        meta = metadata.get(str(tid), {})
        artist = str(meta.get("artist_name", "")).lower().strip()
        raw_tags = meta.get("tag_list") or []
        tags = ({str(t).lower().strip() for t in raw_tags if str(t).strip()}
                if isinstance(raw_tags, list) else set())
        title = str(meta.get("track_name", ""))
        album = str(meta.get("album_name", ""))
        # track_meta_toks is the bag-of-tokens over title + artist + album + up to 12 tags.
        # The 12-tag cap bounds the per-track token set (some catalog tags are very long);
        # this construction is ported verbatim from the validated builder (see module docstring).
        meta_parts = [title, str(meta.get("artist_name", "")), album]
        if isinstance(raw_tags, list):
            meta_parts.extend(str(t) for t in raw_tags[:12])
        track_artist[tid] = artist
        track_tags[tid] = tags
        track_title_toks[tid] = tokens(title)
        track_artist_toks[tid] = tokens(meta.get("artist_name", ""))
        track_meta_toks[tid] = tokens(" ".join(meta_parts))

    print(f"  [dev_payload] built {n} cases in {time.time() - t0:.1f}s "
          f"({len(all_tids)} mapped tracks)", flush=True)
    return {
        "cases": cases, "src_a": src_a, "src_b": src_b, "src_c": src_c,
        "src_d": src_d, "src_f": src_f,
        "track_artist": track_artist, "track_tags": track_tags,
        "track_title_toks": track_title_toks, "track_artist_toks": track_artist_toks,
        "track_meta_toks": track_meta_toks,
    }


# ---------------- track popularity (train music-turn counts) ----------------

def _track_popularity() -> dict[str, int]:
    """``{track_id: count}`` over all music turns in the train split (Counter).

    This is the prior the inference featuriser reads from ``C.TRACK_POP`` — a global
    play-count popularity signal. It is counted over the TRAIN split (not the dev cases),
    so it is independent of which dev cases are built and is unaffected by ``smoke`` mode.
    """
    from datasets import DownloadConfig, load_dataset
    # local_files_only=False allows fetching the train split from the HF hub if it is not
    # already cached locally (unlike the dev split, which is read from the cached arrow above).
    train = load_dataset(C.DS_CONVO,
                         download_config=DownloadConfig(local_files_only=False))["train"]
    counts: Counter = Counter()
    # Count one play per music turn; the same track played across many sessions accumulates.
    for item in train:
        for c in item["conversations"]:
            if c["role"] == "music":
                counts[str(c["content"]).strip()] += 1
    return counts


# ---------------- build ----------------

def build(force: bool = False, smoke: bool = False) -> None:
    """Build the dev payload + the inference metadata maps + track popularity.

    Writes ``C.DEV_PAYLOAD``, ``C.PAYLOAD_MAPS`` and ``C.TRACK_POP``. No-op when all three
    already exist unless ``force=True``. ``smoke=True`` restricts to the first ~200 dev
    cases (fast end-to-end check); the maps/popularity are still produced (popularity is
    catalog-wide and independent of the case subset).
    """
    if (C.DEV_PAYLOAD.exists() and C.PAYLOAD_MAPS.exists() and C.TRACK_POP.exists()
            and not force):
        print("  [skip] dev payload + maps + track_pop present"); return

    print("[dev_payload] loading dev cases (all user turns with ground truth) …", flush=True)
    cases = load_dev_cases()
    if smoke:
        cases = cases[:200]
        print(f"  [smoke] limiting to {len(cases)} cases")
    else:
        print(f"  {len(cases)} dev cases", flush=True)

    payload = build_payload(cases)

    C.DEV_PAYLOAD.parent.mkdir(parents=True, exist_ok=True)
    with open(C.DEV_PAYLOAD, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Inference metadata maps (the 5 maps the featuriser reads) — re-emitted from the payload.
    maps = {k: payload[k] for k in ("track_artist", "track_tags", "track_title_toks",
                                    "track_artist_toks", "track_meta_toks")}
    C.PAYLOAD_MAPS.parent.mkdir(parents=True, exist_ok=True)
    with open(C.PAYLOAD_MAPS, "wb") as f:
        pickle.dump(maps, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("[dev_payload] track popularity …", flush=True)
    track_pop = _track_popularity()
    C.TRACK_POP.parent.mkdir(parents=True, exist_ok=True)
    with open(C.TRACK_POP, "w") as f:
        json.dump(track_pop, f)

    print(f"[dev_payload] wrote payload ({len(cases)} cases) -> {C.DEV_PAYLOAD}\n"
          f"             maps ({len(maps['track_artist'])} tracks) -> {C.PAYLOAD_MAPS}\n"
          f"             track_pop ({len(track_pop)} tracks) -> {C.TRACK_POP}", flush=True)
