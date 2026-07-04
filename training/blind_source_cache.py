"""Precompute the per-blind-session base-source lists the BGE-large blind encode
(``training.struct_large``) consumes.

What this is
------------
A one-time cache of every struct_large-independent retriever output (qwen/bm25/cf-bpr + ALS +
text_retriever, plus the parsed session fields) for the blind set. The heavy ``struct_large``
blind encode loads this cache instead of re-running BM25 / qwen / cf-bpr / ALS / text-retriever
once per fold; the per-session record is also the format the downstream ensemble/candidate
tooling reads. Each record is::

    {session_id, turn_number, user_query, history, music_turns,
     src_a, src_b, src_c, src_d, src_f,
     als_tracks, als_vec (list[float] | None),
     r21_list, r21_rank_map}

(the ``src_*`` / ``r21_*`` keys are the on-disk record schema, kept verbatim for format
compatibility) and the consolidated artifact is ``{session_id: record}`` at
``C.BLIND_SRC_CACHE`` (``cache/blind_a/source_cache.pkl`` — the legacy path, kept so the encode
contract is unchanged).

Retrieval reuse — faithful to inference, and decoupled from struct_base/struct_large
------------------------------------------------------------------------------------
This runs exactly the *inference-side* base retrievers (BM25 ``topk=100``, qwen recent/neighbours
over the last 3 played, cf-bpr, ALS oldest-weighted ``0.8**i`` with an L2-normalised session
vector, text-retriever last-3-turns top-300 with played excluded) and *never* loads the
struct_base / struct_large ensemble lists. It reuses the SAME shared building blocks the live
recommender uses (``recommender.sources._VecIndex`` / ``_maxrecent_topn`` for qwen + cf-bpr;
``bm25s`` for the bm25 sources; the text-retriever ``SentenceTransformer`` + catalog embeddings;
the ALS factors) with the inference params — but deliberately does NOT instantiate
``recommender.sources.Sources``, whose ``__init__`` eagerly reads ``C.STRUCT_BASE_BLIND_LISTS``
and ``C.STRUCT_LARGE_ENSEMBLE``. That matters because ``struct_large.build`` calls this *before*
it writes ``C.STRUCT_LARGE_ENSEMBLE`` (the blind encode needs the cache to build the queries),
so depending on ``Sources`` would deadlock on a clean rebuild. The per-source math here is
identical to the corresponding ``Sources.src_*`` methods (verified against them), so the lists
match inference.

In particular the ALS path reproduces ``Sources.src_als`` exactly (oldest-weighted decay +
normalisation) — this is the *blind/inference* ALS weighting; the offline dev featuriser's
newest-weighted variant in ``training.case_features`` is a separate, documented divergence.
``r21_list`` is the text-retriever top-300 with played excluded and ``r21_rank_map`` is
``{tid: rank}`` over its first 300.

The cache is written atomically as the consolidated ``{sid: record}`` dict the encode consumes.
"""
from __future__ import annotations

import json
import pickle
import time
from datetime import datetime

import numpy as np

from recommender import config as C
from recommender.data import load_als, load_blind_sessions, load_track_metadata
from recommender.sources import _VecIndex, _maxrecent_topn
from recommender.text import query_parts

_TEXT_TOPK = 300         # text-retriever blind list length (top-300 + rank_map)
_BM25_TOPK = 100         # bm25 blind list length (inference param)


def _ts() -> str:
    # Wall-clock timestamp prefix for the human-readable progress logs below
    # (e.g. "[2026-06-04 17:52:01]"). Purely cosmetic; not parsed by anything.
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


class _BaseRetrievers:
    """The qwen/bm25/cf-bpr + ALS + text_retriever base retrievers — minus struct_base/struct_large.

    Each method reproduces the corresponding live ``recommender.sources.Sources.src_*`` exactly
    (same params), so the cached lists match inference; only the eager struct_base/struct_large
    ensemble loading of ``Sources.__init__`` is omitted (it is not needed here and would create
    an ordering deadlock with ``struct_large``).
    """

    def __init__(self, metadata, als):
        import bm25s
        from sentence_transformers import SentenceTransformer
        import torch

        self.metadata = metadata
        # ALS factors: (n_tracks, d) item-factor matrix; als_to_idx maps track_id -> row.
        self.als_factors, self.als_ids, self.als_to_idx = als

        # BM25 (B/C). load_corpus=True keeps the corpus alongside the index so retrieve()
        # can return document records; bm25_ids maps a retrieved doc id -> track_id.
        self._bm25s = bm25s
        self.bm25 = bm25s.BM25.load(str(C.BM25_INDEX), load_corpus=True)
        self.bm25_ids = json.loads((C.BM25_INDEX / "track_ids.json").read_text())

        # qwen3 (A', D) + cf-bpr (F): the same dense L2-normalised _VecIndex objects the
        # live sources use, so a dot product equals cosine similarity.
        self.qwen = _VecIndex(C.QWEN_DIR)
        self.cfbpr = _VecIndex(C.CFBPR_DIR)

        # text-retriever (r21 == text_retriever; legacy token, see module docstring) model +
        # precomputed L2-normalised catalog embeddings (text_embs: (n_tracks, d)). Prefer GPU.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_model = SentenceTransformer(str(C.TEXT_RETRIEVER_DIR / "model"), device=device)
        self.text_ids = json.loads((C.TEXT_RETRIEVER_DIR / "track_ids.json").read_text())
        self.text_embs = np.load(C.TEXT_RETRIEVER_DIR / "track_embeddings.npy")

    # --- per-source retrieval (verbatim with Sources.src_*) ---
    # bm25_retrieve: tokenize a lowercased query string and pull the top-`topk`
    # documents from the prebuilt BM25 index, mapping each retrieved doc id back to
    # its track_id via bm25_ids. Shared helper for both BM25 query variants below.
    def bm25_retrieve(self, query, topk):
        toks = self._bm25s.tokenize([query.lower()])
        res = self.bm25.retrieve(toks, k=topk, return_as="tuple")
        return [self.bm25_ids[item["id"]] for item in res.documents[0]]

    # qwen_recent (source A'): max-pool the qwen vectors of the last A_PRIME_RECENT_K
    # played tracks and rank the catalog by best similarity to any of them. Returns
    # the top-`topn` track_ids; empty handled by the caller when there is no history.
    def src_qwen_recent(self, played, topn=100):
        return _maxrecent_topn(played, C.A_PRIME_RECENT_K, self.qwen.vectors,
                               self.qwen.id_to_idx, self.qwen.track_ids, topn)

    # qwen_neighbors (source D): nearest neighbours of the SINGLE most recent played
    # track (the anchor) in qwen space. No history -> no anchor -> empty list.
    def src_qwen_neighbors(self, played, topk=100):
        anchor = played[-1] if played else None
        return self.qwen.neighbors(anchor, topk) if anchor else []

    # cfbpr_recent (source F): same max-recent scheme as qwen_recent but over the
    # collaborative-filtering BPR vectors and CFBPR_RECENT_K most-recent tracks.
    def src_cfbpr_recent(self, played, topn=100):
        return _maxrecent_topn(played, C.CFBPR_RECENT_K, self.cfbpr.vectors,
                               self.cfbpr.id_to_idx, self.cfbpr.track_ids, topn)

    # bm25 (sources B and C): two BM25 lookups built from DIFFERENT query assemblies.
    # q_b ("last_music_meta") = source B (bm25_lastmusic): query built from the metadata
    # of the most recent music turn. q_c ("full") = source C (bm25_convo): query built
    # from the whole conversation. query_parts() (recommender.text) does the assembly;
    # `or user_query` falls back to the raw request if a part list comes back empty.
    def src_bm25(self, history, user_query):
        hd = [{"role": h["role"], "content": h["content"]} for h in history]
        q_b = " ".join(query_parts(hd, user_query, self.metadata, "last_music_meta"))
        q_c = " ".join(query_parts(hd, user_query, self.metadata, "full"))
        return (self.bm25_retrieve(q_b or user_query, _BM25_TOPK),
                self.bm25_retrieve(q_c or user_query, _BM25_TOPK))

    def src_als(self, played, topk=200):
        # Build a single session vector as a weighted average of the played tracks' ALS
        # item-factors, then rank the whole catalog by dot product against it.
        idxs = [self.als_to_idx[t] for t in played if t in self.als_to_idx]
        if not idxs:
            return [], None
        # Oldest-weighted decay: weight 0.8**i with i=0 for the FIRST (oldest) played track,
        # so earlier tracks dominate the session vector. This is the blind/inference ALS
        # weighting (it matches Sources.src_als); the dev featuriser uses a newest-weighted
        # variant instead — a deliberate, separately-documented divergence (see module docstring).
        w = np.array([0.8 ** i for i in range(len(idxs))], dtype=np.float32)
        w /= w.sum()
        sv = np.zeros(self.als_factors.shape[1], dtype=np.float32)
        for wi, idx in zip(w, idxs):
            sv += wi * self.als_factors[idx]
        norm = np.linalg.norm(sv)
        # Guard against a (near-)zero session vector (no usable factors / cancellation).
        if norm < 1e-8:
            return [], None
        sv /= norm  # legacy normalization (matches Sources.src_als / als_retrieve_simple)
        scores = self.als_factors @ sv  # (n_tracks,) dot product to the session vector
        # Never recommend an already-played track: push its score to -inf before ranking.
        for t in played:
            if t in self.als_to_idx:
                scores[self.als_to_idx[t]] = -np.inf
        # O(n) top-k: argpartition isolates the top `topk` (unordered), then sort just those.
        top = np.argpartition(-scores, topk)[:topk]
        top = top[np.argsort(-scores[top])]
        # Return both the ranked ids AND the normalised session vector sv: the encode caches
        # als_vec so it can be reused downstream without recomputing the weighted average.
        return [self.als_ids[j] for j in top], sv

    def src_text_retriever(self, history, user_query, played_set, topk=300):
        # Query = the last TEXT_QUERY_TURNS user utterances (prior user turns + current
        # request), encoded by the fine-tuned text retriever and matched against the
        # precomputed catalog embeddings. normalize_embeddings=True so the dot product below
        # is cosine similarity (text_embs are already L2-normalised).
        parts = [str(h["content"]) for h in history if h["role"] == "user"] + [user_query]
        q = " ".join(parts[-C.TEXT_QUERY_TURNS:])
        qe = self.text_model.encode([q], normalize_embeddings=True).astype(np.float32)[0]
        scores = self.text_embs @ qe  # (n_tracks,) cosine to the query
        # Exclude already-played tracks (-inf) so they cannot appear in the retrieved list.
        for ti, tid in enumerate(self.text_ids):
            if tid in played_set:
                scores[ti] = -np.inf
        top = np.argpartition(-scores, topk)[:topk]
        top = top[np.argsort(-scores[top])]
        return [self.text_ids[j] for j in top]


def _build_one(session: dict, r: "_BaseRetrievers") -> dict:
    """Run every struct_large-independent retriever for one blind session (verbatim record).

    ``session`` is a ``recommender.data.parse_last_turn`` dict. Sources use the inference
    params (qwen recent-3, BM25 top-100, ALS top-200 oldest-weighted, text-retriever top-300
    played-excluded).
    """
    sid = str(session["session_id"])
    history = session["history"]
    user_query = session["user_query"]
    music_turns = session["music_turns"]
    played_set = set(music_turns)

    src_a = r.src_qwen_recent(music_turns, topn=100) if music_turns else []
    src_d = r.src_qwen_neighbors(music_turns, topk=100) if music_turns else []
    src_f = r.src_cfbpr_recent(music_turns, topn=100) if music_turns else []
    src_b, src_c = r.src_bm25(history, user_query)
    als_tracks, als_vec = r.src_als(music_turns, topk=200)
    text_list = r.src_text_retriever(history, user_query, played_set, topk=_TEXT_TOPK)

    # dict KEYS (src_a..src_f, als_*, r21_list, r21_rank_map) are the on-disk record schema of
    # the source-cache (cache/blind_a/source_cache.pkl) — kept verbatim for format compatibility.
    return {
        "session_id": sid,
        "turn_number": session["turn_number"],
        "user_query": user_query,
        "history": history,
        "music_turns": music_turns,
        "src_a": src_a,
        "src_b": src_b,
        "src_c": src_c,
        "src_d": src_d,
        "src_f": src_f,
        "als_tracks": als_tracks,
        # Store als_vec as a plain list[float] (not ndarray) so the pickle stays portable;
        # None when src_als found no usable factors.
        "als_vec": als_vec.tolist() if als_vec is not None else None,
        # r21_list / r21_rank_map (r21 == text_retriever; legacy token kept for the on-disk
        # record schema): the top-300 retrieved ids, plus a {track_id: 1-based rank} map over
        # them. Ranks are 1-based (r_ + 1) so a smaller value means a better position; this is
        # the rank the downstream featuriser reads to build the text-retriever rank feature.
        "r21_list": text_list,
        "r21_rank_map": {tid: r_ + 1 for r_, tid in enumerate(text_list[:_TEXT_TOPK])},
        "conversation_goal": session.get("conversation_goal"),
    }


def build(force: bool = False, smoke: bool = False, blind_name: str = "blind_a") -> dict:
    """Precompute the blind base-source cache -> ``C.BLIND_SRC_CACHE``.

    No-op when the consolidated cache already exists unless ``force=True`` (``struct_large``'s
    blind encode calls this only when the cache is missing). Returns the consolidated
    ``{session_id: record}`` dict.

    ``smoke=True`` builds only the first few blind sessions (the same handful ``struct_large``'s
    smoke ensemble uses) so the path runs in seconds; it writes the cache for those sessions and
    does NOT treat a full cache as present (a later non-smoke ``build(force=True)`` produces the
    full 80-session cache).
    """
    cache_path = C.BLIND_SRC_CACHE
    # Skip only on a real (non-smoke) cache hit: a present full cache is reused as-is, but a
    # smoke run must not be short-circuited by (or mistaken for) the full cache.
    if cache_path.exists() and not force and not smoke:
        print("  [skip] blind source cache present"); return pickle.load(open(cache_path, "rb"))

    t0 = time.time()
    print(f"{_ts()} [blind_source_cache] loading {blind_name} sessions + base retrievers …",
          flush=True)
    sessions = load_blind_sessions(blind_name)
    if smoke:
        sessions = sessions[:8]
    print(f"  {len(sessions)} sessions (smoke={smoke})", flush=True)

    metadata = load_track_metadata()
    als = load_als()
    retrievers = _BaseRetrievers(metadata, als)

    consolidated: dict[str, dict] = {}
    for i, session in enumerate(sessions):
        record = _build_one(session, retrievers)
        consolidated[record["session_id"]] = record
        # Progress/ETA via a simple linear extrapolation: average seconds per
        # session so far (sec_per) times the number of sessions still remaining.
        # Logging only — does not affect the cache contents.
        elapsed = time.time() - t0
        sec_per = elapsed / (i + 1)
        eta = (len(sessions) - i - 1) * sec_per
        print(f"  [{i + 1}/{len(sessions)}] sid={record['session_id'][:8]}  "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: pickle to a sibling ".tmp" then rename into place. The rename is atomic on
    # the same filesystem, so a reader (e.g. struct_large's encode) never sees a half-written
    # cache even if this build is interrupted mid-dump.
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(consolidated, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(cache_path)
    print(f"{_ts()} [blind_source_cache] wrote {len(consolidated)} sessions -> {cache_path} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return consolidated
