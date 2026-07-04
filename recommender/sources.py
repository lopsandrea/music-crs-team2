"""Candidate sources (Level 1): each turns one parsed session into a ranked list
of candidate track_ids over the full all_tracks catalog.

The sources are deliberately *heterogeneous* — lexical (BM25), semantic-metadata
(Qwen3), collaborative (CF-BPR, ALS), acoustic (CLAP), and supervised
(text_retriever / struct_base / struct_large) — because the downstream RRF fusion
(Level 2) only gains when sources make *different* mistakes. Two families exist:
  - LIVE sources, computed here at inference time (qwen / bm25 / qwen_neighbors /
    cfbpr / als / text_retriever / clap).
  - CACHED sources, read from pre-computed 5-fold blind ensembles keyed by
    session_id (struct_base / struct_large), produced offline by the training
    pipeline (see the from-scratch reproduction section of the README).

All embedding indices store L2-NORMALISED vectors, so a dot product equals
cosine similarity and `argpartition`/`argsort` on the dot products ranks by
cosine. All inputs are on-disk artifacts and the scoring is pure numpy on fixed
matrices, so repeated runs return identical lists.

Naming note (locked legacy tokens): r21 == text_retriever, r54 == struct_base,
r84 == struct_large. The tokens are baked into shipped on-disk artifacts
(LightGBM feature_name vectors, path strings), so they are retained verbatim.
"""
from __future__ import annotations

import json

import numpy as np

from . import config as C
from .text import query_parts


def _maxrecent_topn(played, recent_k, vectors, id_to_idx, track_ids, topn):
    """Score every track by max cosine over the last `recent_k` played (shared
    by the qwen, cf-bpr and CLAP sources). Vectors are L2-normalised.

    The "max over recent anchors" is an OR over recent interests: a candidate is
    scored by its single best match among the recent tracks, not an average, so
    it stays robust when the last `recent_k` tracks span different moods. The
    three callers are the same algorithm in three different geometries
    (metadata / collaborative / audio).

    Args:
        played: ordered list of already-played track_ids (most-recent last).
        recent_k: how many of the most-recent played tracks to use as anchors.
        vectors: (n_tracks, d) L2-normalised embedding matrix for this index.
        id_to_idx / track_ids: id<->row mappings for `vectors`.
        topn: number of candidate ids to return.
    Returns: up to `topn` track_ids, ranked by descending max-cosine, with the
    anchor tracks themselves excluded.
    """
    # Map the last `recent_k` played ids to their row indices; drop ids absent
    # from this index (a track may not have an embedding in every space).
    idxs = [id_to_idx.get(str(t)) for t in played[-recent_k:]]
    idxs = [i for i in idxs if i is not None]
    if not idxs:
        return []
    anchors = vectors[idxs]                       # (k, d) anchor rows
    # vectors @ anchors.T is (n_tracks, k) of cosines; max(axis=1) keeps each
    # track's best similarity to any anchor -> (n_tracks,) score vector.
    scores = (vectors @ anchors.T).max(axis=1)
    exclude = set(idxs)
    # Pull a few extra (topn + #anchors) candidates so that after dropping the
    # anchors below we still have enough to fill `topn`. argpartition is O(n):
    # it places the `cap` best scores first (unordered), then we sort only those.
    cap = min(len(scores), topn + len(exclude))
    cand = np.argpartition(-scores, cap - 1)[:cap]
    cand = cand[np.argsort(-scores[cand])]        # exact descending order within the top `cap`
    out = []
    for i in cand:
        if int(i) in exclude:                     # never recommend an anchor (an already-played track)
            continue
        out.append(track_ids[int(i)])
        if len(out) >= topn:
            break
    return out


class _VecIndex:
    """A dense embedding index loaded from disk: an L2-normalised `vectors.npy`
    (n_tracks, d) plus the parallel `track_ids.json` row->id mapping. Backs the
    Qwen3, CF-BPR and CLAP sources; supports item-to-item nearest-neighbour."""

    def __init__(self, npy_dir):
        # vectors: (n_tracks, d) float matrix, rows aligned to track_ids.
        self.vectors = np.load(npy_dir / "vectors.npy")
        self.track_ids = json.loads((npy_dir / "track_ids.json").read_text())
        self.id_to_idx = {t: i for i, t in enumerate(self.track_ids)}

    def neighbors(self, anchor, topk):
        """Top-`topk` nearest tracks to a single `anchor` id (classic item-to-item
        i2i lookup). One matrix-vector product `vectors @ vectors[i]` gives the
        cosine to every track (vectors are L2-normalised); the anchor itself is
        excluded. Returns [] if the anchor id is not in this index."""
        i = self.id_to_idx.get(str(anchor))
        if i is None:
            return []
        scores = self.vectors @ self.vectors[i]   # (n_tracks,) cosine to the anchor
        # Fetch topk+1 so we still have topk after dropping the anchor (cosine 1.0 to itself).
        n = min(topk + 1, len(scores))
        # argpartition(-scores, n-1) is O(n_tracks): negating turns "largest cosine"
        # into "smallest value", and the pivot at position n-1 guarantees the n best
        # land in [:n] (unordered). We then argsort ONLY those n to get exact order.
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [self.track_ids[int(j)] for j in top if int(j) != i][:topk]


class Sources:
    """Loads all source indices/models once; retrieves per session."""

    def __init__(self, metadata, als, device=None):
        # All indices/models are loaded ONCE here and reused across every session;
        # per-session work happens in the src_* methods below.
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.metadata = metadata
        # ALS artifacts (from als_factors.npz): item factor matrix + id<->row maps.
        self.als_factors, self.als_ids, self.als_to_idx = als

        # BM25. Single lexical index over track metadata (name+artist+album+tags);
        # both BM25 sources query it, differing only in the query text built by
        # `src_bm25`. `load_corpus=True` keeps the stored corpus so `.retrieve`
        # can hand back per-document records (we only read each record's row "id").
        # `bm25_ids` maps a corpus row index -> track_id; `_bm25s` is stashed so the
        # tokenizer (`bm25s.tokenize`) is reachable from the retrieval methods.
        import bm25s
        self.bm25 = bm25s.BM25.load(str(C.BM25_INDEX), load_corpus=True)
        self.bm25_ids = json.loads((C.BM25_INDEX / "track_ids.json").read_text())
        self._bm25s = bm25s

        # Dense embedding indices: qwen (semantic metadata) + cf-bpr (collaborative).
        self.qwen = _VecIndex(C.QWEN_DIR)
        self.cfbpr = _VecIndex(C.CFBPR_DIR)

        # CLAP acoustic source. Loaded only when enabled (fusion weight > 0).
        self.clap = (_VecIndex(C.CLAP_DIR)
                     if C.SW_BASELINE.get("clap_recent", 0.0) > 0 and C.CLAP_DIR.exists()
                     else None)

        # text-retriever model + catalog embeddings (r21 == text_retriever; the
        # only SUPERVISED source that runs its model live at inference time).
        # The fine-tuned SentenceTransformer (bi-encoder) encodes the conversation
        # query; the catalog is pre-encoded once into track_embeddings.npy.
        from sentence_transformers import SentenceTransformer
        self.text_model = SentenceTransformer(str(C.TEXT_RETRIEVER_DIR / "model"), device=self.device)
        self.text_ids = json.loads((C.TEXT_RETRIEVER_DIR / "track_ids.json").read_text())
        self.text_embs = np.load(C.TEXT_RETRIEVER_DIR / "track_embeddings.npy")

        # struct_base (r54) blind ensemble + struct_large (r84) ensemble: these two
        # supervised retrievers run NO model live — their 5-fold blind outputs were
        # pre-computed offline and are read here as {session_id: [[track_id, score], ...]}.
        # Both JSONs store the lists under a "lists" key; struct_large tolerates the
        # legacy flat-dict layout too (large_raw itself is the {sid: pairs} map).
        self.struct_base_blind = json.loads(C.STRUCT_BASE_BLIND_LISTS.read_text()).get("lists", {})
        large_raw = json.loads(C.STRUCT_LARGE_ENSEMBLE.read_text())
        self.struct_large_blind = large_raw.get("lists", large_raw)

    # --- per-source retrieval ---
    def bm25_retrieve(self, query, topk):
        """Lexical BM25 retrieval: lower-case + tokenize the query, return the top-k
        track_ids. Shared by both bm25 sources — they differ only in the query text."""
        # Lower-case so matching is case-insensitive; bm25s.tokenize takes a LIST of
        # query strings and returns its own token-id structure (one row per query).
        toks = self._bm25s.tokenize([query.lower()])
        # return_as="tuple" gives a Results object; .documents[0] is the ranked hit
        # list for our single query. Each hit's "id" is the corpus ROW index (not a
        # track_id), so map it back through bm25_ids to recover the track_id.
        res = self.bm25.retrieve(toks, k=topk, return_as="tuple")
        return [self.bm25_ids[item["id"]] for item in res.documents[0]]

    def src_qwen_recent(self, played, topn=100):
        """qwen_recent: semantic-metadata recall — max-cosine over the last
        A_PRIME_RECENT_K (=3) played tracks in the Qwen3 metadata space."""
        return _maxrecent_topn(played, C.A_PRIME_RECENT_K, self.qwen.vectors,
                               self.qwen.id_to_idx, self.qwen.track_ids, topn)

    def src_clap_recent(self, played, topn=100):
        """clap_recent: acoustic neighbours of the played tracks (CLAP space) —
        max-cosine over the last A_PRIME_RECENT_K. Empty when the source is disabled."""
        if self.clap is None:
            return []
        return _maxrecent_topn(played, C.A_PRIME_RECENT_K, self.clap.vectors,
                               self.clap.id_to_idx, self.clap.track_ids, topn)

    def src_qwen_neighbors(self, played, topk=100):
        """qwen_neighbors: item-to-item neighbours of ONLY the single last played
        track in the Qwen3 space. Conditioning on one track makes this the noisiest
        source — it carries zero fusion weight and serves as a ranker-feature provider."""
        anchor = played[-1] if played else None
        return self.qwen.neighbors(anchor, topk) if anchor else []

    def src_cfbpr_recent(self, played, topn=100):
        """cfbpr_recent: collaborative recall in the CF-BPR behavioural item space —
        "tracks co-played by the same sessions as the recent ones"; max-cosine over
        the last CFBPR_RECENT_K (=3) played."""
        return _maxrecent_topn(played, C.CFBPR_RECENT_K, self.cfbpr.vectors,
                               self.cfbpr.id_to_idx, self.cfbpr.track_ids, topn)

    def src_bm25(self, history, user_query, topk=100):
        """The two BM25 sources, both over the same lexical index but with
        different query text built by `query_parts`:
          - q_b / bm25_lastmusic: "last_music_meta" (titled metadata of the
            single LAST played track);
          - q_c / bm25_convo: "full" (recency-boosted conversation turns +
            untitled metadata of all played tracks).
        Each falls back to the raw user_query when its built query is empty.
        Returns (b_list, c_list), each top-`topk`."""
        # Rebuild history as plain {role, content} dicts so query_parts sees a
        # uniform shape regardless of any extra keys the caller's turns carry.
        hd = [{"role": h["role"], "content": h["content"]} for h in history]
        q_b = " ".join(query_parts(hd, user_query, self.metadata, "last_music_meta"))
        q_c = " ".join(query_parts(hd, user_query, self.metadata, "full"))
        return (self.bm25_retrieve(q_b or user_query, topk),
                self.bm25_retrieve(q_c or user_query, topk))

    def src_als(self, played, topk=200):
        """als_session (ALS): session-level collaborative recall via the ALS
        matrix-factorization item factors. Builds one taste vector for the whole
        session, scores all items by dot with it, returns top-`topk` plus the
        vector itself.

        The session vector is a weighted sum of the played tracks' item factors
        (geometric weights 0.8^i over the played list consumed front-to-back,
        normalised to sum 1 — NB this weights the OLDEST track highest, a locked
        legacy quirk of the validated pipeline, kept verbatim because the shipped
        rankers were trained against exactly this behaviour). Returns ([], None)
        if no played track has an ALS factor or the session vector is degenerate.

        The returned vector `sv` is the ONLY source output reused as a ranker
        feature: at featurization `als_dot = dot(sv, item_factor)` per candidate.
        """
        idxs = [self.als_to_idx[t] for t in played if t in self.als_to_idx]
        if not idxs:
            return [], None
        # Geometric weights over the played list, normalised to sum 1.
        w = np.array([0.8 ** i for i in range(len(idxs))], dtype=np.float32)
        w /= w.sum()
        sv = np.zeros(self.als_factors.shape[1], dtype=np.float32)  # (n_factors,)
        for wi, idx in zip(w, idxs):
            sv += wi * self.als_factors[idx]
        norm = np.linalg.norm(sv)
        if norm < 1e-8:
            return [], None
        sv /= norm  # scale-invariant for the list; the magnitude matters for als_dot
        scores = self.als_factors @ sv         # (n_items,) collaborative affinity to the session
        # Never recommend an already-played track.
        for t in played:
            if t in self.als_to_idx:
                scores[self.als_to_idx[t]] = -np.inf
        top = np.argpartition(-scores, topk)[:topk]
        top = top[np.argsort(-scores[top])]
        return [self.als_ids[j] for j in top], sv

    def src_text_retriever(self, history, user_query, played_set, topk=300):
        """text_retriever (r21): supervised conversation->track recall. Build the
        query from the last TEXT_QUERY_TURNS (=3) USER turns + user_query, encode
        it live with the fine-tuned bi-encoder (L2-normalised), score the
        pre-encoded catalog by dot (= cosine), mask played tracks, return top-`topk`.
        This is the only supervised source that runs a model at inference time."""
        parts = [str(h["content"]) for h in history if h["role"] == "user"] + [user_query]
        parts = parts[-C.TEXT_QUERY_TURNS:]
        q = " ".join(parts)
        qe = self.text_model.encode([q], normalize_embeddings=True).astype(np.float32)[0]  # (d,)
        scores = self.text_embs @ qe           # (n_catalog,) cosine to the query
        for ti, tid in enumerate(self.text_ids):
            if tid in played_set:              # mask already-played tracks out of the ranking
                scores[ti] = -np.inf
        top = np.argpartition(-scores, topk)[:topk]
        top = top[np.argsort(-scores[top])]
        return [self.text_ids[j] for j in top]

    def src_struct_base(self, sid):
        """struct_base (r54): supervised structured-query recall, read from the
        pre-computed 5-fold blind ensemble keyed by session id `sid` (no model
        runs live). Returns the top-POOL_K ids plus the FULL {track_id: cosine}
        score map — that cosine becomes the ranker's `cosine` feature."""
        pairs = self.struct_base_blind.get(sid, [])
        return [t for t, _ in pairs[:C.POOL_K]], {t: float(s) for t, s in pairs}

    def src_struct_large(self, sid):
        """struct_large (r84): same cached-read pattern as struct_base but the
        BGE-large ensemble (the strongest single retriever; its feature triple
        fills feats_large's last 3 columns). NOTE the blind lists here are raw
        top-300 by cosine with played NOT excluded up-front (the played mask is
        applied later, at Level 6)."""
        pairs = self.struct_large_blind.get(sid, [])
        return [t for t, _ in pairs[:C.POOL_K]], {t: float(s) for t, s in pairs}
