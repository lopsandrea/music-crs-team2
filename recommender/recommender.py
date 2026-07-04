"""Recommender facade: conversation session -> top-20 track ids.

Wires the candidate sources -> weighted-RRF pool -> 37-feature matrices
(struct_base + struct_large variants) -> base/large LambdaRank with selective
routing -> CatBoost-ensemble blend -> top-20. Loads all models/indices once;
`batch_recommend` runs a list of sessions.

This module is the orchestrator for the recommendation half of the composite
(the nDCG@20 score). It runs the pipeline per session: L1 sources (sources.py)
-> L2 RRF fusion (fusion.py) -> L3 featurization (features.py) -> L4 LightGBM +
routing, L5 CatBoost-ensemble blend, L6 top-20 selection (ranker.py). Level 7
(the Gemini response) is handled separately in response_gen / pipeline.py and
does not affect nDCG.
"""
from __future__ import annotations

from . import config as C
from .data import (load_als, load_supporting_maps, load_track_metadata,
                   render_recommend_item)
from .features import N_R39, featurize
from .fusion import weighted_rrf
from .ranker import load_rankers, score_route_top20
from .sources import Sources


class Recommender:
    """Top-level inference facade. Construction loads every model/index ONCE
    (metadata maps, ALS factors, the candidate Sources, both LightGBM rankers and
    the CatBoost ensemble); then `recommend_session` / `batch_recommend` run the
    full Level 1->6 pipeline per session. Stateless per call after construction."""

    def __init__(self, device=None):
        self.metadata = load_track_metadata()
        # Featurization lookup tables: artist/tags/token sets per track + popularity + album.
        self.maps, self.track_pop, self.track_album = load_supporting_maps()
        # Denominator for the normalised `popularity` feature (guard empty map with 1).
        self.max_pop = max(self.track_pop.values()) if self.track_pop else 1
        self.als = load_als()                      # (factors, ids, to_idx)
        self.sources = Sources(self.metadata, self.als, device=device)
        # Two LambdaRank boosters (base/large); they differ only in the last-3
        # feature columns (struct_base vs struct_large triple) — see ranker.py routing.
        self.lgbm_base, self.lgbm_large = load_rankers()
        from catboost import CatBoostRanker         # CatBoost YetiRank ensemble (Level 5)
        self.cb_model = CatBoostRanker()
        self.cb_model.load_model(str(C.CB_MODEL))
        # Unpack the five per-track maps into the fixed positional order featurize() expects.
        m = self.maps
        self._maps = (m["track_artist"], m["track_tags"], m["track_title_toks"],
                      m["track_artist_toks"], m["track_meta_toks"])

    def recommend_session(self, s: dict) -> dict:
        """Run Levels 1->6 for one parsed session `s` and return its prediction.

        The session dict (from recommender.data) carries session_id, turn_number,
        user_query, history, and music_turns (the ordered already-played track_ids).
        Returns a dict with the top-20 `predicted_track_ids` (the nDCG@20 half),
        the `recommend_item` metadata string of the #1 track (fed to response_gen),
        and routing diagnostics `_used_large` / `_margin`.

        `_used_large` is True when selective routing chose lgbm_large for this session;
        `_margin` is the base ranker's top1-minus-top2 raw-score gap that drove the choice
        (lgbm_large is used when margin < ROUTE_LOW=0.25 or >= ROUTE_HIGH=1.5, else lgbm_base).
        Both are diagnostics only — they do not affect the returned track ids beyond the
        routing they describe, and are not part of the Codabench payload.
        """
        sid, turn = s["session_id"], s["turn_number"]
        uq, hist, played = s["user_query"], s["history"], s["music_turns"]
        played_set = set(played)
        S = self.sources

        # --- Level 1: run the candidate sources. Letters a/b/c/d/f are the legacy
        # source codes (a=qwen_recent, b=bm25_lastmusic, c=bm25_convo, d=qwen_neighbors,
        # f=cfbpr_recent); src_bm25 returns the B and C lists together.
        a = S.src_qwen_recent(played, topn=100); b, c = S.src_bm25(hist, uq, topk=100)
        d = S.src_qwen_neighbors(played, topk=100); f = S.src_cfbpr_recent(played, topn=100)
        clap = S.src_clap_recent(played, topn=100)   # acoustic "sounds-like" source
        # als_session returns the ranked ids AND the session vector `als_vec`, which is
        # reused at featurization as the als_dot feature.
        als_tracks, als_vec = S.src_als(played)
        text_list = S.src_text_retriever(hist, uq, played_set, topk=300)
        # struct_base / struct_large: cached 5-fold blind lists + their score maps
        # (the raw retriever cosine becomes each variant's `cosine` feature).
        base_list, base_score = S.src_struct_base(sid)
        large_list, large_score = S.src_struct_large(sid)

        src_lists = {"qwen_recent": a, "bm25_lastmusic": b, "bm25_convo": c,
                     "qwen_neighbors": d, "cfbpr_recent": f,
                     "als_session": als_tracks, "text_retriever": text_list,
                     "struct_base": base_list, "struct_large": large_list,
                     "clap_recent": clap}
        # --- Level 2: weighted-RRF fusion of the source lists -> recall pool of POOL_K (300) ids.
        pool = weighted_rrf(src_lists, C.SW_BASELINE, topk=C.POOL_K, k=C.RRF_K)

        # 1-based rank maps for the two supervised sources whose rank/presence become
        # ranker features (text_retriever -> r21 cols; struct_base -> the last-3 triple).
        text_rank_map = {tid: r + 1 for r, tid in enumerate(text_list[:C.POOL_K])}
        base_rank_map = {tid: r + 1 for r, tid in enumerate(base_list[:C.POOL_K])}
        ta, tt, ttl, tat, tmt = self._maps
        # --- Level 3: featurize the pool. feats_base's last 3 columns carry the struct_base triple.
        feats_base = featurize(pool, src_lists, text_rank_map, base_rank_map, base_score,
                               uq, hist, played, played_set, ta, tt, ttl, tat, tmt,
                               self.als[0], self.als[2], als_vec,
                               self.track_pop, self.max_pop, self.track_album)
        # feats_large is an identical copy EXCEPT the last 3 columns (indices N_R39+0..2),
        # which are overwritten with the struct_large rank_inv / presence / cosine triple.
        # N_R39 == 34, so these are columns 34/35/36 of the 37-column matrix.
        feats_large = feats_base.copy()
        large_rank_map = {tid: r + 1 for r, tid in enumerate(large_list[:C.POOL_K])}
        for k, tid in enumerate(pool):
            feats_large[k, N_R39 + 0] = (1.0 / large_rank_map[tid]) if tid in large_rank_map else 0.0
            feats_large[k, N_R39 + 1] = 1.0 if tid in large_rank_map else 0.0
            feats_large[k, N_R39 + 2] = large_score.get(tid, 0.0)

        # --- Level 5 input: the CatBoost ensemble scores the struct_large variant (it was
        # trained on the feats_large layout, so it must be fed feats_large, NOT feats_base);
        # cb_scores is a length-len(pool) array of raw scores that score_route_top20
        # z-normalises before blending with the routed LightGBM scores.
        cb_scores = self.cb_model.predict(feats_large)
        # --- Levels 4-6: LightGBM scoring with selective routing, z-score blend with the
        # CatBoost scores (weight CB_W=1.0), then argsort desc / skip played -> top-20.
        top20, used_large, margin = score_route_top20(
            self.lgbm_base, self.lgbm_large, feats_base, feats_large, pool, played_set,
            ce_scores=cb_scores, ce_w=C.CB_W)

        return {"session_id": sid, "turn_number": turn,
                "predicted_track_ids": top20,
                # recommend_item: human-readable metadata of the #1 track, consumed by response_gen (Level 7).
                "recommend_item": render_recommend_item(top20[0], self.metadata) if top20 else "",
                "_used_large": used_large, "_margin": margin}

    def batch_recommend(self, sessions: list[dict]) -> list[dict]:
        """Recommend for a list of sessions sequentially, printing a progress line
        every 20 sessions. Returns one prediction dict per input session, in order."""
        out = []
        for i, s in enumerate(sessions):
            out.append(self.recommend_session(s))
            if (i + 1) % 20 == 0:
                print(f"  recommended {i + 1}/{len(sessions)}", flush=True)
        return out
