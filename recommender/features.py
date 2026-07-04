"""37-feature builder (base + album + struct_base cols) — ported verbatim from the
validated phase-3 blind featuriser. The large-ranker variant reuses the same 34 base
columns and substitutes the last 3 (struct_base → struct_large) — see ranker.py.

This is Level 3 of the pipeline: turn the RRF recall pool (300 candidate ids) into a
(300 x 37) float matrix — one row per candidate, one column per feature — that the
LightGBM/CatBoost rankers consume. Columns fall into blocks:
  - 0..7   context/content: rrf rank, artist/tag match to last played, query-token
           overlaps, is_played, recency_score.
  - 8..20, 26  source-membership for the 6 live sources (rank_inv 8..13, presence
           14..19), n_sources (20), and source_count_v2 (26, a duplicate of 20 kept
           from the validated pipeline).
  - 21..25 collaborative/global: als_dot (the only use of als_vec), n_history,
           popularity, pool artist fraction/count.
  - 27..28 text_retriever (r21) rank_inv / presence.
  - 29..33 album-overlap features (offsets n_base+0..4).
  - 34..36 the retriever triple (offsets N_R39+0..2): rank_inv / presence / cosine of
           struct_base (in feats_base) OR struct_large (in feats_large) — the ONLY 3
           columns that differ between the two ranker variants.

Locked legacy tokens in the feature-name lists below (r21 == text_retriever,
r54 == struct_base) are kept on purpose: the names are baked into the shipped
LightGBM models' feature_name vector, so renaming them would require a full
GPU retrain. They are NOT bugs.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .text import tokens

FEAT_BASE = [
    "rrf_rank_inv", "last_artist_match", "last_tag_jaccard",
    "query_artist_tok_overlap", "query_title_tok_overlap", "query_meta_tok_overlap",
    "is_played", "recency_score",
    "src_a_rank_inv", "src_b_rank_inv", "src_c_rank_inv", "src_d_rank_inv",
    "src_f_rank_inv", "src_als_rank_inv",
    "src_a_pres", "src_b_pres", "src_c_pres", "src_d_pres", "src_f_pres", "src_als_pres",
    "n_sources", "als_dot", "n_history",
    "popularity", "pool_artist_frac", "pool_artist_count", "source_count_v2",
    # locked: these 2 (text-retriever) names are baked into the shipped LightGBM models
    # feature_name vector (lgbm_base.txt/lgbm_large.txt) — rename needs a retrain.
    "r21_rank_inv", "r21_presence",
]
FEAT_ALBUM = [
    "same_album_last1", "same_album_last3", "same_album_any",
    "album_history_count", "pool_same_album_count",
]
# FEAT_R39_ALL: the 34 columns common to BOTH ranker variants (29 base + 5 album).
# "r39" is the legacy experiment code for this 34-column block; the suffix is purely
# a name (the block is 34 wide, not 39) and is retained to match the shipped artifacts.
FEAT_R39_ALL = FEAT_BASE + FEAT_ALBUM            # 34
# FEAT_LAST3: human-readable labels for the variant-specific triple (cols 34..36).
# This list is documentation only — it is NOT referenced by featurize(); the triple's
# real, locked feature_name entries are the "r54_*" strings appended into FEAT_ALL below.
FEAT_LAST3 = ["rank_inv", "presence", "cosine"]  # struct_base or struct_large triple
# locked: these 3 names are baked into the shipped LightGBM models (rename needs a retrain)
# r54 == struct_base; the SAME 3 column names label the struct_large triple in feats_large
# (only the column VALUES differ between variants — the names stay r54_* in both models).
FEAT_ALL = FEAT_R39_ALL + ["r54_rank_inv", "r54_presence", "r54_cosine"]  # 37
N_R39 = len(FEAT_R39_ALL)                         # 34: width of the shared block / start col of the triple
# POOL_K: nominal pool size (RRF emits 300 candidates). Used as the default `pool_K` cap
# for the pool-artist distribution; the actual pool may be shorter, so code guards with max(len(pool),1).
POOL_K = 300


def active_feat_all() -> list[str]:
    """The feature-name list the rankers were trained on (width 37)."""
    return FEAT_ALL


def featurize(pool, src_lists, text_rank_map, last_rank_map, last_score_map,
              case_query, history, played, played_set, ta, tt, ttl, tat, tmt,
              als_factors, als_to_idx, als_vec, track_pop, max_pop, track_album,
              pool_K=POOL_K):
    """Return the (len(pool), 37) feature matrix.

    last_* = struct_base (or struct_large) rank/score maps. One row per pool candidate,
    one column per feature in active_feat_all(). The caller builds this twice: once with the
    struct_base maps (feats_base) and once overwriting the last-3 columns of the 37-col block
    with struct_large (feats_large); only those 3 columns differ. The five `t*` args are
    per-track lookup maps:
      ta=track_artist, tt=track_tags(set), ttl=title tokens, tat=artist tokens,
      tmt=meta tokens. `als_vec` is the session vector from src_als (None if the
      session had no ALS-known tracks), reused here as the `als_dot` feature.
    """
    n_base = len(FEAT_BASE)                       # 29: index where the album block starts
    n_r39 = N_R39                                 # 34: index where the retriever triple starts
    # (n_candidates, 37) — float64 so the GBDT rankers see exact feature values.
    feats = np.zeros((len(pool), len(active_feat_all())), dtype=np.float64)

    # --- per-session context computed ONCE, reused for every candidate row below ---
    # All user messages (history user turns + the current query); used for token-overlap features.
    user_msgs = [str(r["content"]) for r in history if r["role"] == "user"] + [case_query]
    n_hist = len(played)
    now_tok = tokens(user_msgs[-1]) if user_msgs else set()   # tokens of the CURRENT user turn
    all_tok = tokens(" ".join(user_msgs))                     # tokens across ALL user turns
    l_artist = ta.get(played[-1], "") if played else ""       # artist of the LAST played track
    l_tags = tt.get(played[-1], set()) if played else set()   # tags of the LAST played track
    # `prior`: per history entry a (weight, artist, tags) tuple for recency_score.
    # reversed(played) -> most-recent first, so j=0 is the last track and weight 1/(j+1) decays.
    prior = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
             for j, t in enumerate(reversed(played))]
    # Artist distribution within the pool (for the pool_artist_frac / _count features).
    # NOTE: counts are taken over only the first pool_K candidates (head of the pool),
    # but pool_artist_frac below normalises by the FULL len(pool); kept as-is to match the
    # validated featuriser (the two rarely differ since the pool is ~pool_K long anyway).
    pool_artists = [ta.get(tid, "") for tid in pool[:pool_K]]
    artist_counts = Counter(a for a in pool_artists if a)
    # Per-source 1-based rank maps: {source_name: {track_id: rank}} for rank_inv/presence features.
    src_rank = {sn: {tid: r + 1 for r, tid in enumerate(sl)} for sn, sl in src_lists.items()}

    # Album context of the played history (for the album-overlap block, cols 29..33).
    last1_album = track_album.get(played[-1], "") if played else ""
    last3_albums = {track_album.get(t, "") for t in played[-3:]} - {""}
    all_albums = [track_album.get(t, "") for t in played]
    album_hist_counts = Counter(a for a in all_albums if a)

    # `rank` is 1-based position in the fused pool; row = the candidate's feature vector.
    for rank, tid in enumerate(pool, start=1):
        ca = ta.get(tid, "")                      # candidate artist
        ct = tt.get(tid, set())                   # candidate tags (set)
        row = feats[rank - 1]
        # --- context/content block (cols 0..7) ---
        row[0] = 1.0 / rank                       # rrf_rank_inv: higher for better-ranked pool items
        row[1] = 1.0 if ca and ca == l_artist else 0.0   # last_artist_match
        if ct or l_tags:
            row[2] = len(ct & l_tags) / len(ct | l_tags)  # last_tag_jaccard (|∩| / |∪|)
        # query/title/meta token-overlap counts vs the current turn / all user turns.
        row[3] = float(len(tat.get(tid, set()) & now_tok))   # query_artist_tok_overlap
        row[4] = float(len(ttl.get(tid, set()) & now_tok))   # query_title_tok_overlap
        row[5] = float(len(tmt.get(tid, set()) & all_tok))   # query_meta_tok_overlap
        row[6] = 1.0 if tid in played_set else 0.0           # is_played
        # recency_score: artist-match + tag-jaccard against each prior track, recency-weighted 1/(j+1).
        rec = 0.0
        for wd, pa, pt in prior:
            am = 1.0 if ca and ca == pa else 0.0
            tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
            rec += wd * (am + tm)
        row[7] = rec
        # --- source-membership block: 6 live sources, in this fixed order ---
        # rank_inv (cols 8..13): 1/rank in each source (0 if absent).
        for fi, sname in enumerate(["qwen_recent", "bm25_lastmusic", "bm25_convo", "qwen_neighbors", "cfbpr_recent", "als_session"]):
            sr = src_rank[sname].get(tid)
            row[8 + fi] = 1.0 / sr if sr else 0.0
        # presence (cols 14..19): 1 if the candidate appears in that source.
        for fi, sname in enumerate(["qwen_recent", "bm25_lastmusic", "bm25_convo", "qwen_neighbors", "cfbpr_recent", "als_session"]):
            row[14 + fi] = 1.0 if tid in src_rank[sname] else 0.0
        # n_sources (col 20): how many of the 6 live sources contain the candidate (recall agreement).
        row[20] = sum(1 for sn in ["qwen_recent", "bm25_lastmusic", "bm25_convo", "qwen_neighbors", "cfbpr_recent", "als_session"] if tid in src_rank.get(sn, {}))
        # --- collaborative/global block (cols 21..25) ---
        # als_dot (col 21): cosine-scale affinity of the candidate to the session vector;
        # the ONLY place als_vec enters the features. Left at 0 if no session vector or no factor.
        if als_vec is not None:
            aidx = als_to_idx.get(tid)
            if aidx is not None:
                f = als_factors[aidx]
                row[21] = float(np.dot(als_vec, f))
        row[22] = float(n_hist)                   # n_history: number of played tracks
        row[23] = track_pop.get(tid, 0) / max_pop # popularity: normalised play count in [0, 1]
        row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0  # pool_artist_frac
        row[25] = float(artist_counts.get(ca, 0)) if ca else 0               # pool_artist_count
        row[26] = row[20]                         # source_count_v2: duplicate of n_sources, kept from the validated pipeline
        # --- text_retriever (r21) block (cols 27..28) ---
        row[27] = 1.0 / text_rank_map[tid] if tid in text_rank_map else 0.0  # r21_rank_inv
        row[28] = 1.0 if tid in text_rank_map else 0.0                        # r21_presence

        # --- album-overlap block (cols n_base+0 .. n_base+4 == 29..33) ---
        c_album = track_album.get(tid, "")
        row[n_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0          # same_album_last1
        row[n_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0          # same_album_last3
        row[n_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0     # same_album_any
        row[n_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0  # album_history_count (normalised)
        # O(len(pool)) per row: scans the WHOLE pool counting candidates on the same album.
        # Quadratic over the pool, but len(pool) is bounded (~300) so it stays cheap.
        pool_alb = sum(1 for t2 in pool if track_album.get(t2, "") == c_album) if c_album else 0
        row[n_base + 4] = pool_alb / max(len(pool), 1)                                 # pool_same_album_count (fraction of pool on the same album)

        # --- retriever triple (cols n_r39+0 .. n_r39+2 == 34..36): struct_base here;
        # the caller overwrites these three for the struct_large variant (feats_large). ---
        row[n_r39 + 0] = 1.0 / last_rank_map[tid] if tid in last_rank_map else 0.0     # rank_inv
        row[n_r39 + 1] = 1.0 if tid in last_rank_map else 0.0                           # presence
        row[n_r39 + 2] = last_score_map.get(tid, 0.0)                                   # cosine (raw retriever score)

    return feats
