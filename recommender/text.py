"""Text helpers: tokenization and BM25 query construction (kept verbatim from the
validated pipeline so query behavior matches the shipped rankers exactly)."""
from __future__ import annotations

import re

# Domain stopwords dropped before BM25 matching. Beyond generic English function words, this
# deliberately removes music-request boilerplate ("song", "play", "more", "another", ...) that
# carries no discriminative signal in this dataset and would otherwise match almost every track.
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "you", "your",
    "song", "songs", "track", "tracks", "music", "artist", "artists",
    "band", "bands", "more", "another", "please", "some", "have",
    "like", "want", "really", "something", "can", "could", "would",
    "from", "about", "into", "give", "play", "just", "any",
}


def tokens(text: str) -> set[str]:
    """Lowercase, regex-tokenise, and filter `text` into a BM25-ready set of terms.

    Keeps alphanumeric/apostrophe runs (``[a-z0-9']+``) longer than 2 chars that are not
    stopwords. Returns a SET (de-duplicated, order-independent) — callers use it for term
    overlap, not for term-frequency weighting.
    """
    return {
        t for t in re.findall(r"[a-z0-9']+", str(text).lower())
        if len(t) > 2 and t not in STOPWORDS
    }


def meta_text(track_id: str, metadata: dict[str, dict], include_track_name: bool = True) -> str:
    """Flatten a track's catalog metadata into one searchable text blob.

    Concatenates artist + album + up to the first 8 tags, optionally prefixed by the track name.
    `include_track_name` is turned OFF when the played track is itself the query anchor (so a
    sequel/neighbour search isn't dominated by the exact title); see query_parts' 'last_music_meta'
    mode. Missing fields collapse to "" and are dropped from the join, so the result never carries
    stray separators. Unknown track_id -> "".
    """
    meta = metadata.get(str(track_id), {})
    parts = []
    if include_track_name:
        parts.append(str(meta.get("track_name", "")))
    parts.append(str(meta.get("artist_name", "")))
    parts.append(str(meta.get("album_name", "")))
    tags = meta.get("tag_list", [])
    # tag_list is normally a list (cap at 8 tags to bound query length); tolerate a bare scalar.
    if isinstance(tags, list):
        parts.extend(str(t) for t in tags[:8])
    elif tags:
        parts.append(str(tags))
    return " ".join(p for p in parts if p)


def query_parts(history: list[dict], user_query: str, metadata: dict[str, dict], mode: str) -> list[str]:
    """Assemble the text "parts" that form a BM25 query, weighting signals via repetition.

    Returns a list of strings; the caller tokenises and concatenates them, so REPEATING a part
    is how a signal is up-weighted (the current user_query is emitted x3 to dominate older
    context). The two shipped modes:

      'full' (bm25_convo)            -> every user turn once, with the MOST RECENT user turn
                                        emitted twice (recency boost), then the latest request
                                        x3; plus UNTITLED metadata text of ALL played tracks.
      'last_music_meta' (bm25_lastmusic) -> metadata of only the LAST played track, and
                                        (uniquely) it includes that track's name
                                        (include_track_name=True) to anchor on the exact title.

    Note: 'music' history turns carry a TRACK_ID in their content (see data.parse_last_turn),
    hence each is passed to meta_text(tid, ...) to expand into artist/album/tags. Empty parts
    are filtered out at the end so blank turns never pollute the query.
    """
    parts = []
    user_turns = [h["content"] for h in history if h["role"] == "user"]
    music_turns = [h["content"].strip() for h in history if h["role"] == "music"]

    if mode == "full":
        # All user turns, with an extra copy of the most recent one (recency boost),
        # then the latest request tripled to make it the dominant BM25 signal.
        for i, turn in enumerate(user_turns):
            parts.append(turn)
            if i == len(user_turns) - 1:
                parts.append(turn)
        parts.extend([user_query] * 3)

    if mode in ("full", "last_music_meta"):
        # Which played tracks expand into metadata text, and whether their titles are kept:
        #   last_music_meta -> last 1, titled (anchor-on-exact-title behavior)
        #   full            -> all, untitled
        selected = music_turns[-1:] if mode == "last_music_meta" else music_turns
        titled = mode == "last_music_meta"
        for tid in selected:
            parts.append(meta_text(tid, metadata, include_track_name=titled))

    return [p for p in parts if p]
