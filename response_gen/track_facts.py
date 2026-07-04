"""Track-fact enrichment for response generation.

Renders the recommended track's real Track-Metadata — name/artist/album, year, duration and
~15 cleaned descriptor tags — into one compact fact string. Feeding these concrete musical
qualities to the generator makes the WHY specific instead of generic, which lifts the LLM
judge's Explanation-Quality axis. The signal is profile-independent, so it also helps the cold
(null-profile) sessions where personalization is impossible.

Use `rich_item` in place of the recommender's plain `recommend_item`, and append
`GROUNDING_DIRECTIVE` to the system prompt so the model is told to ground the WHY in these facts.
"""

from __future__ import annotations

import re


def clean_tags(tags, n: int = 15) -> list[str]:
    """Keep up to `n` human-readable descriptor tags, dropping catalog junk.

    Track-Metadata tag lists mix real descriptors (genres, moods) with noise: numeric ids,
    over-long machine strings, and "N-star" rating tags. Filter those out and de-duplicate.
    """
    out, seen = [], set()
    for t in (tags or []):
        t = str(t).strip()
        tl = t.lower()
        # Decade/era tags ("80s", "90s", "2000s", "1970s") are real descriptors — the
        # GROUNDING_DIRECTIVE explicitly asks the model to ground the WHY in the track's era —
        # so keep them despite the digit. Every OTHER digit-bearing tag (catalog ids, "best of
        # 2015", "track 7") is still rejected as junk.
        is_decade = re.fullmatch(r"(19|20)?\d0s", tl) is not None
        if not t or not (2 <= len(t) <= 22) or "star" in tl:
            continue
        if re.search(r"\d", t) and not is_decade:
            continue
        if tl in seen:
            continue
        seen.add(tl)
        out.append(t)
        if len(out) >= n:
            break
    return out


def rich_item(meta: dict, tid) -> str:
    """Build the enriched fact string for one track id, or "" if it is unknown.

    `meta` is the id->metadata catalog (the recommender's `metadata`); `tid` is the recommended
    track id. Returns "" when the id is missing so the caller can fall back to the plain item.
    """
    m = meta.get(str(tid), {}) if meta else {}
    if not m:
        return ""

    # Catalog name/date fields are multi-valued LISTS (a track can appear on several releases);
    # `first` collapses each to a single display value so the fact string reads "track_name: Foo",
    # not "track_name: ['Foo']" with stray brackets/quotes. Mirrors recommender.data.first /
    # render_recommend_item so the enriched item matches the plain item's rendering.
    def first(v):
        return v[0] if isinstance(v, list) and v else (v if not isinstance(v, list) else "")

    tags = clean_tags(m.get("tag_list"))
    yr = str(first(m.get("release_date")) or "")[:4]
    dur = m.get("duration")
    dur_s = f", duration: {int(dur) // 60000}min" if isinstance(dur, (int, float)) and dur else ""
    # Popularity (0-100 in Track-Metadata; median ~38, p95 ~68). Surface only at the EXTREMES, as a
    # factual reach note the model can use to colour the WHY ("a deep cut" vs "a staple") — not in the
    # mid-range, where it would just invite a generic "popular track" adjective the judge penalises.
    aud_s = ""
    try:
        pop = float(first(m.get("popularity")))
    except (TypeError, ValueError):
        pop = None
    if pop is not None:
        if pop >= 70:
            aud_s = ", reach: a widely known, popular track"
        elif 0 < pop <= 12:
            aud_s = ", reach: a deep cut, under the radar"
    return (f"track_name: {first(m.get('track_name'))}, artist_name: {first(m.get('artist_name'))}, "
            f"album_name: {first(m.get('album_name'))}{', year: ' + yr if yr else ''}{dur_s}{aud_s}, "
            f"descriptors/tags: {', '.join(tags)}")


# Appended to the system prompt when enrichment is on: tells the model to ground the WHY in the
# recommended track's facts (the generator appends them as a "Recommended track: ..." message right
# after the system prompt, i.e. "listed below"). Kept VERBATIM from the blind_B config that produced
# the measured numbers (explains_why-NO 9, fabrication 5%) — the only grounding wording with results.
GROUNDING_DIRECTIVE = (
    "\n\nGround the WHY in the track's SPECIFIC musical qualities listed below (genre, mood, "
    "instrumentation, era) — name the concrete quality that fits the request, not generic praise; "
    "ignore any junk tags."
)
