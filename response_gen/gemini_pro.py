"""gemini-3.1-pro-preview response generator with MusicMind persona + stochastic style packet.

Replica of the collega's v19 architecture:
  - Roleplay: MusicMind (record-store-veteran persona)
  - Response framework: banned phrases extensive (no sycophantic, no filler)
  - Stochastic style packet: 12 opening × 7 closing × 6 body = 504 combos
  - Contextual: vocab register by age, temporal anchor by decade, tonal energy
  - 120-word minimum (no short responses)
  - Refiner Layer 2: optional (off by default like v19)
  - Thinking mode active: max_output_tokens=8192 to leave room for reasoning

Drop-in for GEMINI_MODEL — same interface (response_generation, batch_response_generation).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
import re
from collections import Counter
from pathlib import Path

from .gemini import GEMINI_MODEL, _resolve_api_key, _hash_key


# ===== Stochastic style packet dimensions =====================================
# Three independent banks of one-line writing directives. One line is sampled
# from each per response; the product (12 openings × 6 body × 7 closings = 504)
# is the combinatorial diversity that keeps the generated responses from
# collapsing onto a single template across thousands of sessions. Counts must
# stay in sync with the v19 architecture the module docstring references.

_OPENING_ANGLES = [
    "Open by painting the mood or atmosphere this track evokes from the very first second.",
    "Open by describing what you actually hear: the instrumentation, the production texture, the way the elements interlock.",
    "Open by naming the track and artist immediately, then hook with the single most distinctive quality.",
    "Open by placing the track in its scene or era — when it was made, what was happening musically around it.",
    "Open by addressing the listener directly — what specifically about their request this answers.",
    "Open with a sonic image: the way the song unfolds in time, what catches the ear first.",
    "Open with the lyrical or thematic angle: what the song is about, what it makes you feel without naming the feeling.",
    "Open with a comparison or lineage: which artist or scene this track descends from, or what it predicts.",
    "Open by setting a scene where this track plays — a specific time of day, a specific kind of room.",
    "Open with a hook from the track itself — a famous line, a riff, a moment listeners single out.",
    "Open by naming the genre and what this artist does with it that no one else quite does.",
    "Open with what makes this track stand out from the listener's recent reference points.",
]

_CLOSING_FORMATS = [
    "Close with a vivid, specific scene tied to the track — a moment, a setting, a mental image the song unlocks.",
    "Close with a concrete follow-up recommendation: name a specific album, artist, or track that pairs naturally with this one.",
    "Close with a challenge: invite the listener to explore a related genre, scene, or era they haven't touched.",
    "Close with a question rooted in the listener's personal world — their city, a memory, an experience that maps to this song.",
    "Close with a cultural reference that opens a door — a film, a book, a piece of art that shares this track's spirit.",
    "Close by zooming in on one specific moment in the track — a transition, a lyric, a sonic detail worth pointing at.",
    "Close with what to listen for next time — a layer or detail that rewards a second listen.",
]

_BODY_STRUCTURES = [
    "In the body, lead with the specific production choices and how they shape the listening experience.",
    "In the body, trace the emotional arc of the track — how it builds, where it pivots, what it leaves with you.",
    "In the body, zoom in on one specific moment in the track — a riff, a rhythmic detail, a vocal turn — and unpack why it works.",
    "In the body, weave the artist's lineage and what this track inherits from them.",
    "In the body, contrast this track with adjacent tracks the listener might know, sharpening what makes it distinctive.",
    "In the body, foreground the lyrical content and how the vocal delivery shapes its meaning.",
]

# Vocabulary register keyed by the listener's age_group profile field. Picks a
# tone the model should adopt; keys must match the age_group string values, with
# "25-34" used as the neutral default when age is unknown (see _age_to_register).
_VOCAB_BY_AGE = {
    "13-17": "Use informal, high-energy language — the kind you would use telling a friend about a track that just blew your mind.",
    "18-24": "Use informal, culturally aware language — enthusiastic and specific, like a peer who knows their music deeply.",
    "25-34": "Use the language of a knowledgeable music enthusiast — direct, engaged, peer-to-peer.",
    "35-49": "Write with warm authority — someone who has lived with music long enough to place it in context.",
    "50+":   "Write with the depth of someone who has followed music across decades — reference legacy, cultural context.",
}

# Tonal-energy directives selected on a 2x2 grid of (specificity, category),
# where each axis is High or Low. HH/HL/LH/LL = the four corners; chosen by
# _pick_tonal from the listener's profile signals.
_TONAL_HH = "Be specific and discerning — be precise, use technical vocabulary where it earns its place."
_TONAL_HL = "Be engaged but open — balance warmth with musical knowledge, avoid overload."
_TONAL_LH = "Be in discovery mode but adventurous — suggest new directions with confidence."
_TONAL_LL = "Be broadly open — keep your language accessible and welcoming, avoid jargon."


def _decade(release_date: str | None) -> int | None:
    """Extract the release decade (e.g. 1990) from a release-date string.

    Pulls the first 4-digit run as the year. Returns None for missing input, no
    year found, or years before 1960 — pre-1960 is treated as "no useful decade
    anchor" so the temporal directive is simply omitted rather than dating the
    track to an era the persona shouldn't lean on.
    """
    if not release_date:
        return None
    m = re.search(r"(\d{4})", str(release_date))
    if not m:
        return None
    year = int(m.group(1))
    if year < 1960:
        return None
    return (year // 10) * 10


def _age_to_register(age_group: str | None) -> str:
    """Map an age_group profile value to its vocabulary-register directive.

    Unknown or missing age falls back to the neutral "25-34" register.
    """
    if not age_group:
        return _VOCAB_BY_AGE["25-34"]
    s = str(age_group).strip()
    return _VOCAB_BY_AGE.get(s, _VOCAB_BY_AGE["25-34"])


def _pick_tonal(specificity: str = "M", category: str = "M") -> str:
    """Select one of the four tonal directives from two High/Low profile signals.

    Each axis is "High" iff its string starts with "H" or "1" (so values like
    "High", "h", "1" all count as high); anything else — including the "M"
    medium default — counts as Low. Returns the corner of the 2x2 grid:
    (specificity, category) -> HH / HL / LH / LL.
    """
    high_s = str(specificity).upper().startswith(("H", "1"))
    high_c = str(category).upper().startswith(("H", "1"))
    if high_s and high_c:
        return _TONAL_HH
    if high_s and not high_c:
        return _TONAL_HL
    if not high_s and high_c:
        return _TONAL_LH
    return _TONAL_LL


def build_style_packet(
    user_profile: dict | None = None,
    track_metadata: dict | None = None,
    seed: int | None = None,
) -> str:
    """Compose a 6-line stochastic + contextual style packet for one response.

    Lines (order matters — opening, body, closing, then context):
      1. opening angle   — sampled from _OPENING_ANGLES
      2. body structure  — sampled from _BODY_STRUCTURES
      3. closing format  — sampled from _CLOSING_FORMATS
      4. register        — deterministic from user_profile["age_group"]
      5. temporal anchor — deterministic from track release decade (may be "")
      6. tonal energy    — deterministic from profile specificity/category

    The first three lines are the *stochastic* dimension; passing ``seed`` makes
    the sampling reproducible (the caller derives a stable per-(session,turn)
    seed from the recommend_item hash). With ``seed is None`` the module-global
    ``random`` is used, i.e. nondeterministic. Empty lines (e.g. no decade) are
    dropped by the caller's join.
    """
    rng = random.Random(seed) if seed is not None else random
    opening = rng.choice(_OPENING_ANGLES)
    closing = rng.choice(_CLOSING_FORMATS)
    body = rng.choice(_BODY_STRUCTURES)
    user_profile = user_profile or {}
    register = _age_to_register(user_profile.get("age_group"))
    track_metadata = track_metadata or {}
    decade = _decade(track_metadata.get("release_date"))
    temporal = f"Where relevant, ground your writing in the {decade}s — the era this track comes from." if decade else ""
    tonal = _pick_tonal(
        user_profile.get("specificity", "M"),
        user_profile.get("category", "M"),
    )
    lines = [opening, body, closing, register, temporal, tonal]
    return "\n".join(l for l in lines if l)


class GEMINI_PRO_MODEL(GEMINI_MODEL):
    """Extension of GEMINI_MODEL for gemini-3.1-pro-preview + MusicMind + style packet.

    Public API matches GEMINI_MODEL exactly so it's a drop-in via lm_type field.
    Style packet & MusicMind prompts are applied INSIDE batch_response_generation
    by prepending to the system prompt argument received from CRS_BASELINE.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.1-pro-preview",
        api_key: str | None = None,
        max_concurrent: int = 8,
        cache_dir: str = "./cache/gemini",
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        thinking_budget: int | None = None,
        use_style_packet: bool = True,
        use_refiner: bool = False,
        min_words: int | None = None,
        seen_bigrams: Counter | None = None,
        device: str = "cuda",
        attn_implementation: str = "eager",
        dtype=None,
    ):
        # Resolve Pro-specific generation defaults BEFORE calling super().__init__,
        # because they diverge from the Flash parent: a larger output budget (room
        # for the mandatory reasoning trace), a hotter temperature (1.0, for style
        # diversity), and a non-zero thinking budget (Pro requires thinking mode
        # on). Each falls back to a GEMINI_PRO_* env var so a variant can retune
        # without code edits; an explicit constructor arg always wins over the env.
        # The device / attn_implementation / dtype args are inherited LLAMA_MODEL
        # compat parameters and are unused by the Gemini HTTP client.
        # gemini-3.1-pro-preview requires more tokens for thinking trace.
        if max_new_tokens is None:
            max_new_tokens = int(os.environ.get("GEMINI_PRO_MAX_TOKENS", 8192))
        if temperature is None:
            temperature = float(os.environ.get("GEMINI_PRO_TEMP", 1.0))
        if thinking_budget is None:
            # Gemini 2.5 Pro REQUIRES thinking_budget > 0 (mandatory thinking mode).
            thinking_budget = int(os.environ.get("GEMINI_PRO_THINKING_BUDGET", 2048))
        if min_words is None:
            # GEMINI_PRO_MIN_WORDS env lets variants control verbosity (lex tuning).
            # Set to 0 to skip the "minimum X words" directive entirely.
            min_words = int(os.environ.get("GEMINI_PRO_MIN_WORDS", 120))
        super().__init__(
            model_name=model_name, api_key=api_key, max_concurrent=max_concurrent,
            cache_dir=cache_dir, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, thinking_budget=thinking_budget,
            device=device, attn_implementation=attn_implementation, dtype=dtype,
        )
        self.use_style_packet = bool(use_style_packet)
        self.use_refiner = bool(use_refiner)
        self.min_words = int(min_words)
        self._seen_bigrams = seen_bigrams

    # Style packet derives a stable per-(session,turn) seed from the recommend_item
    # hash for reproducibility within a single submission.
    def _style_packet_seed(self, recommend_item: str) -> int:
        """Derive a deterministic 32-bit seed from the recommend_item string.

        Uses the first 4 bytes of its MD5 digest as a big-endian uint32. Because
        the top-1 track metadata differs per (session, turn), this gives each
        response its own stable style-packet sampling that nonetheless replays
        identically on re-runs of the same submission. MD5 here is a fast
        non-cryptographic hash, not a security primitive.
        """
        h = hashlib.md5(recommend_item.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "big")

    def _augment_system_prompt(
        self,
        sys_prompt: str,
        recommend_item: str,
        user_profile: dict | None = None,
        track_metadata: dict | None = None,
    ) -> str:
        """Append per-response directives to the base system prompt and return it.

        Builds, in order, an optional style packet, an optional minimum-word
        directive, an instruction to start with the recommendation text (and not
        echo the metadata), the track metadata block itself, and — when a
        ``seen_bigrams`` counter was supplied — a list of session-wide overused
        bigrams to avoid (anti-repetition across the batch). All addenda are
        joined to the original ``sys_prompt`` with blank-line separators; if no
        addenda apply the prompt is returned unchanged.
        """
        addenda = []
        if self.use_style_packet:
            packet = build_style_packet(
                user_profile=user_profile,
                track_metadata=track_metadata,
                seed=self._style_packet_seed(recommend_item),
            )
            addenda.append(
                "Structural directives for this response — apply these within the WHY + personalization framework above, not instead of it:"
            )
            addenda.append(packet)
        if self.min_words > 0:
            addenda.append(
                f"Your response must be at least {self.min_words} words. Never write a short answer — every "
                "recommendation needs enough space to cover the WHY, the personal connection, and a meaningful closing."
            )
        addenda.append(
            "Begin your response directly with the recommendation text. Do not echo, repeat, or reference the track metadata string below."
        )
        addenda.append(f"Track to recommend for this listener:\n{recommend_item}")
        if self._seen_bigrams:
            # Surface up to the 20 most frequent bigrams that appear >=3 times
            # across the whole session set and ask the model to avoid them. This
            # is the global anti-repetition guard: it stops stock phrases from
            # recurring across many responses in the same submission.
            top_overused = [
                " ".join(bg) for bg, c in self._seen_bigrams.most_common(20) if c >= 3
            ]
            if top_overused:
                addenda.append(
                    "IMPORTANT — these bigrams are overused across the session set, "
                    f"avoid them: {', '.join(top_overused)}."
                )
        if addenda:
            return sys_prompt + "\n\n" + "\n\n".join(addenda)
        return sys_prompt

    # Override format_messages: with the augmented system prompt the recommend_item
    # is already embedded → we don't need to add it as a model turn.
    def _format_messages(self, sys_prompt: str, chat_history: list[dict], recommend_item: str):
        """Build the Gemini ``contents`` turn list for one augmented session.

        Differs from the parent GEMINI_MODEL._format_messages in one way: it does
        NOT inject a synthetic ``model`` turn echoing ``recommend_item``. Here the
        recommendation (and its metadata block) is already woven into the system
        prompt by ``_augment_system_prompt`` (called from batch_response_generation
        below), so re-stating it as a model turn would be redundant and could
        prime the model to parrot the raw metadata. The chat history is still
        replayed verbatim, and the conversation is closed on a trailing ``user``
        turn ("Respond now.") — Gemini requires the final turn to be ``user`` for
        it to generate a reply. Returns the list[{"role", "parts":[{"text"}]}]
        structure Gemini's chat API expects.
        """
        # NB: sys_prompt is already augmented in batch_response_generation, so
        # recommend_item is inside sys_prompt. Still keep the conversation history.
        msgs = []
        # Replay each prior turn, mapping our role vocabulary onto Gemini's:
        # everything that is not "user" (i.e. the assistant) becomes "model".
        for h in chat_history:
            role = "user" if h["role"] == "user" else "model"
            msgs.append({"role": role, "parts": [{"text": h["content"]}]})
        # Final turn MUST be "user" so Gemini produces an assistant ("model")
        # completion; this generic nudge carries no content because the actual
        # instructions all live in the augmented system prompt.
        msgs.append({"role": "user", "parts": [{"text": "Respond now."}]})
        return msgs

    def batch_response_generation(
        self,
        sys_prompts: list[str],
        chat_histories: list[list[dict]],
        recommend_items: list[str],
        max_new_tokens: int | None = None,
        user_profiles: list[dict | None] | None = None,
        track_metadatas: list[dict | None] | None = None,
    ) -> list[str]:
        """Augment each row's system prompt, then defer to GEMINI_MODEL's batch.

        Extends the base signature with optional per-row ``user_profiles`` and
        ``track_metadatas`` (each aligned positionally with ``sys_prompts``);
        when absent, the style packet falls back to its neutral defaults. After
        injecting the style packet / directives / metadata into each system
        prompt via ``_augment_system_prompt``, the actual API calls and caching
        are handled by the parent class.
        """
        # Augment per-row
        augmented = []
        for i, (sp, ri) in enumerate(zip(sys_prompts, recommend_items)):
            user_profile = (user_profiles[i] if user_profiles else None) or None
            track_metadata = (track_metadatas[i] if track_metadatas else None) or None
            augmented.append(self._augment_system_prompt(sp, ri, user_profile, track_metadata))
        return super().batch_response_generation(augmented, chat_histories, recommend_items, max_new_tokens)
