"""Two-pass refiner variant.

Pass 1: GEMINI_PRO_MODEL produces a draft response.
Pass 2: a cheaper "critic" model (gemini-3.5-flash by default) rewrites the
        draft to strip AI-helpdesk phrases / filler while preserving the
        recommendation, personalisation, and music knowledge.

If the refiner output is suspiciously short (<100 words) we fall back to the
draft to avoid catastrophic over-shortening regressing the lex / judge score.
"""

from __future__ import annotations

from .gemini import GEMINI_MODEL
from .gemini_pro import GEMINI_PRO_MODEL


# System/critic instruction for pass 2. The Flash critic is told exactly what to
# strip (helpdesk openers, filler phrases) and what to preserve (track, length,
# personalization, music knowledge); the trailing "REFINED:" cue primes it to
# emit only the rewritten response with no preamble.
# NOTE on the two length numbers: this prompt ASKS the critic to keep >=120 words
# (rule 4), but the actual programmatic acceptance floor in _agenerate_one is a
# laxer 100 words. The gap is deliberate: 120 is the soft target steering the
# edit, while the 100-word hard floor is the regression guard that only triggers
# on a genuinely over-shortened rewrite (the model usually lands above 120).
CRITIC_PROMPT = """You are a critic and editor. Your job: improve the following music recommendation response by:
1. Removing AI-helpdesk phrases ("Absolutely", "Sure", "I think", "Honestly")
2. Removing filler ("isn't just", "perfect blend", "sonic landscape")
3. Tightening repetitive sentences
4. Keeping the response at least 120 words
5. Keeping the recommendation, the personalization, and the music knowledge intact

DO NOT shorten dramatically. DO NOT change the recommended track. Output ONLY the refined response, no preamble."""


class GEMINI_PRO_REFINER(GEMINI_PRO_MODEL):
    """Two-pass generator: GEMINI_PRO draft, then a cheap Flash critic rewrite.

    Inherits the Pro draft path unchanged (style packet, prompt augmentation,
    etc.) and only overrides ``_agenerate_one`` to chain a second pass. This is
    the reference generator selected by ``build_generator``.

    Subclassing GEMINI_PRO_MODEL (not the plain GEMINI_MODEL) matters: the draft
    pass inherits the full Pro machinery — MusicMind persona, the stochastic
    style packet, and the per-row ``_augment_system_prompt`` that bakes the
    recommendation/metadata into the system prompt. Only the second (critic) pass
    is the cheaper plain GEMINI_MODEL built in __init__. The critic is a SEPARATE
    object held as ``self.refiner``; it is not part of this class's own MRO call
    chain, so ``super()._agenerate_one`` below always reaches the Pro draft path.
    """

    def __init__(self, *args, refiner_model: str = "gemini-3.5-flash", **kwargs):
        """Init the Pro draft model, then build a separate critic generator.

        The critic is a plain GEMINI_MODEL on ``refiner_model`` with a lower
        temperature (0.6) for a steadier edit and a smaller token budget (2048,
        enough for a polished single response). It reuses this instance's
        concurrency limit and cache directory so both passes share one cache.
        """
        super().__init__(*args, **kwargs)
        self.refiner = GEMINI_MODEL(
            model_name=refiner_model,
            temperature=0.6,
            max_new_tokens=2048,
            max_concurrent=self.max_concurrent,
            cache_dir=str(self.cache_dir),
        )

    async def _agenerate_one(self, sys_prompt, chat_history, recommend_item):
        """Produce a Pro draft, then return the Flash-refined rewrite of it.

        Pass 1 is the inherited Pro generation. Pass 2 sends the draft to the
        critic wrapped in CRITIC_PROMPT. Guard rails: an empty draft short-
        circuits (nothing to refine); a refined output that is empty or under
        100 words is discarded in favour of the draft, preventing the critic
        from catastrophically over-shortening and regressing the score.
        """
        draft = await super()._agenerate_one(sys_prompt, chat_history, recommend_item)
        if not draft:
            return draft
        critic_history = [
            {
                "role": "user",
                "content": f"{CRITIC_PROMPT}\n\nORIGINAL:\n{draft}\n\nREFINED:",
            }
        ]
        # The critic runs as a generic editor: the draft + instructions live in
        # the user turn, so the system prompt is intentionally minimal.
        # recommend_item is still passed so it enters the critic's own cache key
        # (it disambiguates otherwise-identical drafts across sessions).
        refined = await self.refiner._agenerate_one(
            sys_prompt="You are an expert editor.",
            chat_history=critic_history,
            recommend_item=recommend_item,
        )
        # Length floor: reject suspiciously short rewrites and keep the draft.
        if not refined or len(refined.split()) < 100:
            return draft
        return refined
