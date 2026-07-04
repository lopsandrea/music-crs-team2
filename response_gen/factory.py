"""Factory for the response-generation LM family.

`build_generator()` returns a configured Gemini response generator: the two-pass
Pro->Flash refiner on gemini-3.1-pro-preview, temperature 0.8, thinking budget 2048.

Selectable via `gemini_class`:
  - "pro"     -> GEMINI_PRO_MODEL        (single-pass Pro draft)
  - "refiner" -> GEMINI_PRO_REFINER      (two-pass Pro draft -> Flash critic)  [DEFAULT]
"""

from __future__ import annotations

from .gemini_pro import GEMINI_PRO_MODEL
from .gemini_pro_refiner import GEMINI_PRO_REFINER


# Registry mapping the public ``gemini_class`` selector string to its generator
# class. "refiner" (the two-pass default) and "pro" (single-pass) are the only
# advertised choices; build_generator validates against these keys.
LM_CLASSES = {
    "pro": GEMINI_PRO_MODEL,
    "refiner": GEMINI_PRO_REFINER,
}


def build_generator(
    # The default literals below are the verified reference configuration of the
    # shipped submissions; changing any of them moves off that path, so they are
    # pinned here.
    gemini_class: str = "refiner",                 # two-pass default (see LM_CLASSES)
    model_name: str = "gemini-3.1-pro-preview",    # Pass-1 draft model
    temperature: float = 0.8,                      # draft sampling temperature
    thinking_budget: int = 2048,                   # tokens reserved for the model's reasoning trace
    max_new_tokens: int = 8192,                    # output budget large enough to leave room after thinking
    max_concurrent: int = 8,                       # in-flight async requests cap (rate-limit friendly)
    cache_dir: str = "./cache/gemini",             # shared on-disk response cache for both passes
    use_style_packet: bool = True,                 # enable the stochastic MusicMind style packet
    # class-specific
    refiner_model: str = "gemini-3.5-flash",   # refiner: cheaper Pass-2 Flash critic model
    **extra,
):
    """Construct a configured response generator.

    Defaults: the two-pass refiner on gemini-3.1-pro-preview, temp 0.8, thinking 2048.
    Pair with a prompt variant from ``response_gen.prompts.load_prompt_variant(...)``.

    Returns an object exposing the text-in/out API:
        ``batch_response_generation(sys_prompts, chat_histories, recommend_items)``.
    """
    if gemini_class not in LM_CLASSES:
        raise ValueError(
            f"unknown gemini_class {gemini_class!r}; choose one of {sorted(LM_CLASSES)}"
        )

    # Common constructor kwargs accepted by both generator classes.
    kwargs = dict(
        model_name=model_name,
        max_concurrent=max_concurrent,
        cache_dir=cache_dir,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        thinking_budget=thinking_budget,
        use_style_packet=use_style_packet,
    )
    # refiner_model is only meaningful for the two-pass refiner; the single-pass
    # "pro" class does not accept it.
    if gemini_class == "refiner":
        kwargs["refiner_model"] = refiner_model
    # **extra is an escape hatch for less-common constructor args (e.g. top_p,
    # min_words, seen_bigrams); it can override anything assembled above.
    kwargs.update(extra)

    return LM_CLASSES[gemini_class](**kwargs)
