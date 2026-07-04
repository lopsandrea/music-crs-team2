"""response_gen — standalone music-recommendation response generator.

Extracted from the RecSys Challenge 2026 Music-CRS project. Produces the natural-language
``predicted_response`` that drives the LLM-as-judge metric. The default configuration
is the two-pass Pro->Flash refiner on gemini-3.1-pro-preview.

Quick start (pure text-in / text-out):

    from response_gen import build_generator, load_prompt_variant, assemble_system_prompt

    gen = build_generator()                       # two-pass refiner (default)
    rp, rg, pe = load_prompt_variant("musicmind")
    sys_prompt = assemble_system_prompt(rp, rg, pe, profile_str="age_group: 25-34\\ncountry_name: Italy")

    out = gen.batch_response_generation(
        [sys_prompt],
        [[{"role": "user", "content": "something jazzy and late-night"}]],
        ["track_id: T123, artist: ..., genres: ..., ..."],
    )

Auth: set ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` or write the key to
``~/.config/gemini/api_key`` (chmod 600).
"""

from __future__ import annotations

# Re-export the package's public surface so callers import everything they need
# from ``response_gen`` directly (see the quick-start above). The generator
# classes are exposed alongside the higher-level factory/prompt helpers because
# ``__all__`` advertises them as supported entry points.
from .gemini import GEMINI_MODEL
from .gemini_pro import GEMINI_PRO_MODEL, build_style_packet
from .gemini_pro_refiner import GEMINI_PRO_REFINER
from .factory import build_generator, LM_CLASSES
from .prompts import load_prompt_variant, assemble_system_prompt

__all__ = [
    "build_generator",
    "LM_CLASSES",
    "load_prompt_variant",
    "assemble_system_prompt",
    "build_style_packet",
    "GEMINI_MODEL",
    "GEMINI_PRO_MODEL",
    "GEMINI_PRO_REFINER",
]
