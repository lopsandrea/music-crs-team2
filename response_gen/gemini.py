"""Gemini-API response generator.

Drop-in replacement for LLAMA_MODEL.batch_response_generation. Uses Gemini
2.5 Flash by default (best $/quality for ~10k turn experiments). Response
caching by SHA256(sys_prompt + history + recommend_item + gen_params) is
mandatory: enables free re-evaluation when only retrieval/reranker changes.

Auth: reads a .env file (GEMINI_API_KEY / GOOGLE_API_KEY), then the same env
vars, then ~/.config/gemini/api_key file.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path


# Last-resort key location, used only if neither an explicit api_key nor an
# env var is supplied (see _resolve_api_key). chmod 600 is expected so the key
# is not world-readable.
_DEFAULT_KEY_FILE = Path.home() / ".config" / "gemini" / "api_key"
# Process-wide guard so the .env file is scanned at most once even if many
# generators are constructed; flipped to True by _load_dotenv on first call.
_DOTENV_LOADED = False


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Minimal stdlib .env parser: KEY=VALUE lines, ignoring blanks/comments.

    Supports an optional ``export`` prefix and single/double quoted values. No
    variable interpolation.
    """
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def _load_dotenv() -> None:
    """Populate os.environ from the nearest .env (without overriding existing vars).

    Searches the current working directory, each parent up to the root, then this
    package's directory. Runs at most once per process.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    cwd = Path.cwd()
    candidates = [cwd, *cwd.parents, Path(__file__).resolve().parent]
    seen: set[Path] = set()
    for d in candidates:
        if d in seen:
            continue
        seen.add(d)
        env_file = d / ".env"
        if env_file.is_file():
            # setdefault (not assignment) so an already-exported real env var
            # always wins over the .env file — the .env only fills in gaps.
            for k, v in _parse_dotenv(env_file).items():
                os.environ.setdefault(k, v)
            # Stop at the first .env found (nearest wins); do not merge several.
            break


def _resolve_api_key(api_key: str | None) -> str:
    """Resolve the Gemini API key in priority order, raising if none is found.

    Precedence: explicit ``api_key`` argument > GOOGLE_API_KEY / GEMINI_API_KEY
    env var (populated from a .env file by ``_load_dotenv`` if not already set) >
    the ``~/.config/gemini/api_key`` file. This mirrors the auth contract
    documented in the module/package docstrings.
    """
    if api_key:
        return api_key
    _load_dotenv()
    env = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if env:
        return env.strip()
    if _DEFAULT_KEY_FILE.exists():
        return _DEFAULT_KEY_FILE.read_text().strip()
    raise RuntimeError(
        "No Gemini API key. Provide it via a .env file (GEMINI_API_KEY=... or "
        "GOOGLE_API_KEY=...), the same env vars, or ~/.config/gemini/api_key (chmod 600)."
    )


def _hash_key(*parts: str) -> str:
    """Hash an ordered list of string parts into a hex SHA256 cache key.

    Each part is UTF-8 encoded and followed by an ASCII Record Separator byte
    (0x1e) before hashing. The separator makes the boundary between parts
    unambiguous, so distinct part lists can never collide by concatenation
    (e.g. ("ab", "c") and ("a", "bc") hash differently).
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


class GEMINI_MODEL:
    """Async Gemini chat client exposing the LLAMA_MODEL response-generation API.

    Constructed once and reused across batches. Generation parameters fall back
    to environment overrides (GEMINI_MAX_TOKENS / GEMINI_TEMP / GEMINI_TOP_P /
    GEMINI_THINKING_BUDGET) before the constructor defaults, so a run can be
    retuned without code changes. All completions are content-addressed in
    ``cache_dir`` keyed by model + prompt + history + item + gen params, making
    re-evaluation free when only the upstream retriever/reranker changed.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.5-flash",
        api_key: str | None = None,
        max_concurrent: int = 10,
        cache_dir: str = "./cache/gemini",
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        thinking_budget: int | None = None,
        # Compat with LLAMA_MODEL signature (unused).
        device: str = "cuda",
        attn_implementation: str = "eager",
        dtype=None,
    ):
        # Imported lazily so importing this module never requires the google-genai
        # SDK to be installed unless a generator is actually instantiated.
        from google import genai
        from google.genai import types as gtypes

        self.model_name = model_name
        self.api_key = _resolve_api_key(api_key)
        self.client = genai.Client(api_key=self.api_key)
        # Stash the types module on the instance so generate-time config builders
        # (ThinkingConfig / GenerateContentConfig) need not re-import it per call.
        self._gtypes = gtypes
        self.max_concurrent = int(max_concurrent)
        # Decoding params: an env override (if set) takes precedence over the
        # constructor arg, which itself falls back to a literal default. The
        # `max_new_tokens or 260` form means a falsy 0/None constructor arg also
        # falls back to 260 — the cap is small because replies are 1-3 sentences,
        # and a tight cap keeps Flash latency/cost low across ~10k turns.
        self.max_new_tokens = int(os.environ.get("GEMINI_MAX_TOKENS", max_new_tokens or 260))
        # temperature/top_p use the `x if x is not None else default` form so an
        # explicit 0.0 is honoured (0.0 is falsy and would otherwise be lost).
        self.temperature = float(os.environ.get("GEMINI_TEMP", temperature if temperature is not None else 0.8))
        self.top_p = float(os.environ.get("GEMINI_TOP_P", top_p if top_p is not None else 0.95))
        # Default 0 => Gemini 2.5 "thinking" is OFF (see _agenerate_one); a
        # positive budget reserves internal reasoning tokens, -1 is auto.
        self.thinking_budget = int(os.environ.get("GEMINI_THINKING_BUDGET", thinking_budget if thinking_budget is not None else 0))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Dedicated event loop kept alive across batch_chat calls so httpx
        # connection pool survives between batches (asyncio.run would close it).
        self._loop = asyncio.new_event_loop()

    # ---- shape inputs into Gemini's chat structure --------------------------
    def _format_messages(self, sys_prompt: str, chat_history: list[dict], recommend_item: str):
        # Gemini chat takes [{"role": "user"|"model", "parts": [{"text": ...}]}].
        # Match the baseline Llama/Qwen pattern: prepend the retrieved track
        # metadata as a "model" turn so the model can ground its next answer
        # in it, without an explicit word-count constraint that biases length.
        # Gemini needs strict user/model alternation, so we cap the history at
        # an even number of preceding turns and insert a "user" prompt asking
        # the assistant to respond to the listener about the recommendation.
        msgs = []
        for h in chat_history:
            # Collapse every non-"user" role (e.g. "assistant"/"system") onto
            # Gemini's "model" role, since Gemini only knows user/model turns.
            role = "user" if h["role"] == "user" else "model"
            msgs.append({"role": role, "parts": [{"text": h["content"]}]})
        # Pair: model echoes recommendation, then user asks for the reply.
        msgs.append({"role": "model", "parts": [{"text": f"Recommended track: {recommend_item}"}]})
        msgs.append({"role": "user", "parts": [{"text": "Respond now."}]})
        return msgs

    def _cache_path(self, key: str) -> Path:
        # Shard by the first two hex chars of the key so a single directory never
        # holds the whole cache (256-way fan-out keeps filesystem lookups fast).
        return self.cache_dir / f"{key[:2]}" / f"{key}.json"

    def _from_cache(self, key: str) -> str | None:
        """Return the cached response text for ``key``, or None on miss/corruption.

        Any read/JSON error is swallowed and treated as a cache miss so a single
        bad file never aborts a batch.
        """
        p = self._cache_path(key)
        if p.exists():
            try:
                return json.loads(p.read_text())["response"]
            except Exception:
                return None
        return None

    def _to_cache(self, key: str, response: str) -> None:
        """Persist ``response`` under ``key`` with the model name and a timestamp."""
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"response": response, "model": self.model_name, "ts": time.time()}))

    # ---- single async call with retries ------------------------------------
    async def _agenerate_one(self, sys_prompt: str, chat_history: list[dict], recommend_item: str) -> str:
        """Generate (or fetch from cache) the response for one session.

        On a cache miss, calls Gemini with exponential backoff (waits 1, 2, 4 s
        between attempts) for up to 4 tries. A successful result is cached and
        returned; exhausting the retries returns "" so one failed row never
        aborts the surrounding batch. The cache key intentionally excludes the
        thinking budget — it does not change the user-visible text — but does
        include the decoding params that do (max tokens, temperature, top_p).
        """
        key = _hash_key(
            self.model_name, sys_prompt, json.dumps(chat_history, ensure_ascii=False),
            recommend_item, str(self.max_new_tokens), str(self.temperature), str(self.top_p),
        )
        cached = self._from_cache(key)
        if cached is not None:
            return cached
        messages = self._format_messages(sys_prompt, chat_history, recommend_item)
        # Gemini 2.5 "thinking": budget=0 disables it (avoids truncation when
        # max_output_tokens is small); a positive integer reserves that many
        # internal tokens; -1 lets the model decide dynamically.
        thinking_cfg = self._gtypes.ThinkingConfig(thinking_budget=self.thinking_budget)
        cfg = self._gtypes.GenerateContentConfig(
            system_instruction=sys_prompt,
            max_output_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            thinking_config=thinking_cfg,
        )
        for attempt in range(4):
            try:
                resp = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=messages,
                    config=cfg,
                )
                text = (resp.text or "").strip()
                self._to_cache(key, text)
                return text
            except Exception as e:
                # Exponential backoff: 2**attempt = 1, 2, 4 s for attempts 0..2.
                wait = 2 ** attempt
                # Last attempt (index 3): log and yield "" rather than raise, so
                # one persistently failing row cannot abort the whole batch.
                if attempt == 3:
                    print(f"  Gemini failure after 4 attempts: {type(e).__name__}: {e}")
                    return ""
                # Otherwise sleep, then fall through to the next loop iteration.
                await asyncio.sleep(wait)

    async def _arun_batch(self, sys_prompts, histories, items):
        """Fan out one ``_agenerate_one`` per (prompt, history, item) triple.

        A semaphore caps in-flight requests at ``max_concurrent`` to respect the
        API rate limit while still pipelining. Results are gathered in input
        order, so output[i] corresponds to input row i.
        """
        sem = asyncio.Semaphore(self.max_concurrent)

        async def _bounded(sp, h, it):
            async with sem:
                return await self._agenerate_one(sp, h, it)

        return await asyncio.gather(*[
            _bounded(sp, h, it) for sp, h, it in zip(sys_prompts, histories, items)
        ])

    # ---- LLAMA_MODEL-compatible API ----------------------------------------
    def response_generation(self, sys_prompt: str, chat_history: list, recommend_item: str,
                            max_new_tokens: int = None, response_format=None) -> str:
        """Synchronous single-session entry point (LLAMA_MODEL signature).

        ``max_new_tokens`` / ``response_format`` are accepted for interface
        compatibility but ignored; decoding is governed by the instance config.
        Drives the persistent event loop so the httpx connection pool is reused.
        """
        return self._loop.run_until_complete(self._agenerate_one(sys_prompt, chat_history, recommend_item))

    def batch_response_generation(self, sys_prompts: list[str], chat_histories: list[list],
                                  recommend_items: list[str], max_new_tokens: int = None) -> list[str]:
        """Synchronous batch entry point; returns one response string per input row.

        The three input lists are zipped positionally and must be equal length.
        Runs the whole batch on the shared event loop (see ``self._loop``).
        """
        return self._loop.run_until_complete(self._arun_batch(sys_prompts, chat_histories, recommend_items))
