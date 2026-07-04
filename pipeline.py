"""End-to-end pipeline: blind data → recommender → response_gen → submission.zip.

`run_pipeline(...)` produces a Codabench-ready zip. For ONE blind dataset it runs,
per session:

  Levels 0-6  recommender/  -> `predicted_track_ids` (top-20) + `recommend_item` (top-1 metadata).
              `Recommender.batch_recommend` runs the whole pipeline (parse, candidate
              sources, weighted-RRF fusion -> 300-pool, featurize -> 300x37, LightGBM LambdaRank +
              selective routing, CatBoost-YetiRank ensemble, argsort+skip-played -> top-20).
  Level 7     response_gen/ -> `predicted_response` (Gemini Pro draft -> Flash "refiner" polish).
  Packaging   a byte-for-byte deterministic prediction.json/submission.zip.

The Codabench composite has two INDEPENDENT halves: `predicted_track_ids` scored by
nDCG@20 (the recommender half), and `predicted_response` scored by an LLM judge (the
response half). They are produced by the two stages above and packaged side by side
per case. The recommendation half is fully deterministic; the response half calls the
Gemini API and is NOT bit-reproducible across runs (see README).
"""
# NOTE: the module docstring above must remain the very first statement, i.e. ABOVE the
# `from __future__` line below, or this file stops importing (a __future__ import is only
# legal at the top of a module, after at most the docstring).
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from recommender import Recommender
from recommender.data import load_blind_sessions
from response_gen import assemble_system_prompt, build_generator, load_prompt_variant


def _profile_str(session: dict) -> str:
    """Render the session's user_profile as newline-joined "key: value" lines for the prompt.

    Feeds the `profile_str` slot of the response_gen system prompt (Level 7) so the Gemini
    refiner can lightly personalise the reply. Only the three demographic fields the dataset
    exposes are surfaced, in a fixed order; missing/falsy values are skipped, so the result is
    "" when no profile is present (the prompt then simply omits the profile block).
    """
    # `user_profile` may be absent or null in a blind item -> coerce to an empty dict.
    p = session.get("user_profile") or {}
    bits = []
    # Fixed field order keeps the rendered prompt deterministic across runs/sessions.
    for k in ("age_group", "country_name", "gender"):
        # Guard against a non-dict profile (e.g. malformed item) before `.get`.
        v = p.get(k) if isinstance(p, dict) else None
        if v:
            bits.append(f"{k}: {v}")
    return "\n".join(bits)


def _chat_history(session: dict, catalog) -> list[dict]:
    """Render the conversation for response_gen (music turns → track string).

    Turns the parsed session `history` (Level 0 output: a list of prior turns, see
    recommender/data.py) into the `[{"role", "content"}, ...]` list the Gemini generator
    expects. The dataset uses three roles: 'user', 'assistant', and 'music' — a 'music' turn's
    content is a bare track_id (a track already recommended/played earlier in the dialogue). For
    the chat transcript we replace that opaque id with a human-readable "(recommended: NAME by
    ARTIST)" line so the LLM sees what it previously suggested rather than a meaningless id; every
    other role is passed through verbatim. The current `user_query` (the latest user turn that the
    whole pipeline is answering) is appended last so it is the final message in the transcript.

    `catalog` is the track-metadata map (`Recommender.metadata`), id -> metadata dict.
    """
    hist = []
    for h in session["history"]:
        role, content = h["role"], h["content"]
        if role == "music":
            # 'music' content is a track_id -> look up its metadata; default {} if id is unknown.
            m = catalog.get(str(content).strip(), {})
            name = m.get("track_name", ""); art = m.get("artist_name", "")
            # Some metadata fields are stored as single-element lists; unwrap to the scalar so the
            # rendered line reads "NAME by ARTIST" rather than "['NAME'] by ['ARTIST']".
            name = name[0] if isinstance(name, list) and name else name
            art = art[0] if isinstance(art, list) and art else art
            # A prior 'music' turn was the system recommending a track -> render it as an assistant turn.
            hist.append({"role": "assistant", "content": f"(recommended: {name} by {art})"})
        else:
            # 'user' / 'assistant' turns pass through unchanged (content coerced to str for safety).
            hist.append({"role": role, "content": str(content)})
    # The latest user request closes the transcript; it is what the response is generated for.
    hist.append({"role": "user", "content": session["user_query"]})
    return hist


def _deterministic_zip(out_zip: Path, items: list[dict]) -> tuple[str, str]:
    """Write `items` to `out_zip` as a byte-for-byte reproducible `prediction.json` archive.

    Codabench expects a single `prediction.json` inside the submission zip. We make the zip
    deterministic so an unchanged prediction always yields an identical file — useful for
    verification and for de-duplicating identical submissions. Two sources of non-determinism
    are pinned: (1) the member's modification timestamp is forced to the fixed
    (1980,1,1,0,0,0) — 1980-01-01 is the earliest date the ZIP format can store, so it is the
    conventional "zero" timestamp; (2) `json.dumps(..., indent=2)` gives stable, ordered text
    (dict key order is the insertion order used when building each item below).

    Returns (sha256 of the JSON payload, sha256 of the final zip bytes) — the payload hash is
    content-only, the zip hash includes the container; both are printed for traceability.
    """
    # Serialise once; this exact string is both hashed and written into the archive.
    payload = json.dumps(items, indent=2)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    # Fixed member name + fixed timestamp = reproducible archive metadata.
    info = zipfile.ZipInfo("prediction.json", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(out_zip, "w") as zf:
        zf.writestr(info, payload)
    # Hash the content (encode -> bytes) and, separately, the on-disk zip bytes.
    return (hashlib.sha256(payload.encode()).hexdigest(),
            hashlib.sha256(out_zip.read_bytes()).hexdigest())


def run_pipeline(blind_name="blind_b",
                 out_zip="submission.zip",
                 gemini_class="refiner", model_name="gemini-3.1-pro-preview",
                 prompt_variant="blindbnat", limit=None, enrich_items=True):
    """Run the full blind→submission flow and write the zip.

    Four printed stages ([1/4]..[4/4]) mirror the body below: load sessions, run the
    recommender (Levels 0-6), generate responses (Level 7), and package the deterministic
    zip. The defaults are the exact configuration of our validated Blind-B submission: the
    `refiner` generator on `gemini-3.1-pro-preview` with the `blindbnat` prompt variant and
    track-fact enrichment ON.

    Args:
        blind_name: which blind dataset to score ("blind_b" for this release); passed to
                load_blind_sessions. NOTE: serving "blind_a" also requires
                RECSYS_BLIND_NAME=blind_a in the environment so the precomputed
                struct_base/struct_large list paths resolve to that split (see config.py).
        out_zip: path of the submission.zip to write (parent dirs are created).
        gemini_class / model_name: select the response_gen generator class and its base model.
        prompt_variant: system-prompt variant key consumed by load_prompt_variant.
        limit: if set, truncate to the first N sessions (smoke tests / partial runs).
        enrich_items: when True (default), feed the generator the recommended track's rich
                Track-Metadata facts (clean tags + year + duration) instead of the plain top-1
                string, and append the grounding directive to the system prompt.

    Returns:
        {"zip", "n", "zip_sha256"}.
    """
    out_zip = Path(out_zip)
    # [1/4] Load and parse the blind sessions (Level 0 parsing happens inside the recommender,
    # but loading the raw session list happens here). `limit` truncates for quick partial runs.
    print(f"[1/4] loading {blind_name} sessions…", flush=True)
    sessions = load_blind_sessions(blind_name)
    if limit:
        sessions = sessions[:limit]
    print(f"      {len(sessions)} sessions")

    # [2/4] Recommendation half (Levels 0-6). `batch_recommend` returns, per session, a dict with
    # `predicted_track_ids` (top-20, the nDCG@20 half) and `recommend_item` (top-1 metadata string).
    print("[2/4] recommender → top-20 per session…", flush=True)
    rec = Recommender()
    recs = rec.batch_recommend(sessions)
    # Index sessions by id so each recommendation can be re-joined to its source session below.
    # `batch_recommend` may reorder/parallelise, so we look up by session_id rather than position.
    by_sid = {s["session_id"]: s for s in sessions}

    # [3/4] Response half (Level 7). Build the Gemini generator and the prompt-variant pieces once,
    # then assemble per-session inputs. The three lists are POSITIONALLY ALIGNED with `recs`:
    # element i is the system prompt / chat history / top-1 item for recs[i].
    print(f"[3/4] response_gen ({gemini_class}/{model_name}) → responses…", flush=True)
    # The style packet is OFF in the shipped configuration (plain prompt-driven responses).
    gen = build_generator(gemini_class=gemini_class, model_name=model_name,
                          use_style_packet=False)
    # rp/rg/pe = the three prompt fragments (roleplay/guidelines/personalization) of the variant,
    # combined by assemble_system_prompt together with the per-session profile string.
    rp, rg, pe = load_prompt_variant(prompt_variant)
    # Track-fact enrichment: richer item facts + a grounding directive, both gated on
    # `enrich_items`. See response_gen/track_facts.py.
    from response_gen.track_facts import rich_item, GROUNDING_DIRECTIVE
    sys_prompts, chat_histories, recommend_items = [], [], []
    user_profiles, track_metadatas = [], []
    for r in recs:
        s = by_sid[r["session_id"]]
        sp = assemble_system_prompt(rp, rg, pe, profile_str=_profile_str(s))
        if enrich_items:
            sp = sp + GROUNDING_DIRECTIVE
        sys_prompts.append(sp)
        # `rec.metadata` is the id->metadata catalog used to render 'music' turns readable.
        chat_histories.append(_chat_history(s, rec.metadata))
        # The top-1 track's metadata string is the concrete item the reply is written about; when
        # enriching, swap in the rich fact string (falling back to the plain item if id unknown).
        top1 = r["predicted_track_ids"][0] if r["predicted_track_ids"] else None
        if enrich_items and top1 is not None:
            recommend_items.append(rich_item(rec.metadata, top1) or r["recommend_item"])
        else:
            recommend_items.append(r["recommend_item"])
        # Per-row personalisation inputs accepted by the GEMINI_PRO generator family.
        user_profiles.append(s.get("user_profile"))
        track_metadatas.append(rec.metadata.get(str(top1)) if top1 is not None else None)
    # Batched generation (async fan-out inside the generator) -> one response string per case.
    # Only the GEMINI_PRO family accepts the per-row kwargs; pass them when supported.
    import inspect
    if "user_profiles" in inspect.signature(gen.batch_response_generation).parameters:
        responses = gen.batch_response_generation(sys_prompts, chat_histories, recommend_items,
                                                  user_profiles=user_profiles,
                                                  track_metadatas=track_metadatas)
    else:
        responses = gen.batch_response_generation(sys_prompts, chat_histories, recommend_items)

    # [4/4] Package the two halves side by side into the Codabench submission record. The dict key
    # order here is the on-disk JSON field order (relied upon by the deterministic-zip hashing).
    print("[4/4] packaging submission.zip…", flush=True)
    # `responses` is positionally aligned with `recs` (zip pairs case i with its response i).
    items = [{"session_id": r["session_id"], "turn_number": int(r["turn_number"]),
              "predicted_track_ids": r["predicted_track_ids"],
              "predicted_response": resp}
             for r, resp in zip(recs, responses)]
    pred_sha, zip_sha = _deterministic_zip(out_zip, items)
    # Diagnostic: count blank responses (a non-zero count flags a Gemini/quota failure to inspect).
    n_empty = sum(1 for it in items if not it["predicted_response"].strip())
    print(f"      {len(items)} cases | empty responses: {n_empty}")
    # Print only the leading 16 hex chars of each sha256 — enough to eyeball-match runs, not noisy.
    print(f"      prediction.json sha256 {pred_sha[:16]}… | zip {zip_sha[:16]}… → {out_zip}")
    return {"zip": str(out_zip), "n": len(items), "zip_sha256": zip_sha}
