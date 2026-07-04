"""Train the structured-query retriever (``struct_base``): 5-fold OOF dev lists +
a 5-fold blind ensemble.

What ``struct_base`` is
-----------------------
A supervised BGE-base bi-encoder (``BAAI/bge-base-en-v1.5``), same architecture and
contrastive recipe as the text_retriever (``training.text_retriever``), but fed a richer
**structured query** instead of the text_retriever's flat last-few-turns string. The query is
rendered as

    [QUERY] <current user turn> [HISTORY] <prior user turns> [CONTEXT] <recent tracks>

(see ``build_query_structured`` below). It is one of the nine candidate sources the
champion fuses (``recommender.config.SW_BASELINE["struct_base"]``).

Naming note: ``struct_base`` is the clean name for the legacy experiment-log token ``r54``
(``r54 == struct_base``). This trainer uses only the clean
names — its config constants are ``C.STRUCT_BASE_OOF`` / ``C.STRUCT_BASE_BLIND_LISTS`` /
``C.STRUCT_BASE_FOLD_DIRS`` (renamed from the locked ``R54_*`` constants). The ``r54`` token
itself survives ONLY where it is baked into a shipped artifact (e.g. the ``"r54_*"`` LightGBM
``feature_name`` entries and the ``case_features.pkl`` dict keys), none of which live in this
file; it is retained on purpose because renaming those would force the deferred GPU retrain.

Unlike the text_retriever — which ships a single *production* model that re-encodes the catalog
live — struct_base ships as a **5-fold ensemble**: at serve time
``recommender.sources.src_struct_base`` does NOT run a model; it reads the pre-computed blind
lists ``C.STRUCT_BASE_BLIND_LISTS`` (``{session_id: [[tid, score], ...]}``), produced here by
averaging the five fold models' cosine scores over the blind set. So this stage's two products
are:

* **OOF dev lists** (``C.STRUCT_BASE_OOF``) — for ``training.case_features``. Every dev case must
  be scored by a model that did NOT train on it, so the dev set is split into the same 5
  grouped-by-session folds as the rest of the pipeline (``training.folds``) and each fold's
  held-out cases are retrieved with the model trained on the other four. The five per-fold
  list blocks are stitched case-index-ordered into
  ``{"lists": [n_cases][[tid, score], ...]}`` — exactly the format
  ``case_features._load_struct_base_oof`` reads (it takes ``tid``/``score`` from each pair).

* **Blind ensemble lists** (``C.STRUCT_BASE_BLIND_LISTS``) — for blind/live inference. Each blind
  query is encoded by all five fold models, their cosine scores against each fold's cached
  catalog embeddings are averaged, and the top-300 (played excluded) is written per session
  as ``{"lists": {sid: [[tid, score], ...]}, "manifest": {...}}`` — the contract
  ``recommender.sources.src_struct_base`` consumes (it slices ``pairs[:C.POOL_K]``).

Fold contract (critical — must match the rest of the pipeline)
--------------------------------------------------------------
Folds are assigned by ``training.folds.grouped_session_folds(sessions, seed=0, k=5)`` over
the dev cases in their ``C.DEV_PAYLOAD["cases"]`` order — the SAME split the text_retriever's
OOF folds, struct_large's OOF folds, and ``case_features`` use, so every retriever holds out
the same cases per fold. Verified byte-identical to the canonical fold map (8000/8000).

Why struct_base's fold models are an ensemble, not a single production model
---------------------------------------------------------------------------
Each fold model trains on 80% of dev (the four held-in folds) plus a fixed 20k-pair
train-split sample. The union of all five folds' dev data covers 100% of dev (each dev
case is held-in for four of the five models), so the ensemble is functionally close to a
"train on all dev" production model while also damping the per-fold cosine variance.

Provenance: faithfully reconstructed from the validated structured-query phase-3 5-fold trainer
+ blind ensemble; the heavy 5-fold + blind-encode rebuild is the user's deferred command
(~GPU-hours on the 2× RTX 4090). The original trainer's diagnostic/operational machinery
(mid-epoch checkpoint/resume, smoke-fold reuse, standalone manifest annotations) is dropped;
only what produces the champion artifacts (the per-fold OOF blocks, the aggregated
``C.STRUCT_BASE_OOF``, and the blind ensemble lists) is kept. The structured-query construction,
the contrastive training recipe, the OOF retrieval, and the average-cosine ensemble are
reproduced exactly.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np

from recommender import config as C
from recommender.data import load_blind_sessions, load_track_metadata
from training.folds import grouped_session_folds

# --- training hyper-parameters (verbatim from the validated phase-3 trainer) ---
MODEL_NAME = "BAAI/bge-base-en-v1.5"
EPOCHS = 1                  # struct_base trains ONE epoch per fold (text_retriever uses 2)
BATCH_SIZE = 32
LR = 2e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
TEMPERATURE = 0.05          # in-batch contrastive softmax temperature (legacy ``TAU``)
MAX_LENGTH = 256            # tokenizer truncation length (legacy ``MAX_SEQ_LEN``)
TOPK = 300                  # OOF / blind list length
N_FOLDS = 5
FOLD_SEED = 0               # grouped-session CV seed — the pipeline-wide fold contract

# structured-query window sizes (verbatim)
HISTORY_TURNS = 3           # trailing prior user utterances kept in [HISTORY]
CONTEXT_TRACKS = 5          # trailing played tracks rendered in [CONTEXT]

# train-split sampling (verbatim): a fixed extra-supervision pool drawn from the train
# split, capped per session, shared by all folds.
TRAIN_SPLIT_SAMPLE_PAIRS = 20000
MAX_PAIRS_PER_SESSION = 2
SAMPLING_SEED = 0
TRAINING_SEED = 0           # per-fold seed = TRAINING_SEED + fold_idx

# encode batch sizes (verbatim: cuda/cpu split in the legacy trainer)
ENCODE_TRACK_BS_CUDA = 256
ENCODE_TRACK_BS_CPU = 128
ENCODE_QUERY_BS_CUDA = 256
ENCODE_QUERY_BS_CPU = 64
ENCODE_BLIND_BS = 32        # blind-ensemble query encode batch (legacy ``encode_queries``)


def _ts() -> str:
    """Wall-clock ``[YYYY-MM-DD HH:MM:SS]`` stamp prefixed to the progress prints."""
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


def _device() -> str:
    """``"cuda"`` if a GPU is visible, else ``"cpu"``.

    Wrapped in try/except so the module still imports (and the CPU path runs) on a
    box without torch installed — selection is by availability, not configuration.
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------- text construction (struct_base-specific; ported verbatim) ----------------

def _year_of(release_date) -> "int | None":
    """First 4-digit year from a release_date (scalar ``"2006-12-06"`` or ``["2006-…"]``).

    Returns ``None`` when absent, non-string, or not a plausible 4-digit year (1900–2100) —
    so the 1.4% of tracks without a parseable date simply get no era clause.
    """
    v = release_date[0] if isinstance(release_date, list) and release_date else release_date
    if not isinstance(v, str) or len(v) < 4 or not v[:4].isdigit():
        return None
    y = int(v[:4])
    return y if 1900 <= y <= 2100 else None


def build_track_text(tid: str, meta: dict[str, dict]) -> str:
    """Catalog/positive-side text: ``"<name> by <artist>. Album: <alb>. Tags: <tags>"``.

    Identical to the text_retriever's ``build_track_text``; reproduced here so
    struct_base has no cross-module dependency.
    """
    m = meta.get(tid, {})
    names = m.get("track_name", [])
    artists = m.get("artist_name", [])
    album = m.get("album_name", [])
    tags = m.get("tag_list", [])
    name = names[0] if isinstance(names, list) and names else str(names)
    artist = ", ".join(artists) if isinstance(artists, list) else str(artists)
    alb = album[0] if isinstance(album, list) and album else str(album)
    tag_str = ", ".join(str(t) for t in tags[:10]) if isinstance(tags, list) else str(tags)
    return f"{name} by {artist}. Album: {alb}. Tags: {tag_str}"


def build_short_track_ref(tid: str, meta: dict[str, dict]) -> str:
    """Compact ``"<name> by <artist>"`` used inside the query's ``[CONTEXT]`` block."""
    m = meta.get(tid, {})
    names = m.get("track_name", [])
    artists = m.get("artist_name", [])
    name = names[0] if isinstance(names, list) and names else str(names)
    artist = artists[0] if isinstance(artists, list) and artists else str(artists)
    return f"{name} by {artist}"


def build_query_structured(current_query: str, prior_user_utterances: list[str],
                           played: list[str], meta: dict[str, dict]) -> str:
    """The struct_base structured query: ``[QUERY] … [HISTORY] … [CONTEXT] …``.

    Renders the current user turn, the trailing ``HISTORY_TURNS`` *prior* user utterances,
    and the trailing ``CONTEXT_TRACKS`` played tracks (as ``"<name> by <artist>"``).
    Empty ``[HISTORY]`` / ``[CONTEXT]`` blocks are omitted. This is the single
    construction shared by the dev-OOF and blind sides (the legacy
    ``build_query_structured_from_dev`` / ``build_query_structured_blind``, which were
    line-for-line equivalent given prior-only history).

    Args:
        current_query: the current user turn (the case's ``user_query`` / blind last user
            turn).
        prior_user_utterances: user-role utterances strictly before the current turn, in
            order; the trailing ``HISTORY_TURNS`` are kept.
        played: track ids played strictly before the current turn, in order; the trailing
            ``CONTEXT_TRACKS`` present in ``meta`` become the context refs.
    """
    history = prior_user_utterances[-HISTORY_TURNS:]
    context_tracks = [build_short_track_ref(t, meta) for t in played[-CONTEXT_TRACKS:]
                      if t in meta]
    parts = [f"[QUERY] {current_query}"]
    if history:
        parts.append(f"[HISTORY] {' '.join(history)}")
    if context_tracks:
        parts.append(f"[CONTEXT] {'; '.join(context_tracks)}")
    return " ".join(parts)


def _query_from_dev_case(case: dict, meta: dict[str, dict]) -> str:
    """Structured query for a dev case (``history`` is prior-only, ``user_query`` current).

    Verbatim ``build_query_structured_from_dev``: collect prior user-role contents and
    prior played tids from ``case["history"]``, then render via ``build_query_structured``.
    """
    prior_user = [str(h.get("content", "")) for h in case["history"]
                  if h.get("role", "") == "user"]
    played = [str(h.get("content", "")).strip() for h in case["history"]
              if h.get("role", "") == "music"]
    return build_query_structured(case["user_query"], prior_user, played, meta)


def _query_from_session(user_msgs_so_far: list[str], played_so_far: list[str],
                        current_user_msg: str, meta: dict[str, dict]) -> str:
    """Structured query for a *train-split* turn (``user_msgs_so_far`` INCLUDES current).

    Verbatim ``build_query_structured_from_session``: because the running utterance list
    already contains the current message, the prior history is ``history[:-1]`` (the
    trailing ``HISTORY_TURNS`` taken first, then the current one dropped).
    """
    history = user_msgs_so_far[-HISTORY_TURNS:] if len(user_msgs_so_far) > HISTORY_TURNS \
        else user_msgs_so_far
    older_history = history[:-1] if history else []
    return build_query_structured(current_user_msg, older_history, played_so_far, meta)


# ---------------- catalog ----------------

def _load_catalog() -> tuple[dict[str, dict], list[str]]:
    """``(meta, all_track_ids)`` over the full ``all_tracks`` catalog.

    Reuses ``recommender.data.load_track_metadata`` (the same HF ``all_tracks`` split the
    legacy ``load_catalog`` read); the ordered id list is the dict's insertion order, i.e.
    the dataset row order — identical to the legacy trainer's arrow-iteration order, so the
    per-fold catalog embeddings line up with ``all_track_ids``.
    """
    meta = load_track_metadata()
    all_track_ids = list(meta.keys())
    assert len(all_track_ids) == len(set(all_track_ids)), "duplicate track IDs in catalog"
    return meta, all_track_ids


# ---------------- train-split extra-supervision sample (shared by all folds) ----------------

def _build_train_split_sample(meta: dict[str, dict], dev_session_ids: set[str],
                              smoke: bool = False) -> list[tuple[str, str]]:
    """The fixed 20k ``(query, positive)`` pool drawn from the train split.

    Verbatim ``build_train_split_sample``: every (user-turn -> next in-catalog track)
    transition in the train split (excluding sessions that appear in dev) becomes a
    structured-query/track-text pair; pairs are capped at ``MAX_PAIRS_PER_SESSION`` per
    session (seeded subsample) and the capped pool is subsampled to
    ``TRAIN_SPLIT_SAMPLE_PAIRS`` (seed ``SAMPLING_SEED``). Under ``smoke`` only a tiny
    pool is drawn so the path still exercises real train-split pairs without the full scan.
    """
    from collections import defaultdict

    from datasets import DownloadConfig, load_dataset
    try:
        train_ds = load_dataset(C.DS_CONVO,
                                download_config=DownloadConfig(local_files_only=True))["train"]
    except Exception:
        train_ds = load_dataset(C.DS_CONVO)["train"]

    target_pairs = 256 if smoke else TRAIN_SPLIT_SAMPLE_PAIRS
    session_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    n_dev_overlap = 0
    for item in train_ds:
        sid = item["session_id"]
        if sid in dev_session_ids:
            n_dev_overlap += 1
            continue
        user_msgs_so_far: list[str] = []
        played_so_far: list[str] = []
        most_recent_user_msg = ""
        for conv in item["conversations"]:
            role = conv["role"]
            content = str(conv["content"])
            if role == "user":
                user_msgs_so_far.append(content)
                most_recent_user_msg = content
            elif role == "music":
                tid = content.strip()
                if tid not in meta:
                    played_so_far.append(tid)
                    continue
                if most_recent_user_msg:
                    q = _query_from_session(user_msgs_so_far, played_so_far,
                                            most_recent_user_msg, meta)
                    session_pairs[sid].append((q, build_track_text(tid, meta)))
                played_so_far.append(tid)
        # Under smoke, stop scanning once we have comfortably more capped pairs than needed.
        if smoke and len(session_pairs) * MAX_PAIRS_PER_SESSION >= target_pairs * 4:
            break

    rng = np.random.RandomState(SAMPLING_SEED)
    capped_pool: list[tuple[str, str, str]] = []
    for sid in sorted(session_pairs.keys()):
        pairs = session_pairs[sid]
        if len(pairs) <= MAX_PAIRS_PER_SESSION:
            capped_pool.extend((sid, q, t) for q, t in pairs)
        else:
            idx = rng.choice(len(pairs), MAX_PAIRS_PER_SESSION, replace=False)
            capped_pool.extend((sid, pairs[j][0], pairs[j][1]) for j in idx)

    if len(capped_pool) <= target_pairs:
        sampled = capped_pool
    else:
        idx = rng.choice(len(capped_pool), target_pairs, replace=False)
        sampled = [capped_pool[i] for i in idx]
    n_sessions = len({s for s, _, _ in sampled})
    print(f"  [struct_base] train_split: {n_dev_overlap} dev-overlap skipped, "
          f"{len(sampled)} pairs from {n_sessions} sessions", flush=True)
    return [(q, t) for _, q, t in sampled]


# ---------------- training / encoding ----------------

def _train_fold(fold_idx: int, all_pairs: list[tuple[str, str]], model_dir, device: str,
                epochs: int = EPOCHS):
    """Fine-tune BGE-base with the in-batch contrastive loss; save to ``model_dir``.

    Verbatim recipe from the legacy ``train_fold``: per-fold seed ``TRAINING_SEED +
    fold_idx`` (numpy + torch), one shuffled permutation of the pairs, then per batch
    encode queries and positives WITH grads, score ``q @ p.T / TEMPERATURE``, cross-entropy
    against the diagonal, AdamW(lr, weight_decay), grad-clip 1.0. (The original's mid-epoch
    pickle checkpoint/resume is dropped — per-fold *artifact* resume is handled in
    ``build`` by skipping folds whose ``oof_lists.json`` exists.)
    """
    import torch
    import torch.nn.functional as F_t
    from sentence_transformers import SentenceTransformer

    total_batches = (len(all_pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  [struct_base] fold {fold_idx}: {len(all_pairs)} pairs, {total_batches} batches, "
          f"device={device}", flush=True)

    np.random.seed(TRAINING_SEED + fold_idx)
    torch.manual_seed(TRAINING_SEED + fold_idx)
    if device == "cuda":
        torch.cuda.manual_seed_all(TRAINING_SEED + fold_idx)
    perm = np.random.permutation(len(all_pairs)).astype(np.int64)

    model = SentenceTransformer(MODEL_NAME, device=device)
    tokenizer = model.tokenizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()

    def encode_with_grad(texts):
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH,
                            return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out = model.forward(encoded)
        return F_t.normalize(out["sentence_embedding"], dim=-1)

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx in range(total_batches):
            start = batch_idx * BATCH_SIZE
            batch_indices = perm[start:start + BATCH_SIZE]
            queries = [all_pairs[int(i)][0] for i in batch_indices]
            positives = [all_pairs[int(i)][1] for i in batch_indices]
            if not queries:
                continue
            q_emb = encode_with_grad(queries)
            p_emb = encode_with_grad(positives)
            sim = q_emb @ p_emb.T / TEMPERATURE
            labels = torch.arange(len(queries), device=sim.device)
            loss = F_t.cross_entropy(sim, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"      f{fold_idx} batch {n_batches}/{total_batches}: "
                      f"loss={loss.item():.4f}", flush=True)
        print(f"    f{fold_idx} epoch {epoch}: loss={epoch_loss / max(n_batches, 1):.4f}",
              flush=True)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))
    return model


def _encode_catalog(model, meta: dict[str, dict], all_track_ids: list[str],
                    device: str) -> np.ndarray:
    """L2-normalised catalog embeddings ``(n_tracks, dim)`` (verbatim ``encode_catalog``)."""
    print(f"{_ts()}   encoding {len(all_track_ids)} tracks on {device} …", flush=True)
    model.eval()
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]
    bs = ENCODE_TRACK_BS_CUDA if device == "cuda" else ENCODE_TRACK_BS_CPU
    return model.encode(track_texts, batch_size=bs, show_progress_bar=False,
                        normalize_embeddings=True).astype(np.float32)


def _retrieve(model, queries: list[str], track_embs: np.ndarray, track_ids: list[str],
              played_lists: list[list[str]], device: str, topk: int = TOPK):
    """Per-query top-``topk`` ``[(tid, cosine), ...]`` lists, played tracks excluded.

    Verbatim ``retrieve``: encode the queries, cosine vs the cached catalog embeddings
    (both normalised), argsort descending and take the first ``topk`` not in the case's
    played set, keeping the cosine score on each pair (the score the OOF/blind contracts
    carry).
    """
    print(f"{_ts()}   encoding {len(queries)} queries on {device} …", flush=True)
    bs = ENCODE_QUERY_BS_CUDA if device == "cuda" else ENCODE_QUERY_BS_CPU
    q_embs = model.encode(queries, batch_size=bs, show_progress_bar=False,
                          normalize_embeddings=True).astype(np.float32)
    print(f"{_ts()}   retrieving top-{topk} …", flush=True)
    results = []
    for i in range(len(queries)):
        played_set = {str(t) for t in played_lists[i]} if played_lists[i] else set()
        sims = q_embs[i] @ track_embs.T
        ranked = np.argsort(-sims)
        top = []
        for j in ranked:
            tid = track_ids[j]
            if tid not in played_set:
                top.append((tid, float(sims[j])))
                if len(top) >= topk:
                    break
        results.append(top)
    return results


# ---------------- per-fold OOF ----------------

def _run_fold(fold_i: int, folds: list[np.ndarray], cases: list[dict],
              meta: dict[str, dict], all_track_ids: list[str],
              train_split_pairs: list[tuple[str, str]], device: str,
              smoke: bool = False, force: bool = False) -> list:
    """Train fold ``fold_i``, encode the catalog, retrieve its held-out cases.

    Writes ``<fold_dir>/model``, ``<fold_dir>/track_embs.npy`` and
    ``<fold_dir>/oof_lists.json`` (``{"lists": [...], "manifest": {...}, "val_idx": [...]}``)
    and returns the per-case top-``TOPK`` ``[(tid, score), ...]`` blocks for this fold.
    Per-fold artifacts are reused if present (artifact-level resume).
    """
    fold_dir = C.STRUCT_BASE_FOLD_DIRS[fold_i]
    model_dir = fold_dir / "model"
    embs_path = fold_dir / "track_embs.npy"
    lists_path = fold_dir / "oof_lists.json"
    fold_t0 = time.time()
    print(f"\n{_ts()} === FOLD {fold_i} (device={device}) ===", flush=True)

    if lists_path.exists() and not force:
        print(f"  [struct_base] fold {fold_i}: oof_lists.json present — reusing", flush=True)
        return json.load(open(lists_path))["lists"]

    n = len(cases)
    val_idx = folds[fold_i].tolist()
    held_in = set(range(n)) - set(val_idx)
    if smoke:
        # tiny but real: a small held-in training slice + a small held-out slice
        held_in = list(held_in)[:256]
        val_idx = val_idx[:64]
    train_dev_cases = [cases[j] for j in held_in]
    val_cases = [cases[j] for j in val_idx]

    # held-in dev pairs (structured query -> gold track text), gold must be in catalog
    dev_pairs = [(_query_from_dev_case(c, meta), build_track_text(c["gt"], meta))
                 for c in train_dev_cases if c["gt"] in meta]
    all_pairs = dev_pairs + train_split_pairs
    print(f"  [struct_base] fold {fold_i}: train_dev={len(dev_pairs)} "
          f"train_split={len(train_split_pairs)} total={len(all_pairs)} "
          f"val={len(val_cases)}", flush=True)

    if model_dir.exists() and (model_dir / "config.json").exists() and not force:
        print(f"  [struct_base] loading existing fold model {model_dir}", flush=True)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(model_dir), device=device)
    else:
        model = _train_fold(fold_i, all_pairs, model_dir, device,
                            epochs=1 if smoke else EPOCHS)

    if embs_path.exists() and not force:
        track_embs = np.load(embs_path)
    else:
        track_embs = _encode_catalog(model, meta, all_track_ids, device)
        np.save(embs_path, track_embs)

    val_queries = [_query_from_dev_case(c, meta) for c in val_cases]
    # ``music_turns`` is the case's already-played track ids; passed so _retrieve excludes
    # them from the held-out ranking (a track already played is never a valid next-track pick).
    val_played = [c["music_turns"] for c in val_cases]
    fold_lists = _retrieve(model, val_queries, track_embs, all_track_ids, val_played, device)

    manifest = {
        "fold": fold_i, "n_train_pairs": len(all_pairs), "n_dev_pairs": len(dev_pairs),
        "n_train_split_pairs": len(train_split_pairs), "n_val_cases": len(val_cases),
        "epochs": (1 if smoke else EPOCHS), "lr": LR, "batch_size": BATCH_SIZE,
        "temperature": TEMPERATURE, "max_length": MAX_LENGTH,
        "training_seed": TRAINING_SEED + fold_i, "sampling_seed": SAMPLING_SEED,
        "query_format": "structured", "positive_format": "text_retriever_track_text",
        "device": device, "smoke": smoke, "topk": TOPK,
        "elapsed_s": time.time() - fold_t0, "created_at": datetime.now().isoformat(),
    }
    with open(lists_path, "w") as f:
        json.dump({"lists": fold_lists, "manifest": manifest, "val_idx": val_idx}, f)

    # Diagnostic only (not used downstream): fraction of held-out cases whose gold track is
    # recalled in the top-200 — a quick recall@200 sanity signal that the fold model trained.
    hit = sum(1 for k, c in enumerate(val_cases)
              if c["gt"] in [t for t, _ in fold_lists[k][:200]])
    print(f"  [struct_base] fold {fold_i} saved ({time.time() - fold_t0:.0f}s); "
          f"val hit@200 {hit}/{len(val_cases)} ({hit / max(len(val_cases), 1):.4f})",
          flush=True)
    return fold_lists


# ---------------- blind ensemble ----------------

def _blind_ensemble(meta: dict[str, dict], all_track_ids: list[str], device: str,
                    fold_dirs: list, smoke: bool = False, blind_name: str = "blind_a") -> dict:
    """Average the fold models' cosine over the blind set; return ``{sid: [[tid, score]]}``.

    Verbatim the validated phase-3 blind ensemble: build each blind session's structured
    query, then for each fold load its cached ``track_embs.npy`` and encode the queries with
    its model, accumulate ``q @ track_embs.T`` into a running sum, divide by the number of
    folds, mask played tracks (``-inf``) and take the top-``TOPK`` per session. Under
    ``smoke`` a single fold and a handful of sessions are used so the path runs in seconds.
    """
    print(f"\n{_ts()} [struct_base] blind ensemble over {len(fold_dirs)} fold(s) …", flush=True)
    sessions = load_blind_sessions(blind_name)
    if smoke:
        sessions = sessions[:8]
    n_tracks = len(all_track_ids)

    queries, sids, played_lists = [], [], []
    for s in sessions:
        prior_user = [str(h["content"]) for h in s["history"] if h["role"] == "user"]
        queries.append(build_query_structured(s["user_query"], prior_user, s["music_turns"], meta))
        sids.append(str(s["session_id"]))
        played_lists.append(s["music_turns"])

    from sentence_transformers import SentenceTransformer
    avg_scores = np.zeros((len(queries), n_tracks), dtype=np.float64)
    for fi, fold_dir in enumerate(fold_dirs):
        print(f"{_ts()}   fold {fi}: load embeddings + encode {len(queries)} queries …",
              flush=True)
        track_embs = np.load(fold_dir / "track_embs.npy")
        assert track_embs.shape[0] == n_tracks, \
            f"fold {fi} track_embs rows {track_embs.shape[0]} != {n_tracks}"
        model = SentenceTransformer(str(fold_dir / "model"), device=device)
        q_embs = model.encode(queries, batch_size=ENCODE_BLIND_BS, show_progress_bar=False,
                              normalize_embeddings=True).astype(np.float32)
        avg_scores += (q_embs @ track_embs.T).astype(np.float64)
        del track_embs, q_embs, model
    avg_scores /= len(fold_dirs)

    print(f"{_ts()}   top-{TOPK} per session (played excluded) …", flush=True)
    blind_lists: dict[str, list] = {}
    for i, sid in enumerate(sids):
        played_set = {str(t) for t in played_lists[i]} if played_lists[i] else set()
        scores = avg_scores[i].copy()
        for ti, tid in enumerate(all_track_ids):
            if tid in played_set:
                scores[ti] = -np.inf
        topk = min(TOPK, n_tracks)
        # Two-step top-k: argpartition is O(n_tracks) and gives the top-`topk` UNORDERED, then
        # a small argsort orders just those `topk` by descending score (cheaper than a full
        # n_tracks sort). Played tracks were set to -inf above so they fall out of the top-k.
        top_idx = np.argpartition(-scores, topk - 1)[:topk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        blind_lists[sid] = [(all_track_ids[j], float(scores[j])) for j in top_idx]
    return blind_lists


# ---------------- build ----------------

def build(force: bool = False, smoke: bool = False) -> None:
    """Train struct_base: 5-fold OOF dev lists -> ``C.STRUCT_BASE_OOF``, then the blind
    ensemble -> ``C.STRUCT_BASE_BLIND_LISTS``.

    No-op when both outputs already exist unless ``force=True``. Per-fold artifacts
    (model + catalog embeddings + ``oof_lists.json``) are written under
    ``C.STRUCT_BASE_FOLD_DIRS[fold]`` and reused if present, so an interrupted multi-hour run
    continues fold-by-fold.

    ``smoke=True`` runs a genuinely small but complete pass: ONE fold trained on a tiny
    held-in slice for one short epoch (plus a tiny train-split pool), then that single fold
    is "ensembled" over a handful of blind sessions — exercising the whole train + encode +
    retrieve + OOF-write + ensemble + blind-write path quickly (no 5-fold, no full training
    set, no full blind set). It still writes plausibly-shaped ``C.STRUCT_BASE_OOF`` (populated
    only for the smoke fold's held-out cases) and ``C.STRUCT_BASE_BLIND_LISTS`` (the few smoke
    sessions).
    """
    oof_path = C.STRUCT_BASE_OOF
    blind_path = C.STRUCT_BASE_BLIND_LISTS

    if oof_path.exists() and blind_path.exists() and not force:
        print("  [skip] struct_base OOF + blind ensemble present"); return

    device = _device()
    t0 = time.time()
    print(f"{_ts()} [struct_base] device={device}; loading dev payload + catalog …", flush=True)

    import pickle
    with open(C.DEV_PAYLOAD, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    dev_session_ids = set(sessions)

    meta, all_track_ids = _load_catalog()
    print(f"  [struct_base] catalog: {len(all_track_ids)} tracks", flush=True)

    folds = grouped_session_folds(sessions, seed=FOLD_SEED, k=N_FOLDS)
    fold_range = [0] if smoke else list(range(N_FOLDS))

    print(f"{_ts()} [struct_base] sampling train-split (seed={SAMPLING_SEED}) …", flush=True)
    train_split_pairs = _build_train_split_sample(meta, dev_session_ids, smoke=smoke)

    # ===== 5-fold OOF (one tiny fold under smoke) =====
    # Pre-size to n_cases so each fold can drop its held-out block at the case's GLOBAL dev
    # index; the union of the five folds' held-out blocks covers every case exactly once.
    all_oof_lists: list = [[] for _ in range(n)]
    for fold_i in fold_range:
        fold_lists = _run_fold(fold_i, folds, cases, meta, all_track_ids, train_split_pairs,
                              device, smoke=smoke, force=force)
        # stitch this fold's block back to its global case indices
        # ``fold_lists`` is indexed 0..len(val_cases)-1 (fold-local order); ``val_idx`` maps
        # each local position to its global dev-payload index, so the OOF array stays aligned
        # to ``C.DEV_PAYLOAD["cases"]`` order — the contract case_features._load_struct_base_oof reads.
        val_idx = folds[fold_i].tolist()
        if smoke:
            val_idx = val_idx[:64]
        for k_local, k_global in enumerate(val_idx):
            all_oof_lists[k_global] = fold_lists[k_local]

    if not smoke:
        assert all(len(x) > 0 for x in all_oof_lists), "missing OOF lists for some cases"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    with open(oof_path, "w") as f:
        json.dump({
            "lists": all_oof_lists, "n_cases": n,
            "experiment": "struct_base phase-3 full (structured query + 20k train-split), 5-fold OOF",
            "smoke": smoke, "created_at": datetime.now().isoformat(),
        }, f)
    covered = [i for i in range(n) if all_oof_lists[i]]
    hit200 = sum(1 for i in covered if cases[i]["gt"] in [t for t, _ in all_oof_lists[i][:200]])
    print(f"\n{_ts()} [struct_base] OOF -> {oof_path} "
          f"(covered {len(covered)}/{n}, hit@200_rate={hit200 / max(len(covered), 1):.4f})",
          flush=True)

    # ===== blind ensemble =====
    fold_dirs = [C.STRUCT_BASE_FOLD_DIRS[fi] for fi in fold_range]
    blind_lists = _blind_ensemble(meta, all_track_ids, device, fold_dirs, smoke=smoke)
    blind_path.parent.mkdir(parents=True, exist_ok=True)
    with open(blind_path, "w") as f:
        json.dump({
            "lists": blind_lists,
            "manifest": {
                "experiment": "struct_base phase-3 ensemble blind retrieval",
                "blind_name": "blind_a", "blind_dataset": C.BLIND_DATASETS["blind_a"],
                "fold_dirs": [str(d) for d in fold_dirs], "n_blind_sessions": len(blind_lists),
                "topk": TOPK, "method": "average cosine across fold struct_base models",
                "smoke": smoke, "elapsed_s": time.time() - t0,
                "created_at": datetime.now().isoformat(),
            },
        }, f)
    print(f"{_ts()} [struct_base] blind ensemble -> {blind_path} "
          f"({len(blind_lists)} sessions); done in {time.time() - t0:.0f}s", flush=True)
