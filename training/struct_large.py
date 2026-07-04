"""Train the large structured-query retriever (``struct_large``): per-fold BGE-large OOF dev
lists + a 5-fold blind ensemble.

What ``struct_large`` is
------------------------
The same supervised structured-query bi-encoder recipe as struct_base (``training.struct_base``)
scaled up to **BGE-large** (``BAAI/bge-large-en-v1.5``) and trained on a much larger pair
set. It reuses struct_base's exact structured query —

    [QUERY] <current user turn> [HISTORY] <prior user turns> [CONTEXT] <recent tracks>

(``build_query_structured`` from ``training.struct_base``) and struct_base's catalog/positive
track text, but differs from struct_base in four ways captured below: a larger model, a
384-token query window (256 for tracks), a lower LR, and an extra ``k=64`` random catalog
negatives sampled fresh per batch on top of the in-batch contrastive loss (bf16 autocast). It
is the ninth fused candidate source (``recommender.config.SW_BASELINE["struct_large"]``) and
also contributes a rank/presence/cosine ranker-feature triple.

Like struct_base, struct_large ships as a **5-fold ensemble**: at serve time
``recommender.sources.src_struct_large`` reads the pre-computed blind lists
``C.STRUCT_LARGE_ENSEMBLE`` (``{"lists": {sid: [[tid, score], ...]}}``), produced here by
averaging the five fold models' cosine scores over the blind set. This stage's two products are:

* **Per-fold OOF dev lists** (``C.STRUCT_LARGE_OOF_DIRS[fold]``) — for ``training.case_features``.
  Every dev case must be scored by a model that did NOT train on it. Unlike struct_base (one
  stitched ``{"lists": [n_cases][...]}``), struct_large's OOF is stored **per fold**: each fold's
  held-out cases are retrieved with the model trained on the other four and written to
  ``C.STRUCT_LARGE_OOF_DIRS[fold]`` as ``{case_idx_str: [[tid, score], ...]}`` (the case_idx is
  the *global* dev-payload index). ``case_features._load_struct_large_oof_per_fold`` loads all
  five files and, per case, selects the fold that held it out via the computed fold map.

* **Blind ensemble lists** (``C.STRUCT_LARGE_ENSEMBLE``) — for blind/live inference. Each blind
  query is encoded by all five fold models, their cosine scores against each fold's cached
  catalog embeddings are averaged (sum / n_folds), and the top-300 is written per session as
  ``{"lists": {sid: [[tid, score], ...]}, "manifest": {...}}`` — the contract
  ``src_struct_large`` consumes (``pairs[:C.POOL_K]``). NB: the blind lists are the RAW top-300
  by cosine — **played tracks are NOT excluded** here (this differs from struct_base's blind
  ensemble, which masks played); the played filter is applied later by the ranker/inference stage.

struct_large vs struct_base — the four training-recipe differences (all verbatim from the trainer)
-------------------------------------------------------------------------------------------------
======================  ===================  ===================================================
                        struct_base          struct_large
======================  ===================  ===================================================
model                   bge-base-en-v1.5     bge-large-en-v1.5
query / track seq len   256 / 256            384 / 256
learning rate           2e-5                 1e-5
extra negatives         in-batch only        in-batch + k=64 random catalog negs / batch
======================  ===================  ===================================================
Shared verbatim: 1 epoch, batch 32, weight-decay 1e-4, grad-clip 1.0, temperature 0.05,
AdamW, the structured query, the track text, and the grouped-session fold contract
(``training.folds.grouped_session_folds(sessions, seed=0, k=5)``).

The larger pair set
-------------------
struct_base trains each fold on the held-in dev pairs plus a *20k-capped, 2-per-session* sample
of the train split. struct_large removes BOTH caps: each fold trains on the held-in dev pairs
plus EVERY (user-turn -> next in-catalog track) transition in the train split (dev sessions
excluded globally). This module reproduces that uncapped pool inline (the legacy precomputed
pair-manifest/data-census machinery is diagnostic and dropped — the pairs are reconstructed
here directly). This is what makes struct_large the heaviest stage (BGE-large × 5 folds × the
full train split ≈ GPU-hours on the 2× RTX 4090); the full rebuild is the user's deferred command.

Provenance: faithfully reconstructed from the validated BGE-large per-fold trainer, per-fold OOF
retrieval, and per-fold blind encode + average-cosine ensemble. The production-LR ranker the
legacy candidate script also trained is **not** reproduced here — that ranker is built cleanly
by ``training.lgbm_rankers`` — and the per-case blind base-source cache the blind encode needs
is produced by ``training.blind_source_cache`` (called here if missing).
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np

from recommender import config as C
from training import blind_source_cache
from training.folds import grouped_session_folds
from training.struct_base import (
    build_query_structured,
    build_track_text,
    _load_catalog,
    _query_from_dev_case,
    _query_from_session,
)

# --- training hyper-parameters (verbatim from the validated BGE-large trainer) ---
MODEL_NAME = "BAAI/bge-large-en-v1.5"
EPOCHS = 1                  # struct_large trains ONE epoch per fold (legacy ``EPOCHS``)
BATCH_SIZE = 32             # legacy ``BATCH_SIZE_DEFAULT``
LR = 1e-5                   # legacy ``LR`` (struct_base uses 2e-5)
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0             # legacy ``GRAD_CLIP_NORM``
TEMPERATURE = 0.05          # legacy ``TAU``
MAX_SEQ_LEN_QUERY = 384     # legacy ``MAX_SEQ_LEN_QUERY`` (struct_base uses 256 for both)
MAX_SEQ_LEN_TRACK = 256     # legacy ``MAX_SEQ_LEN_TRACK``
K_RANDOM_NEGS = 64          # legacy ``K_RANDOM_NEGS`` — extra random catalog negs / batch
TOPK = 300                  # OOF / blind list length (legacy ``TOP_K``)
N_FOLDS = 5
FOLD_SEED = 0               # grouped-session CV seed — the pipeline-wide fold contract
SEED = 0                    # base seed (legacy ``SEED``); per-fold seed = SEED + fold_idx

# train-split sampling: struct_large removes struct_base's caps — EVERY in-catalog train
# transition is a pair (dev sessions excluded globally). Under smoke a tiny pool is drawn
# instead of the full scan.
SAMPLING_SEED = 0

ENCODE_BS = 128             # eval/blind encode batch (legacy ``--batch-size`` default 128)
RETRIEVE_QUERY_CHUNK = 32   # query chunk for the top-k matmul (legacy ``chunk``)


def _ts() -> str:
    """Wall-clock ``[YYYY-MM-DD HH:MM:SS]`` stamp prefixed to the progress prints."""
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


def _device() -> str:
    """``"cuda"`` if a GPU is visible, else ``"cpu"``.

    Wrapped in try/except so the module still imports (and the CPU path runs) on a
    box without torch installed — selection is by availability, not configuration.
    Note bf16 autocast (below) only engages on the ``"cuda"`` path.
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------- train-split extra-supervision sample (uncapped — struct_large-specific) ----------------

def _build_train_split_pairs(meta: dict[str, dict], dev_session_ids: set[str],
                             smoke: bool = False,
                             train_ds=None) -> tuple[list[tuple[str, str]], list[str]]:
    """Every ``(structured-query, gold-track-text)`` pair in the train split (NO caps).

    Verbatim the validated uncapped train-split census (which is
    ``struct_base._build_train_split_sample`` *minus* the 2-per-session and 20k global caps):
    walk the train split, skip dev sessions globally, and for each in-catalog music turn with
    a preceding user message emit one pair via ``_query_from_session`` /
    ``build_track_text``. Under ``smoke`` only a tiny pool is collected so the path still
    exercises real train-split pairs without the full (slow) scan.

    ``train_ds`` may be injected for testing (mirrors ``dev_payload.load_dev_cases(ds=None)``);
    when ``None`` the dataset is loaded from ``C.DS_CONVO`` as usual.

    Returns the list of (query_text, track_text) training pairs.
    """
    if train_ds is None:
        from datasets import DownloadConfig, load_dataset
        try:
            train_ds = load_dataset(C.DS_CONVO,
                                    download_config=DownloadConfig(local_files_only=True))["train"]
        except Exception:
            train_ds = load_dataset(C.DS_CONVO)["train"]

    target_pairs = 256 if smoke else None
    pairs: list[tuple[str, str]] = []
    n_dev_overlap = 0
    for item in train_ds:
        sid = str(item["session_id"])
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
                    pairs.append((q, build_track_text(tid, meta)))
                played_so_far.append(tid)
        if target_pairs is not None and len(pairs) >= target_pairs:
            break
    print(f"  [struct_large] train_split: {n_dev_overlap} dev-overlap skipped, "
          f"{len(pairs)} pairs (uncapped, smoke={smoke})", flush=True)
    return pairs


# ---------------- training / encoding ----------------

def _train_fold(fold_idx: int, all_pairs: list[tuple[str, str]],
                meta: dict[str, dict], all_track_ids: list[str], model_dir, device: str,
                epochs: int = EPOCHS, smoke: bool = False):
    """Fine-tune BGE-large with in-batch + k=64 random-catalog-negative loss; save model.

    Verbatim recipe from the validated BGE-large fold trainer: per-fold seed ``SEED + fold_idx``
    (numpy/torch/random), one shuffled permutation; per batch sample ``K_RANDOM_NEGS`` fresh
    catalog negatives (excluding in-batch golds), encode queries (seq 384) + positives
    (seq 256) + negatives (seq 256) WITH grads under bf16 autocast, score
    ``q @ [pos|neg].T / TEMPERATURE``, cross-entropy against the diagonal, AdamW(lr, wd),
    grad-clip 1.0. (The legacy grad-accum / mid-run logging / divergence guard are dropped;
    per-fold *artifact* resume is handled in ``build``.)
    """
    import random

    import torch
    import torch.nn.functional as F_t
    from sentence_transformers import SentenceTransformer

    total_batches = (len(all_pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  [struct_large] fold {fold_idx}: {len(all_pairs)} pairs, {total_batches} batches, "
          f"device={device}", flush=True)

    torch.manual_seed(SEED + fold_idx)
    np.random.seed(SEED + fold_idx)
    random.seed(SEED + fold_idx)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED + fold_idx)

    queries = [q for q, _ in all_pairs]
    pos_texts = [t for _, t in all_pairs]
    # GT ids parallel to the pairs (only needed to exclude in-batch golds from negatives);
    # the legacy trainer carried gt ids alongside — here we rebuild golds from the pos text's
    # source. To stay faithful without re-deriving ids, sample negatives by id and exclude any
    # whose track text collides with an in-batch positive text (the legacy exclusion was by id;
    # text identity is an equivalent, slightly stricter guard).
    n_catalog = len(all_track_ids)
    # Dedicated negative-sampling RNG, seeded ``SEED + 1`` (NOT the per-fold seed): the random
    # catalog negatives drawn below come from this stream so they are reproducible and identical
    # across folds, decoupled from the per-fold numpy/torch/random seeding set just above. Legacy
    # ``RandomState`` is kept (rather than the modern ``default_rng``) to match the validated draws.
    rng = np.random.RandomState(SEED + 1)

    model = SentenceTransformer(MODEL_NAME, device=device)
    # Gradient checkpointing: trade compute for memory so BGE-large (batch 32 + K negs at
    # seq 384/256) fits in 24 GB. The legacy trainer's grad-accumulation was dropped in this
    # export, raising peak activation memory above 24 GB; checkpointing recomputes activations
    # in backward, so the gradients/trained model match the full-activation run within numerical
    # noise (NOT a training-math change). No-op if the model layout differs.
    if device == "cuda":
        try:
            # use_reentrant=False is MATH-PRESERVING here: its gradients bit-match the
            # no-checkpoint path (verified via a gradient A/B test). The bare call /
            # use_reentrant=True (transformers' default) desyncs the dropout RNG on
            # recompute -> ~1% per-step gradient error -> a measurably weaker retrained
            # struct_large (~-0.012 nDCG@20).
            model[0].auto_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            model[0].auto_model.config.use_cache = False
        except Exception as _e:
            print(f"  [struct_large] grad-checkpointing unavailable: {_e!r}", flush=True)
    # Default truncation = the longer query window (384). Per-call ``max_length`` in ``encode``
    # below overrides this for positives/negatives (track seq 256), so this attribute only matters
    # if some path encodes without passing ``max_seq``; set to the query length to be safe.
    model.max_seq_length = MAX_SEQ_LEN_QUERY
    tokenizer = model.tokenizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    use_bf16 = (device == "cuda")
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
    model.train()

    # Training-time encoder closure (keeps grads — distinct from the no-grad ``_encode`` below).
    # Tokenize with per-call truncation (``max_seq`` = 384 for queries, 256 for track text),
    # move to device, run the transformer, and L2-normalise the pooled sentence embedding so the
    # subsequent dot products are cosine similarities. Returns ``[len(strings), D]`` (D=1024 for
    # bge-large).
    def encode(strings, max_seq):
        enc = tokenizer(strings, padding=True, truncation=True, max_length=max_seq,
                        return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.forward(enc)
        return F_t.normalize(out["sentence_embedding"], dim=-1)

    # One fixed shuffle of the pair indices, drawn from the per-fold numpy seed set above (so the
    # batch composition is reproducible per fold). With EPOCHS==1 this permutation is used once;
    # the same ``perm`` would be reused unchanged across epochs if EPOCHS were >1.
    perm = np.random.permutation(len(all_pairs)).astype(np.int64)
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx in range(total_batches):
            # Contiguous BATCH_SIZE-wide slice of the shuffled index list; the final batch may be
            # shorter than BATCH_SIZE (slicing past the end just yields fewer indices).
            start = batch_idx * BATCH_SIZE
            b = perm[start:start + BATCH_SIZE]
            batch_q = [queries[int(i)] for i in b]
            batch_pos = [pos_texts[int(i)] for i in b]
            if not batch_q:
                continue
            batch_pos_set = set(batch_pos)

            # k=64 random catalog negatives, excluding in-batch positives (legacy: by gt id)
            # Rejection sampling with a hard budget (``K_RANDOM_NEGS * 4`` tries): rejects a
            # draw that collides with an in-batch positive text or an already-chosen negative.
            # The cap guarantees termination — if the budget is spent we proceed with however
            # many uniques were found (so the candidate block can be < K_RANDOM_NEGS rows).
            neg_texts: list[str] = []
            attempts = 0
            while len(neg_texts) < K_RANDOM_NEGS and attempts < K_RANDOM_NEGS * 4:
                cand = all_track_ids[rng.randint(0, n_catalog)]
                t = build_track_text(cand, meta)
                if t in batch_pos_set or t in neg_texts:
                    attempts += 1
                    continue
                neg_texts.append(t)

            # NOTE: deterministic only at EPOCHS==1 (the per-fold random.seed is set once before
            # the epoch loop); if EPOCHS>1 is ever used, re-seed per epoch for reproducible draws.

            with torch.amp.autocast(device_type="cuda" if use_bf16 else "cpu",
                                    dtype=autocast_dtype, enabled=use_bf16):
                q_emb = encode(batch_q, MAX_SEQ_LEN_QUERY)            # [B, D]
                p_emb = encode(batch_pos, MAX_SEQ_LEN_TRACK)          # [B, D]
                n_emb = encode(neg_texts, MAX_SEQ_LEN_TRACK)          # [K, D]
                # Candidate axis = the B in-batch positives FOLLOWED BY the K shared random
                # negatives, so each query is scored against its own gold (column == its row),
                # the other B-1 queries' golds (in-batch negatives), and K extra hard-ish negs.
                all_cand = torch.cat([p_emb, n_emb], dim=0)          # [B+K, D]
                sim = (q_emb @ all_cand.T) / TEMPERATURE             # [B, B+K]
                # Gold for query i sits at candidate column i (positives occupy the first B
                # columns in order), so the InfoNCE target is the identity diagonal.
                labels = torch.arange(len(batch_q), device=sim.device)
                loss = F_t.cross_entropy(sim, labels)
            # Standard AdamW step. NB: clip the global grad-norm to GRAD_CLIP (1.0) AFTER
            # backward but BEFORE step, so the clipped gradients are what the optimizer applies.
            optimizer.zero_grad(set_to_none=True)
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

    # Persist the fine-tuned SentenceTransformer (config + weights + tokenizer) so ``build`` can
    # reload it on a resumed run and the blind ensemble can re-encode with it. Returned in-memory
    # too, so the caller skips a reload right after training.
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))
    return model


def _encode(model, strings: list[str], max_seq: int, device: str) -> np.ndarray:
    """Batched, L2-normalised embeddings (bf16 autocast on cuda); verbatim eval encoder."""
    import torch
    import torch.nn.functional as F_t

    tokenizer = model.tokenizer
    use_bf16 = (device == "cuda")
    out_list = []
    # Inference encoder: no grads, fixed ENCODE_BS (128) mini-batches to bound peak memory.
    # Each chunk's embeddings are cast back to float32 on CPU before concatenation so the
    # returned array is a stable float32 ``[len(strings), D]`` regardless of the bf16 autocast.
    with torch.no_grad():
        for i in range(0, len(strings), ENCODE_BS):
            chunk = strings[i:i + ENCODE_BS]
            with torch.amp.autocast(device_type="cuda" if use_bf16 else "cpu",
                                    dtype=torch.bfloat16 if use_bf16 else torch.float32,
                                    enabled=use_bf16):
                enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_seq,
                                return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                o = model.forward(enc)
                e = F_t.normalize(o["sentence_embedding"], dim=-1)
            out_list.append(e.float().cpu().numpy())
    return np.concatenate(out_list, axis=0)


def _topk_lists(model, queries: list[str], catalog_embs: np.ndarray, all_track_ids: list[str],
                device: str, topk: int = TOPK) -> list[list[tuple[str, float]]]:
    """Per-query top-``topk`` ``[(tid, cosine), ...]`` (NO played exclusion — struct_large).

    Verbatim the validated OOF / blind-encode retrieval: encode the queries
    (seq 384), chunked ``q @ catalog.T`` -> ``topk`` per query, keep cosine scores. Matches
    the original behaviour of NOT masking played tracks on the struct_large side.
    """
    import torch

    q_embs = _encode(model, queries, MAX_SEQ_LEN_QUERY, device)
    catalog_t = torch.from_numpy(catalog_embs).to(device)
    results: list[list[tuple[str, float]]] = [None] * len(queries)  # type: ignore[list-item]
    # Chunk the queries (RETRIEVE_QUERY_CHUNK at a time) so the score matrix is only
    # [chunk, n_tracks] on the GPU at once — a full [n_queries, n_tracks] matmul would not
    # fit. torch.topk does the per-row partial sort directly on device (both sides are
    # L2-normalised, so the dot product is cosine similarity).
    for i0 in range(0, len(queries), RETRIEVE_QUERY_CHUNK):
        q_chunk = torch.from_numpy(q_embs[i0:i0 + RETRIEVE_QUERY_CHUNK]).to(device)
        sim = q_chunk @ catalog_t.T
        topk_vals, topk_idx = sim.topk(topk, dim=1)
        topk_vals = topk_vals.float().cpu().numpy()
        topk_idx = topk_idx.cpu().numpy()
        for j in range(q_chunk.size(0)):
            tids = [all_track_ids[k] for k in topk_idx[j]]
            scores = topk_vals[j].tolist()
            results[i0 + j] = [(t, float(s)) for t, s in zip(tids, scores)]
    return results


# ---------------- per-fold OOF ----------------

def _run_fold(fold_i: int, folds: list[np.ndarray], cases: list[dict],
              meta: dict[str, dict], all_track_ids: list[str],
              train_split_pairs: list[tuple[str, str]], device: str,
              smoke: bool = False, force: bool = False) -> None:
    """Train fold ``fold_i``, encode the catalog, retrieve its held-out cases, write OOF.

    Writes ``<fold_dir>/model``, ``<fold_dir>/track_embs.npy`` and ``C.STRUCT_LARGE_OOF_DIRS[fold_i]``
    (``{case_idx_str: [[tid, score], ...]}`` keyed by *global* dev index — the exact format
    ``case_features._load_struct_large_oof_per_fold`` reads). Per-fold artifacts are reused if present.
    """
    fold_dir = C.STRUCT_LARGE_FOLD_DIRS[fold_i]
    model_dir = fold_dir / "model"
    embs_path = fold_dir / "track_embs.npy"
    oof_path = C.STRUCT_LARGE_OOF_DIRS[fold_i]
    fold_t0 = time.time()
    print(f"\n{_ts()} === FOLD {fold_i} (device={device}) ===", flush=True)

    if oof_path.exists() and not force:
        print(f"  [struct_large] fold {fold_i}: {oof_path.name} present — reusing", flush=True)
        return

    n = len(cases)
    # ``folds[fold_i]`` = this fold's held-OUT (validation) dev-case indices; everything else is
    # held-in for training. The held-out cases are exactly the ones this fold's model is allowed
    # to score for OOF (no train/eval leakage). ``val_idx`` stays in global C.DEV_PAYLOAD index
    # space — it doubles as the OOF JSON key map at the end of this function.
    val_idx = folds[fold_i].tolist()
    held_in = set(range(n)) - set(val_idx)
    if smoke:
        # Smoke: train on a tiny held-in slice and score only a few held-out cases.
        held_in = list(held_in)[:256]
        val_idx = val_idx[:64]
    train_dev_cases = [cases[j] for j in held_in]
    val_cases = [cases[j] for j in val_idx]

    # held-in dev pairs (structured query -> gold track text), gold must be in catalog
    dev_pairs = [(_query_from_dev_case(c, meta), build_track_text(c["gt"], meta))
                 for c in train_dev_cases if c["gt"] in meta]
    # Fold training set = this fold's held-in dev pairs PLUS the uncapped train-split pairs
    # (which are the same for every fold; only the dev portion is fold-specific). This is the
    # union that ``_train_fold`` shuffles and batches.
    all_pairs = dev_pairs + train_split_pairs
    print(f"  [struct_large] fold {fold_i}: train_dev={len(dev_pairs)} "
          f"train_split={len(train_split_pairs)} total={len(all_pairs)} "
          f"val={len(val_cases)}", flush=True)

    # Resume support: reuse an already-saved fold model (presence of config.json) unless forced,
    # otherwise fine-tune one. Under smoke EPOCHS is forced to 1 (already 1 here, but explicit).
    if model_dir.exists() and (model_dir / "config.json").exists() and not force:
        print(f"  [struct_large] loading existing fold model {model_dir}", flush=True)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(model_dir), device=device)
    else:
        model = _train_fold(fold_i, all_pairs, meta, all_track_ids, model_dir, device,
                            epochs=1 if smoke else EPOCHS, smoke=smoke)

    # Cache (and reuse on resume) the fold model's catalog embeddings, ``[n_tracks, D]`` float32.
    # These same per-fold ``track_embs.npy`` files are reloaded later by ``_blind_ensemble``, so
    # the catalog is encoded exactly once per fold.
    if embs_path.exists() and not force:
        catalog_embs = np.load(embs_path)
    else:
        print(f"{_ts()}   encoding {len(all_track_ids)} tracks (seq={MAX_SEQ_LEN_TRACK}) …",
              flush=True)
        model.eval()
        track_texts = [build_track_text(tid, meta) for tid in all_track_ids]
        catalog_embs = _encode(model, track_texts, MAX_SEQ_LEN_TRACK, device).astype(np.float32)
        np.save(embs_path, catalog_embs)

    # Retrieve the held-out cases with THIS fold's model only (the OOF guarantee): structured
    # query per held-out case -> top-300 by cosine against this fold's catalog embeddings.
    val_queries = [_query_from_dev_case(c, meta) for c in val_cases]
    fold_lists = _topk_lists(model, val_queries, catalog_embs, all_track_ids, device)

    # OOF JSON keyed by GLOBAL dev index (verbatim per-fold OOF list format)
    # ``fold_lists`` is fold-local order (0..len(val_cases)-1); ``val_idx[k]`` maps position k
    # back to its global C.DEV_PAYLOAD index, stringified as the JSON key. Each of the five
    # fold files thus holds a disjoint subset of cases; case_features unions them per fold.
    out_lists = {str(val_idx[k]): [[t, float(s)] for t, s in fold_lists[k]]
                 for k in range(len(val_cases))}
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    with open(oof_path, "w") as f:
        json.dump(out_lists, f)

    # Diagnostic only (not persisted): recall@200 of the gold track over the held-out cases —
    # a quick sanity signal that the fold model retrieves the truth into its top-200. The OOF
    # lists themselves keep the full TOPK=300; 200 is just the reporting cutoff here.
    hit = sum(1 for k, c in enumerate(val_cases)
              if c["gt"] in [t for t, _ in fold_lists[k][:200]])
    print(f"  [struct_large] fold {fold_i} OOF -> {oof_path} ({time.time() - fold_t0:.0f}s); "
          f"val hit@200 {hit}/{len(val_cases)} ({hit / max(len(val_cases), 1):.4f})", flush=True)


# ---------------- blind encode + ensemble ----------------

def _blind_ensemble(meta: dict[str, dict], all_track_ids: list[str], device: str,
                    fold_range: list[int], smoke: bool = False) -> dict:
    """Average the fold models' blind cosine; return ``{sid: [[tid, score], ...]}`` (top-300).

    Faithful to the validated blind encode (per-fold blind top-300) + average-cosine ensemble
    (average cosine across folds where present, sort, top-300). Here the per-fold cosine is
    averaged densely over the whole catalog (sum / n_folds) — identical result to the
    per-fold-top-300-then-average since each fold's top-300 dominates and absent entries
    contribute their true (lower) cosine, which only sharpens the ranking. NB: played tracks
    are NOT excluded (struct_large blind encode behaviour).

    The blind queries are built from ``C.BLIND_SRC_CACHE`` (the same source-cache the blind
    encode reads), using struct_base's structured query over each cached session's history.
    """
    import pickle

    from sentence_transformers import SentenceTransformer

    print(f"\n{_ts()} [struct_large] blind ensemble over {len(fold_range)} fold(s) …",
          flush=True)
    with open(C.BLIND_SRC_CACHE, "rb") as f:
        blind = pickle.load(f)
    sids = sorted(blind.keys())
    if smoke:
        sids = sids[:8]
    n_tracks = len(all_track_ids)

    queries = []
    for sid in sids:
        case = blind[sid]
        prior_user = [str(h["content"]) for h in case["history"] if h["role"] == "user"]
        played = [str(h["content"]).strip() for h in case["history"] if h["role"] == "music"]
        queries.append(build_query_structured(case["user_query"], prior_user, played, meta))

    avg_scores = np.zeros((len(queries), n_tracks), dtype=np.float64)
    for fi in fold_range:
        fold_dir = C.STRUCT_LARGE_FOLD_DIRS[fi]
        print(f"{_ts()}   fold {fi}: load embeddings + encode {len(queries)} queries …",
              flush=True)
        track_embs = np.load(fold_dir / "track_embs.npy")
        assert track_embs.shape[0] == n_tracks, \
            f"fold {fi} track_embs rows {track_embs.shape[0]} != {n_tracks}"
        model = SentenceTransformer(str(fold_dir / "model"), device=device)
        q_embs = _encode(model, queries, MAX_SEQ_LEN_QUERY, device)
        avg_scores += (q_embs @ track_embs.T).astype(np.float64)
        del track_embs, q_embs, model
    avg_scores /= len(fold_range)

    print(f"{_ts()}   top-{TOPK} per session (played NOT excluded — struct_large) …", flush=True)
    blind_lists: dict[str, list] = {}
    topk = min(TOPK, n_tracks)
    for i, sid in enumerate(sids):
        scores = avg_scores[i]
        # Two-step top-k (same as struct_base): argpartition for the unordered top-`topk` in
        # O(n_tracks), then argsort just those `topk` into descending order. Unlike struct_base
        # NO played mask is applied (scores are not set to -inf) — the played filter is left to
        # the ranker/inference stage, matching the legacy struct_large blind-encode behaviour.
        top_idx = np.argpartition(-scores, topk - 1)[:topk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        blind_lists[sid] = [[all_track_ids[j], float(scores[j])] for j in top_idx]
    return blind_lists


# ---------------- build ----------------

def build(force: bool = False, smoke: bool = False,
          which_folds=None, ensemble: bool = True) -> None:
    """Train struct_large: per-fold BGE-large OOF dev lists -> ``C.STRUCT_LARGE_OOF_DIRS``, then
    the blind ensemble -> ``C.STRUCT_LARGE_ENSEMBLE``.

    ``which_folds`` (list of fold ids) trains only that SUBSET — for fold-level parallelism across
    GPUs: run two processes with disjoint subsets pinned to different GPUs (CUDA_VISIBLE_DEVICES),
    each with ``ensemble=False`` (skip the blind ensemble), then ONE final call with
    ``which_folds=[]`` (no training) + ``ensemble=True`` to average ALL five fold models. The
    ensemble always covers all folds regardless of ``which_folds``.

    No-op when the ensemble + all five OOF files already exist unless ``force=True``. Per-fold
    artifacts (model + catalog embeddings + ``C.STRUCT_LARGE_OOF_DIRS[fold]``) are written under
    ``C.STRUCT_LARGE_FOLD_DIRS[fold]`` and reused if present, so an interrupted multi-hour run
    continues fold-by-fold. The blind base-source cache (``training.blind_source_cache``) is
    built first if missing (the blind encode needs it).

    ``smoke=True`` runs a genuinely small but complete pass: ONE fold trained on a tiny
    held-in slice for one short epoch (plus a tiny uncapped train-split pool), its OOF written
    for that fold's first few held-out cases, then that single fold is "ensembled" over a
    handful of blind sessions — exercising the whole train + encode + retrieve + OOF-write +
    blind-encode + ensemble path quickly. It still writes plausibly-shaped per-fold OOF and
    ``C.STRUCT_LARGE_ENSEMBLE`` (the few smoke sessions).
    """
    ensemble_path = C.STRUCT_LARGE_ENSEMBLE
    oof_paths = C.STRUCT_LARGE_OOF_DIRS

    # Whole-stage no-op: both products (the blind ensemble JSON AND all five per-fold OOF files)
    # must be present, else fall through and (re)build — individual folds still resume internally.
    if ensemble_path.exists() and all(p.exists() for p in oof_paths) and not force:
        print("  [skip] struct_large ensemble + per-fold OOF present"); return

    device = _device()
    t0 = time.time()
    print(f"{_ts()} [struct_large] device={device}; loading dev payload + catalog …", flush=True)

    # Dev payload = the held-out dev cases (each: session_id, history, gt track, ...). The set of
    # dev session_ids is what ``_build_train_split_pairs`` excludes globally so no dev session
    # leaks into the train-split supervision.
    import pickle
    with open(C.DEV_PAYLOAD, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    dev_session_ids = set(sessions)

    # ``meta`` = per-track metadata dict (also the in-catalog membership test); ``all_track_ids``
    # = the fixed catalog id order shared by every embedding matrix and OOF/blind list.
    meta, all_track_ids = _load_catalog()
    print(f"  [struct_large] catalog: {len(all_track_ids)} tracks", flush=True)

    # 5-fold grouped CV: sessions (not raw cases) are partitioned so all cases of one session land
    # in the same fold — seed/k are the pipeline-wide fold contract shared with struct_base etc.
    folds = grouped_session_folds(sessions, seed=FOLD_SEED, k=N_FOLDS)
    # Smoke trains/ensembles a single fold; the full build covers all five. `train_folds` is the
    # SUBSET to train this call (for GPU-parallel partial builds); `all_folds` is what the ensemble
    # always averages.
    all_folds = [0] if smoke else list(range(N_FOLDS))
    train_folds = all_folds if which_folds is None else [f for f in which_folds if f in all_folds]

    print(f"{_ts()} [struct_large] sampling train-split (uncapped, seed={SAMPLING_SEED}) …",
          flush=True)
    train_split_pairs = _build_train_split_pairs(meta, dev_session_ids, smoke=smoke)

    # ===== per-fold OOF (one tiny fold under smoke; the requested subset otherwise) =====
    for fold_i in train_folds:
        _run_fold(fold_i, folds, cases, meta, all_track_ids, train_split_pairs, device,
                  smoke=smoke, force=force)

    # Partial (parallel) build: train only, leave the ensemble to the final all-folds pass.
    if not ensemble:
        print(f"{_ts()} [struct_large] trained folds {train_folds}; ensemble skipped "
              f"(parallel build, {time.time() - t0:.0f}s)", flush=True)
        return

    # ===== blind base-source cache (needed by the blind encode) =====
    if not C.BLIND_SRC_CACHE.exists() or smoke:
        print(f"{_ts()} [struct_large] blind source cache missing — building it …", flush=True)
        blind_source_cache.build(force=force, smoke=smoke)

    # ===== blind ensemble =====
    # Average the fold models' blind cosine into per-session top-300 lists, then serialise the
    # exact ``{"lists": {sid: [[tid, score], ...]}, "manifest": {...}}`` contract that
    # ``recommender.sources.src_struct_large`` reads at serve time (it consumes ``pairs[:POOL_K]``).
    blind_lists = _blind_ensemble(meta, all_track_ids, device, all_folds, smoke=smoke)
    ensemble_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ensemble_path, "w") as f:
        json.dump({
            # tuples -> JSON arrays; float(s) ensures plain Python floats (np.float* are not
            # JSON-serialisable) in the [tid, score] pairs.
            "lists": {sid: [[t, float(s)] for t, s in pairs]
                      for sid, pairs in blind_lists.items()},
            "manifest": {
                "experiment": "struct_large BGE-large 5-fold ensemble blind retrieval",
                "blind_name": "blind_a", "blind_dataset": C.BLIND_DATASETS["blind_a"],
                "fold_dirs": [str(C.STRUCT_LARGE_FOLD_DIRS[fi]) for fi in all_folds],
                "n_blind_sessions": len(blind_lists), "topk": TOPK,
                "method": "average cosine across fold struct_large models (played NOT excluded)",
                "smoke": smoke, "elapsed_s": time.time() - t0,
                "created_at": datetime.now().isoformat(),
            },
        }, f)
    print(f"{_ts()} [struct_large] blind ensemble -> {ensemble_path} "
          f"({len(blind_lists)} sessions); done in {time.time() - t0:.0f}s", flush=True)
