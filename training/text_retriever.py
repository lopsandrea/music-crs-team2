"""Train the supervised text retriever (``text_retriever``): 5-fold OOF dev lists +
a production model + full-catalog embeddings.

Naming note: in the locked legacy vocabulary this source is ``r21`` (``r21 ==
text_retriever``). The clean name is used throughout this module, but the on-disk
home is still ``C.TEXT_RETRIEVER_DIR == cache/retrievers/text_retriever`` and the
LightGBM feature columns it feeds are named ``r21_rank_inv`` / ``r21_presence`` —
those ``r21`` tokens are RETAINED ON PURPOSE because they are baked into the shipped
LightGBM ``feature_name`` vector, so renaming would force a full GPU retrain
(legacy tokens: r21 == text_retriever, r54 == struct_base, r84 == struct_large).
It is not a bug or a TODO.

What ``text_retriever`` is
--------------------------
A supervised BGE-base bi-encoder (``BAAI/bge-base-en-v1.5``) fine-tuned to map a
short conversation query (the last few user turns) close to the metadata text of
the gold next track. It is one of the nine candidate sources the champion fuses
(``recommender.config.SW_BASELINE["text_retriever"]``): at serve time
``recommender.sources`` loads ``C.TEXT_RETRIEVER_DIR/model`` + the pre-encoded catalog
(``track_embeddings.npy`` / ``track_ids.json``) and retrieves the nearest tracks
to each query embedding.

Two kinds of artifact are produced:

* **OOF dev lists** (``C.TEXT_RETRIEVER_OOF``) — used by ``training.case_features`` to build
  the per-case ranker-training matrices. To keep the rankers honest, every dev
  case must be scored by a model that did NOT train on it, so the dev set is split
  into 5 grouped-by-session folds and each fold's held-out cases are retrieved with
  a model trained on the other four. The result is one top-300 list per dev case,
  written case-index-ordered as ``[n_cases][track_id]`` — exactly the format
  ``case_features._load_text_oof`` reads.

* **Production model + catalog embeddings** (``C.TEXT_RETRIEVER_DIR``) — a model trained on
  *all* dev cases plus the full catalog encoded once, for blind/live inference.

Fold contract (critical — must match the rest of the pipeline)
--------------------------------------------------------------
Folds are assigned by ``grouped_session_folds(sessions, seed=0, k=5)`` over the
dev cases in their ``C.DEV_PAYLOAD["cases"]`` order. That payload is built by
``training.dev_payload.load_dev_cases``, so the case ordering — and therefore the
``sessions`` array and the resulting fold assignment — is byte-identical to the
payload the original trainer used. The same ``grouped_session_folds(sessions, seed=0, k=5)``
split is the one every retriever holds cases out by (the struct_large OOF fold map
``case_features`` computes), so the text-retriever's OOF folds and struct_large's OOF folds
agree case-for-case.

Provenance: faithfully reproduced from the validated text-retriever 5-fold OOF trainer; the
heavy 5-fold + production rebuild is the user's deferred command (~GPU-hours on the
2x RTX 4090). Diagnostic-only manifest fields from the original are dropped; they never
affected the OOF lists or any champion artifact. The OOF lists, production model, and
catalog embeddings are reproduced exactly.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np

from recommender import config as C
from recommender.data import load_track_metadata
from training.folds import grouped_session_folds

# --- training hyper-parameters (verbatim from the validated text-retriever trainer) ---
MODEL_NAME = "BAAI/bge-base-en-v1.5"
EPOCHS = 2
BATCH_SIZE = 32
LR = 2e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
TEMPERATURE = 0.05          # in-batch contrastive softmax temperature
MAX_LENGTH = 256            # tokenizer truncation length
ENCODE_TRACK_BS = 128       # catalog encode batch size
ENCODE_QUERY_BS = 64        # query encode batch size
TOPK = 300                  # OOF list length (== shipped dev_oof_lists.json inner length)
N_FOLDS = 5
FOLD_SEED = 0               # grouped-session CV seed — the pipeline-wide fold contract
QUERY_TURNS = C.TEXT_QUERY_TURNS  # number of trailing query parts to keep (=3)


def _ts() -> str:
    """Timestamp prefix ``[YYYY-MM-DD HH:MM:SS]`` for progress logs (cosmetic only)."""
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


# ---------------- text construction (text-retriever-specific; ported verbatim) ----------------

def build_track_text(tid: str, meta: dict[str, dict]) -> str:
    """Metadata text for a track: ``"<name> by <artist>. Album: <alb>. Tags: <tags>"``.

    The catalog/positive side of the bi-encoder. Note this is the text-retriever's own format
    — NOT ``recommender.text.meta_text`` (the BM25 tokeniser), which renders a
    different string — so it is reproduced here exactly as the original trainer.
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


def build_query_text(case: dict) -> str:
    """Conversation query: the last ``QUERY_TURNS`` of (prior user turns + current query).

    The query side of the bi-encoder. Concatenates the user-role ``history``
    contents and the current ``user_query``, then keeps the trailing
    ``QUERY_TURNS`` (=3) space-joined.
    """
    parts = []
    for h in case["history"]:
        if h["role"] == "user":
            parts.append(str(h["content"]))
    parts.append(case["user_query"])
    return " ".join(parts[-QUERY_TURNS:])


# ---------------- fold assignment (the pipeline-wide fold contract) ----------------
# The grouped-session CV split now lives in ``training.folds`` (the single source of
# truth shared with struct_base/struct_large/case_features); imported above as
# ``grouped_session_folds``.


# ---------------- catalog ----------------

def _load_catalog() -> tuple[dict[str, dict], list[str]]:
    """``(meta, all_track_ids)`` over the full ``all_tracks`` catalog.

    Reuses ``recommender.data.load_track_metadata`` (the same HF ``all_tracks``
    split the original ``load_catalog`` read); the ordered id list is the dict's
    insertion order, i.e. the dataset row order — identical to the legacy trainer's
    arrow-iteration order, so ``track_ids.json`` / ``track_embeddings.npy`` line up.
    """
    meta = load_track_metadata()
    all_track_ids = list(meta.keys())
    assert len(all_track_ids) == len(set(all_track_ids)), "duplicate track IDs in catalog"
    return meta, all_track_ids


# ---------------- training / encoding ----------------

def _train_model(train_cases: list[dict], meta: dict[str, dict], model_dir,
                 epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LR):
    """Fine-tune BGE-base with an in-batch contrastive loss; save to ``model_dir``.

    Verbatim recipe from the legacy ``train_fold_model``: for each training case
    build a (query_text, gold_track_text) pair, then for every batch encode queries
    and positives with grads, score ``q @ p.T / TEMPERATURE``, and apply
    cross-entropy against the diagonal (each query's positive is its in-batch
    target; the other positives are negatives). AdamW (lr, weight_decay), gradient
    clipping at 1.0, ``epochs`` passes with a per-epoch shuffle.
    """
    import torch
    import torch.nn.functional as F_t
    from sentence_transformers import InputExample, SentenceTransformer

    examples = []
    for c in train_cases:
        gt = c["gt"]
        if gt not in meta:
            continue
        examples.append(InputExample(texts=[build_query_text(c), build_track_text(gt, meta)]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    tokenizer = model.tokenizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    def encode_with_grad(texts):
        # Tokenise + forward WITH autograd (unlike model.encode, which runs no_grad),
        # then L2-normalise so a dot product is cosine similarity. Returns (len(texts), dim).
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH,
                            return_tensors="pt")
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        out = model.forward(encoded)
        return F_t.normalize(out["sentence_embedding"], dim=-1)

    model.train()
    for epoch in range(epochs):
        np.random.shuffle(examples)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            queries = [ex.texts[0] for ex in batch]
            positives = [ex.texts[1] for ex in batch]
            q_emb = encode_with_grad(queries)
            p_emb = encode_with_grad(positives)
            # In-batch contrastive: sim is (B, B) of cosine(query_i, positive_j) / TEMPERATURE.
            # The correct positive for query i is its own positive (the diagonal), so the
            # cross-entropy targets are arange(B); every other in-batch positive is an
            # implicit negative. The low TEMPERATURE (0.05) sharpens the softmax.
            sim = q_emb @ p_emb.T / TEMPERATURE
            labels = torch.arange(len(batch), device=sim.device)
            loss = F_t.cross_entropy(sim, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"      batch {n_batches}: loss={loss.item():.4f}", flush=True)
        print(f"    epoch {epoch}: loss={epoch_loss / max(n_batches, 1):.4f}", flush=True)

    model.save(str(model_dir))
    return model


def _encode_and_retrieve(model, track_texts: list[str], all_track_ids: list[str],
                         val_cases: list[dict], topk: int = TOPK):
    """Encode the catalog + the held-out queries; return per-case top-``topk`` id lists.

    Verbatim from the legacy ``encode_and_retrieve``: cosine over L2-normalised
    embeddings, with each case's already-played tracks masked out (``-inf``) before
    the top-``topk`` argpartition+sort.
    """
    print(f"    encoding {len(all_track_ids)} tracks …", flush=True)
    track_embs = model.encode(track_texts, batch_size=ENCODE_TRACK_BS, show_progress_bar=True,
                              normalize_embeddings=True).astype(np.float32)

    val_queries = [build_query_text(c) for c in val_cases]
    print(f"    encoding {len(val_queries)} queries …", flush=True)
    query_embs = model.encode(val_queries, batch_size=ENCODE_QUERY_BS, show_progress_bar=True,
                              normalize_embeddings=True).astype(np.float32)

    print(f"    retrieving top-{topk} …", flush=True)
    results = []
    for i in range(len(val_cases)):
        # scores[j] = cosine(query_i, track_j) since both sides are L2-normalised; shape (n_tracks,).
        scores = track_embs @ query_embs[i]
        # Never recommend a track the user already played this session: mask to -inf.
        played_set = set(val_cases[i]["music_turns"])
        for idx, tid in enumerate(all_track_ids):
            if tid in played_set:
                scores[idx] = -np.inf
        # argpartition gives the top-`topk` indices unordered (O(n)); the argsort then
        # orders just those `topk` by descending score. Cheaper than a full sort of n_tracks.
        # Negating scores turns "largest cosine" into "smallest value", so partitioning at
        # position `topk` places the `topk` highest-scoring tracks in the leading slice
        # [:topk] (their internal order is arbitrary until the argsort below). Requires
        # topk < n_tracks, which holds since the catalog is far larger than TOPK=300.
        top_idx = np.argpartition(-scores, topk)[:topk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        results.append([all_track_ids[j] for j in top_idx])
    return results


# ---------------- build ----------------

def build(force: bool = False, smoke: bool = False) -> None:
    """Train the text retriever: 5-fold OOF dev lists -> ``C.TEXT_RETRIEVER_OOF``, then the
    production model + catalog embeddings -> ``C.TEXT_RETRIEVER_DIR``.

    No-op when the outputs already exist unless ``force=True``. Per-fold OOF
    artifacts are cached under ``C.TEXT_RETRIEVER_DIR/oof`` and resumed if present, so an
    interrupted multi-hour run continues where it stopped.

    ``smoke=True`` runs a genuinely small but complete pass: ONE fold trained on a
    tiny subset of cases for one short epoch, then that fold's held-out cases are
    retrieved and the full catalog is encoded into a production model — exercising
    the whole train + encode + retrieve + write path quickly (no 5-fold, no full
    training set). It still writes plausibly-shaped ``C.TEXT_RETRIEVER_OOF`` /
    ``track_embeddings.npy`` / ``track_ids.json`` (the OOF list is only populated
    for the smoke fold's held-out cases; the rest stay empty).
    """
    oof_path = C.TEXT_RETRIEVER_OOF
    model_dir = C.TEXT_RETRIEVER_DIR / "model"
    embs_path = C.TEXT_RETRIEVER_DIR / "track_embeddings.npy"
    ids_path = C.TEXT_RETRIEVER_DIR / "track_ids.json"

    if (oof_path.exists() and model_dir.exists() and embs_path.exists()
            and ids_path.exists() and not force):
        print("  [skip] text-retriever OOF + production model present"); return

    t0 = time.time()
    oof_dir = C.TEXT_RETRIEVER_DIR / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)

    print(f"{_ts()} loading dev payload + catalog …", flush=True)
    import pickle
    with open(C.DEV_PAYLOAD, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]

    meta, all_track_ids = _load_catalog()
    print(f"  catalog: {len(all_track_ids)} tracks", flush=True)
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]

    # The pipeline-wide fold contract: same (seed=0, k=5) grouped-session split every
    # retriever holds cases out by, so the OOF lists are mutually consistent (see module docstring).
    folds = grouped_session_folds(sessions, seed=FOLD_SEED, k=N_FOLDS)
    fold_range = [0] if smoke else list(range(N_FOLDS))

    # all_oof_lists[case_idx] = that case's top-300 list, filled in below by whichever
    # fold held the case out; left empty for cases no processed fold covers (smoke mode).
    all_oof_lists: list = [[] for _ in range(n)]
    manifest = {"catalog_size": len(all_track_ids),
                "unique_track_ids": len(set(all_track_ids)),
                "smoke": smoke, "folds": {}}

    # ===== 5-fold OOF (one tiny fold under smoke) =====
    for fold_i in fold_range:
        fold_file = oof_dir / f"fold_{fold_i}_lists.json"

        # Resume support: a finished fold's lists are cached to disk, so an interrupted
        # multi-hour run reloads them instead of retraining. held[] order matches the
        # saved "lists" order (both come from folds[fold_i]), so the local->global scatter holds.
        if fold_file.exists() and not force:
            print(f"\n{_ts()} fold {fold_i}: found cached artifact, loading …", flush=True)
            with open(fold_file) as f:
                fold_data = json.load(f)
            held = folds[fold_i].tolist()
            for j_local, j_global in enumerate(held):
                all_oof_lists[j_global] = fold_data["lists"][j_local]
            manifest["folds"][str(fold_i)] = fold_data["manifest"]
            continue

        # held = this fold's case indices (out-of-fold); train on the other four folds.
        held = folds[fold_i].tolist()
        train_idx = [j for j in range(n) if j not in set(held)]

        if smoke:
            # tiny but real: a small training subset + a small held-out slice
            train_idx = train_idx[:256]
            held = held[:64]

        print(f"\n{_ts()} fold {fold_i}: train={len(train_idx)} val={len(held)}", flush=True)
        fold_model_dir = oof_dir / f"model_fold_{fold_i}"
        train_cases = [cases[j] for j in train_idx]
        fold_model = _train_model(train_cases, meta, fold_model_dir,
                                  epochs=1 if smoke else EPOCHS)

        val_cases = [cases[j] for j in held]
        fold_lists = _encode_and_retrieve(fold_model, track_texts, all_track_ids, val_cases)

        # Scatter the fold's per-held-case lists back into the global case-index array.
        for j_local, j_global in enumerate(held):
            all_oof_lists[j_global] = fold_lists[j_local]

        # Diagnostic only: how often the gold next track lands in the top-200 of the OOF list.
        # Logged to gauge recall headroom; it does NOT gate or alter any written artifact.
        hit = sum(1 for j_local, j_global in enumerate(held)
                  if cases[j_global]["gt"] in fold_lists[j_local][:200])
        fold_manifest = {
            "fold": fold_i, "train_cases": len(train_idx), "val_cases": len(held),
            "model_path": str(fold_model_dir), "hit@200": hit,
            "hit@200_rate": hit / max(len(held), 1),
            "catalog_size": len(all_track_ids), "created_at": datetime.now().isoformat(),
        }
        with open(fold_file, "w") as f:
            json.dump({"lists": fold_lists, "manifest": fold_manifest}, f)
        manifest["folds"][str(fold_i)] = fold_manifest
        print(f"  fold {fold_i}: hit@200={hit}/{len(held)} ({hit / max(len(held), 1):.1%})  "
              f"saved {fold_file}", flush=True)

        # Free the fold model before training the next one (each holds a full BGE-base on GPU).
        del fold_model

    # Combine OOF lists -> C.TEXT_RETRIEVER_OOF (case-index ordered [n_cases][track_id])
    # Full (non-smoke) runs must cover every case across the 5 folds; a gap means a fold
    # was skipped or mis-scattered, which would corrupt the ranker training matrices.
    if not smoke:
        assert all(len(x) > 0 for x in all_oof_lists), "missing OOF lists for some cases"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    with open(oof_path, "w") as f:
        json.dump(all_oof_lists, f)

    covered = [i for i in range(n) if all_oof_lists[i]]
    manifest["total_hit@200"] = sum(1 for i in covered
                                    if cases[i]["gt"] in all_oof_lists[i][:200])
    manifest["total_hit@200_rate"] = manifest["total_hit@200"] / max(len(covered), 1)
    manifest["created_at"] = datetime.now().isoformat()
    with open(C.TEXT_RETRIEVER_DIR / "oof_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{_ts()} OOF lists -> {oof_path}  "
          f"(covered {len(covered)}/{n}, hit@200_rate={manifest['total_hit@200_rate']:.4f})",
          flush=True)

    # ===== production model (all cases) + full-catalog embeddings =====
    # The serve-time model: trained on ALL dev cases (no held-out fold needed — it is never
    # used to score its own training cases offline) and used to encode the catalog once.
    if model_dir.exists() and embs_path.exists() and ids_path.exists() and not force:
        print(f"{_ts()} production model already present, skipping.", flush=True)
    else:
        prod_cases = cases[:256] if smoke else cases
        print(f"\n{_ts()} training production model (all {len(prod_cases)} cases) …", flush=True)
        prod_model = _train_model(prod_cases, meta, model_dir, epochs=1 if smoke else EPOCHS)

        print(f"{_ts()} encoding catalog for production …", flush=True)
        prod_model.eval()
        track_embs = prod_model.encode(track_texts, batch_size=ENCODE_TRACK_BS,
                                       show_progress_bar=True,
                                       normalize_embeddings=True).astype(np.float32)
        np.save(embs_path, track_embs)
        with open(ids_path, "w") as f:
            json.dump(all_track_ids, f)
        print(f"  production embeddings {track_embs.shape} -> {embs_path}", flush=True)
        del prod_model

    print(f"\n{_ts()} done. elapsed {time.time() - t0:.1f}s", flush=True)
