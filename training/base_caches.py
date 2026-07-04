"""Build the base retrieval caches from public data (no training):

  - qwen3 metadata embeddings  -> ``C.QWEN_DIR``   (``vectors.npy`` + ``track_ids.json``)
  - cf-bpr embeddings          -> ``C.CFBPR_DIR``  (``vectors.npy`` + ``track_ids.json``)
  - laion-CLAP audio embeddings -> ``C.CLAP_DIR``  (``vectors.npy`` + ``track_ids.json``)
  - BM25 index (4 metadata fields) -> ``C.BM25_INDEX``

These are the vector/BM25 indices the candidate sources (qwen A'/D, cf-bpr F,
clap_recent the acoustic "sounds-like" source, BM25 B/C) load at inference
(``recommender.sources``) and offline (``training.dev_payload``).

"Base" here means built purely from the public HuggingFace track datasets
(metadata + precomputed embeddings) with NO model training — distinct from the
fine-tuned retriever / ranker artifacts produced elsewhere in ``training/``.

Empty/invalid embedding rows are filtered (matching the validated build); vectors are
L2-normalised. CPU-only (~1 min once the HF embeddings are cached). Idempotent: each
output is skipped when already present unless ``force=True``.
"""
from __future__ import annotations

import json

import numpy as np

from recommender import config as C


def _build_vec_cache(col: str, outdir, dim: int, force=False):
    """Materialise one precomputed-embedding column into a ``(vectors.npy, track_ids.json)`` cache.

    Reads the public per-track embedding dataset and keeps the ``col`` vectors. Rows whose
    embedding is missing or not exactly ``dim`` long are dropped (the dataset stores some
    tracks with no/partial embedding), so the saved ``vectors.npy`` row i corresponds to
    ``track_ids.json`` entry i — the alignment every vector source relies on. Vectors are
    L2-normalised, so a later dot product equals cosine similarity.

    Args:
        col: dataset column holding the embedding (e.g. the qwen3 / cf-bpr / CLAP column).
        outdir: cache directory to write ``vectors.npy`` + ``track_ids.json`` into.
        dim: expected embedding length; rows of any other length are treated as invalid.
        force: rebuild even if both output files already exist.
    """
    if (outdir / "vectors.npy").exists() and (outdir / "track_ids.json").exists() and not force:
        print(f"  [skip] {outdir.name} present"); return
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset(C.DS_TRACK_EMB,
                      download_config=DownloadConfig(local_files_only=False))["all_tracks"]
    ids = [str(t) for t in ds["track_id"]]
    # Keep only well-formed embedding rows; `valid` tracks the surviving track_ids so the two
    # saved arrays stay row-aligned after the drop.
    raw, valid = [], []
    for i, v in enumerate(ds[col]):
        if v is not None and len(v) == dim:
            raw.append(v); valid.append(ids[i])
    M = np.asarray(raw, dtype=np.float32)
    # L2-normalise each row in place; the n==0 guard avoids dividing an all-zero vector by 0
    # (keepdims makes n shape (n_rows, 1) so it broadcasts over the embedding axis).
    n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1.0; M /= n
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "vectors.npy", M)
    (outdir / "track_ids.json").write_text(json.dumps(valid))
    print(f"  {col}: {M.shape} ({len(ids)-len(valid)} empty removed) -> {outdir.name}")


def _build_bm25(force=False):
    """Build the BM25 lexical index over four concatenated metadata fields per track.

    Each track's document is ``track_name + artist_name + album_name + tag_list`` joined into
    one string (list-valued fields like tags are space-joined first); the ``bm25s`` index over
    these documents backs the bm25_lastmusic (B) and bm25_convo (C) candidate sources. The
    parallel ``track_ids.json`` maps an index row back to its track_id (the saved corpus carries
    only integer ids, so the id list is the authoritative row->track_id map).
    """
    if (C.BM25_INDEX / "params.index.json").exists() and not force:
        print("  [skip] BM25 index present"); return
    import bm25s
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset(C.DS_TRACK_META,
                      download_config=DownloadConfig(local_files_only=False))["all_tracks"]
    # The four metadata fields concatenated into each track's searchable document.
    fields = ["track_name", "artist_name", "album_name", "tag_list"]
    ids, corpus = [], []
    for item in ds:
        ids.append(str(item["track_id"]))
        parts = []
        for f in fields:
            val = item.get(f)
            # tag_list (and any list field) is flattened to a space-separated string so it
            # tokenises as plain terms; missing fields contribute the empty string.
            if isinstance(val, list):
                val = " ".join(str(v) for v in val)
            parts.append(str(val) if val is not None else "")
        corpus.append(" ".join(parts))
    model = bm25s.BM25()
    model.index(bm25s.tokenize(corpus))
    C.BM25_INDEX.mkdir(parents=True, exist_ok=True)
    # Persist only a lightweight {"id": row_index} payload alongside the index; the real
    # track_id lookup lives in track_ids.json (written next), keeping the index file compact.
    model.save(str(C.BM25_INDEX), corpus=[{"id": i} for i in range(len(ids))])
    (C.BM25_INDEX / "track_ids.json").write_text(json.dumps(ids))
    print(f"  BM25: {len(ids)} tracks -> {C.BM25_INDEX.name}")


def build(force: bool = False, smoke: bool = False) -> None:
    """Build the qwen3 / cf-bpr / CLAP vector caches + the BM25 index.

    No-op for any output that already exists unless ``force=True``. ``smoke`` is accepted
    for interface uniformity; the build is already a one-shot pass over the catalog (a
    partial index would not match the shipped caches), so it does not subset.

    The three ``_build_vec_cache`` calls below differ only in (source column, output
    dir, expected dim). The ``dim`` values are the native widths of each precomputed
    embedding and act as a per-row validity filter — a row of any other length is
    treated as missing and dropped (see ``_build_vec_cache``):
      qwen3 metadata = 1024-d, cf-bpr collaborative = 128-d, laion-CLAP audio = 512-d.
    """
    # One status line; the ``…``/flush keeps progress visible when stdout is piped.
    print("[base_caches] qwen3 / cf-bpr / CLAP / BM25 …", flush=True)
    _build_vec_cache("metadata-qwen3_embedding_0.6b", C.QWEN_DIR, 1024, force)
    _build_vec_cache("cf-bpr", C.CFBPR_DIR, 128, force)
    # Track 1: the laion-CLAP acoustic ("sounds-like") recall source, most recently
    # shipped (fusion weight 0.5); 512-d audio embeddings -> clap_recent candidates.
    _build_vec_cache("audio-laion_clap", C.CLAP_DIR, 512, force)   # Track 1: acoustic recall source
    _build_bm25(force)
