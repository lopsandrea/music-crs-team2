"""Train ALS collaborative factors -> ``C.ALS_NPZ`` (source ALS + the ``als_dot`` feature).

Verbatim recipe from the validated pipeline (``expS2_lambdarank.build_als`` /
``scripts/rebuild_als_py311.py``): the session x track implicit matrix over the train
split, ALS factors=128, alpha=100, reg=0.05, 20 iters, seed=42 (deterministic, CPU).
``track_ids`` are ``sorted(track_set)`` and the factor rows are ALS item factors in that
order. Requires ``implicit`` (cp311 wheel; this repo's env is Python 3.11).

Writes ``np.savez(C.ALS_NPZ, factors=<float32 (n_tracks, 128)>, track_ids=<str array>)``,
exactly the format ``recommender.data.load_als`` reads.
"""
from __future__ import annotations

import numpy as np

from recommender import config as C

# ALS hyper-parameters — frozen to the validated recipe so re-fits reproduce the
# shipped factors. FACTORS is the latent dimensionality (128 -> factor rows are
# (n_tracks, 128)); ALPHA scales the implicit-feedback confidence weight applied to
# each observed play (confidence = 1 + ALPHA*r, here r is the binary play count);
# REG is the L2 ridge on the factors; ITERS is the number of alternating sweeps;
# SEED makes the (CPU, deterministic) fit reproducible bit-for-bit.
FACTORS = 128
ALPHA = 100
REG = 0.05
ITERS = 20
SEED = 42


def build(force: bool = False, smoke: bool = False) -> None:
    """Build ALS factors and write them to ``C.ALS_NPZ``.

    No-op when the artifact already exists unless ``force=True``. ``smoke`` is accepted
    for interface uniformity but does not subset (the fit is deterministic and fast; a
    partial-data fit would not match the shipped factors).
    """
    # Idempotent build gate: if the .npz is already on disk we assume it is the
    # shipped/validated artifact and skip the (deterministic) refit. force=True
    # overrides to rebuild from scratch.
    if C.ALS_NPZ.exists() and not force:
        print("  [skip] ALS factors present"); return

    print("[als] building session x track matrix + fitting ALS …", flush=True)
    # Heavy deps imported lazily (only when actually rebuilding) so importing this
    # module stays cheap and does not require implicit/datasets to be installed.
    from datasets import DownloadConfig, load_dataset
    from implicit.als import AlternatingLeastSquares
    from scipy import sparse

    # DS_CONVO == "talkpl-ai/TalkPlayData-Challenge-Dataset" (config.C.DS_CONVO); the
    # "train" split is the conversation corpus the factors are fit on. local_files_only=False
    # permits a Hub download when the dataset is not yet cached locally.
    train = load_dataset(C.DS_CONVO,
                         download_config=DownloadConfig(local_files_only=False))["train"]
    # One "session" per conversation; its tracks are the contents of the music-role turns
    # (the songs played during that conversation), in order.
    track_set: set[str] = set()
    session_tracks: list[list[str]] = []
    for item in train:
        tracks = [str(c["content"]).strip() for c in item["conversations"] if c["role"] == "music"]
        session_tracks.append(tracks)
        track_set.update(tracks)
    # Column order of the factor matrix is sorted(track_set) — the SAME ordering
    # recommender.data.load_als assumes when it zips factors[i] back to track_ids[i].
    track_ids = sorted(track_set)
    idx = {t: i for i, t in enumerate(track_ids)}

    # Build the implicit session x track matrix as COO triplets: row = session index,
    # col = track index, value = 1.0 (a play is binary; repeats collapse to one entry in
    # csr_matrix, which sums duplicates — but ALPHA, not the count, drives confidence).
    rows, cols = [], []
    for si, tracks in enumerate(session_tracks):
        for t in tracks:
            rows.append(si); cols.append(idx[t])
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(session_tracks), len(track_ids)), dtype=np.float32)

    model = AlternatingLeastSquares(factors=FACTORS, alpha=ALPHA, regularization=REG,
                                    iterations=ITERS, random_state=SEED, use_gpu=False)
    model.fit(matrix)
    # item_factors has shape (n_tracks, FACTORS): one latent row per catalog track, aligned
    # to track_ids. These are the rows recommender.data.load_als exposes for the als_session
    # source and the als_dot feature (a dot product against the session's history vector).
    factors = model.item_factors
    # On a GPU/cudf build item_factors may be a cupy/cudf object; .to_numpy() pulls it to host.
    # On the CPU build (this repo's path) it is already an ndarray and this branch is skipped.
    if hasattr(factors, "to_numpy"):
        factors = factors.to_numpy()
    factors = np.asarray(factors, dtype=np.float32)

    C.ALS_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(C.ALS_NPZ, factors=factors, track_ids=np.array(track_ids))
    print(f"[als] factors {factors.shape} over {len(track_ids)} tracks -> {C.ALS_NPZ}", flush=True)
