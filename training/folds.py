"""The pipeline-wide grouped-session cross-validation split.

A single source of truth for the 5-fold, session-grouped CV used across the whole
offline pipeline: the text_retriever (``training.text_retriever``), the struct_base
structured-query retriever (``training.struct_base``), the struct_large large retriever,
and the per-case fold map ``training.case_features`` computes.

The contract that makes the offline featurisation honest is that *every* stage which
derives folds does so identically, so a dev case held out by one retriever is held out
by all of them and the rankers never train on a case any retriever already saw. That
shared assignment is exactly this function with ``seed=0, k=5`` over the dev cases in
their ``C.DEV_PAYLOAD["cases"]`` order.

Verified byte-identical to the canonical fold map: running
``grouped_session_folds(sessions, seed=0, k=5)`` over the dev payload's case order
reproduces the recorded ``fold_idx`` for all 8000 cases (8000/8000). Ported verbatim from
the validated grouped-session fold function; ``training.text_retriever`` and
``training.struct_base`` both import it from here so the contract has exactly one
implementation.
"""
from __future__ import annotations

import numpy as np


def grouped_session_folds(sessions: list[str], seed: int = 0,
                          k: int = 5) -> list[np.ndarray]:
    """Split case indices into ``k`` folds by unique ``session_id`` (no session leakage).

    Every turn (case) of a session lands in the same fold, so a model trained on the
    complement of a fold has never seen any turn of that fold's sessions. The unique
    sessions are sorted then shuffled with a seeded RNG (deterministic) and round-robin
    assigned to folds; each case inherits its session's fold.

    Args:
        sessions: ``session_id`` per case, in case-index order (the order of
            ``C.DEV_PAYLOAD["cases"]``).
        seed: RNG seed for the session shuffle. The pipeline-wide contract is ``0``.
        k: number of folds. The pipeline-wide contract is ``5``.

    Returns:
        ``k`` arrays of case indices (``np.int64``); ``folds[f]`` are the cases held
        out in fold ``f``. The k arrays partition ``range(len(sessions))`` exactly once
        (every case appears in exactly one fold), so callers can safely invert them into a
        dense per-case fold map. Why this matters downstream: a retriever trained on the
        complement of ``folds[f]`` produces the out-of-fold (OOF) predictions for those
        cases, and because all three retrievers (text_retriever/struct_base/struct_large)
        share this exact split, no LightGBM/CatBoost ranker is ever trained on a case any
        retriever already saw — the leakage-free guarantee the offline featurisation relies on.
    """
    # sorted() first so the input order of `sessions` cannot affect the result: the shuffle
    # then operates on a canonical list, making the assignment a pure function of (sessions
    # set, seed, k). This is what makes the fold map byte-identical across every stage.
    unique_sessions = sorted(set(sessions))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_sessions)
    # Round-robin the shuffled sessions across folds (i % k): with the shuffle already random,
    # this yields near-equal fold sizes without a second random draw.
    session_to_fold = {sid: i % k for i, sid in enumerate(unique_sessions)}
    folds: list[list[int]] = [[] for _ in range(k)]
    # Every case inherits its session's fold, so all turns of a session co-locate (no leakage).
    # `case_idx` is the position of the case in the original `sessions` list (i.e. its index in
    # C.DEV_PAYLOAD["cases"]); these are exactly the indices callers use to slice OOF lists.
    for case_idx, sid in enumerate(sessions):
        folds[session_to_fold[sid]].append(case_idx)
    # int64 (not the platform-default int) so the arrays index large numpy feature matrices and
    # round-trip through pickled artifacts identically on 32/64-bit and Windows/Linux. A fold can
    # be empty in principle; np.asarray([]) would default to float64, so the explicit dtype also
    # keeps every fold array integer-typed regardless of contents.
    return [np.asarray(f, dtype=np.int64) for f in folds]
