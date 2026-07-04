"""From-scratch rebuild DAG: run every training stage in dependency order.

Each stage is a module under ``training`` exposing ``build(force=False, smoke=False)``.
A stage no-ops when its outputs already exist and ``force`` is False, so with the shipped
artifacts in place ``train_all`` is a no-op; with an empty cache it rebuilds the whole
pipeline (the heavy GPU stages — text_retriever/struct_base/struct_large — dominate the
wall time).

Stage order (and why):

  1. base_caches        BM25 index + track-similarity caches (no deps).
  2. als                ALS factor matrix from the interaction log (no deps).
  3. dev_payload        the 8000 dev cases + base source lists + metadata maps;
                        every downstream stage reads cases from here.
  4. text_retriever     text-retriever BGE-base 5-fold OOF + blind lists  (needs dev_payload).
  5. struct_base        struct_base structured-query 5-fold OOF + blind   (needs dev_payload).
  6. blind_source_cache the blind base-source cache struct_large's blind encode reads
                        (needs base_caches + als + text_retriever).
  7. struct_large       struct_large BGE-large 5-fold OOF + blind ensemble
                        (needs dev_payload + blind_source_cache).
  8. case_features      the 8-source RRF feature matrix the CatBoost ensemble trains on
                        (needs dev_payload + text/struct_base/struct_large OOF + als + maps/pop).
  9. lgbm_rankers       the two LightGBM LambdaRank rankers; internally builds its own
                        9-source (struct_large-in-pool) case_features variant on demand.
 10. catboost_ensemble  the CatBoost-YetiRank ensemble; consumes the 8-source case_features
                        from stage 8.
 11. transfer_weighting adversarial dev-vs-blind importance weights + weighted retrain of
                        stages 9-10's protocol -> the FINAL serving rankers (cache/rankers/).

Modules are imported lazily inside the loop so a heavy import (torch, lightgbm, catboost …)
does not load until that stage actually runs.
"""
from __future__ import annotations

# (module_name, human label) in dependency order. Each module exposes build(force, smoke).
# This list IS the build DAG flattened to a topological order: every stage only reads
# artifacts produced by stages that appear ABOVE it (see the per-stage "needs ..." notes in
# the module docstring), so a plain top-to-bottom walk respects all dependencies. Tuple shape
# is (importable submodule name under `training`, short human-readable label for progress logs).
# `module_name` must match a real training.<name>.py exposing build(force, smoke) -> None.
_STAGES: list[tuple[str, str]] = [
    ("base_caches", "BM25 + track-sim caches"),
    ("als", "ALS factor matrix"),
    ("dev_payload", "dev cases + base sources + maps"),
    ("text_retriever", "text-retriever BGE-base OOF + blind"),
    ("struct_base", "struct_base structured OOF + blind"),
    ("blind_source_cache", "blind base-source cache"),
    ("struct_large", "struct_large BGE-large OOF + blind ensemble"),
    ("case_features", "8-source RRF feature matrix"),
    ("lgbm_rankers", "LightGBM LambdaRank rankers"),
    ("catboost_ensemble", "CatBoost-YetiRank ensemble"),
    ("transfer_weighting", "transfer-weighted final rankers"),
]


def train_all(smoke: bool = False, force: bool = False) -> None:
    """Rebuild every artifact from scratch, in dependency order.

    Each stage's ``build`` is idempotent (skips when its outputs exist unless ``force``),
    so this is safe to re-run; under ``smoke`` each stage does a reduced fast pass.

    Args:
        smoke: run each stage in fast smoke mode (one fold / reduced rounds) for an
            end-to-end wiring check rather than a full rebuild.
        force: rebuild every stage even if its outputs already exist.
    """
    print(f"=== train-all (smoke={smoke}, force={force}) — {len(_STAGES)} stages ===",
          flush=True)
    # flush=True on every print so progress is visible immediately even when stdout is piped
    # to a log file (these runs are long — the heavy GPU stages can take hours).
    # start=1 makes `i` a 1-based, human-readable stage counter for the "stage i/N" banners
    # (matches how the stages are numbered 1..10 in the module docstring's DAG listing).
    # There is no try/except around build(): the loop is intentionally FAIL-FAST — if any stage
    # raises, the exception propagates and aborts the whole rebuild rather than leaving a
    # half-built, internally-inconsistent cache that later stages would silently consume.
    for i, (mod_name, label) in enumerate(_STAGES, start=1):
        print(f"\n[train-all] stage {i}/{len(_STAGES)}: {mod_name} — {label} "
              f"(smoke={smoke}, force={force})", flush=True)
        # fromlist=["build"] makes __import__ return the SUBMODULE training.<mod_name> (not the
        # top-level `training` package), so module.build resolves to that stage's entry point.
        # Importing here, inside the loop, defers each stage's heavy deps to when it actually runs.
        module = __import__(f"training.{mod_name}", fromlist=["build"])
        module.build(force=force, smoke=smoke)
    print("\n=== train-all done ===", flush=True)
