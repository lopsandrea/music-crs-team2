# Rebuilding the models (validation point 2)

Two documented paths, both starting from a clean machine:

| Path | What you get | Time | Hardware |
|---|---|---|---|
| **[1. Exact reproduction](#1-exact-reproduction-of-the-serving-rankers-20-min)** — re-derive the shipped serving rankers from their exact training inputs | the rankers behind submission 827190, verified end-to-end against the submitted payload | ~20 min | 1 GPU |
| **[2. Full retrain from scratch](#2-full-retrain-from-scratch-1-gpu-day)** — rebuild every artifact from the official data only | the entire `cache/` tree regenerated (retrievers included) | ~1 GPU day | 24 GB GPU |

Why two paths: deep-learning fine-tunes are only approximately reproducible on GPUs, so a
full retrain cannot be bit-exact by nature. Path 1 gives the bit-level guarantee on the
trainable final stage; path 2 proves the whole pipeline regenerates from official data.

## 0. Requirements & setup

Same environment as [INFERENCE.md](INFERENCE.md) §0-§2 (uv sync + weights download — the
weights set includes the exact training inputs used by path 1). Additionally for path 2:
an NVIDIA GPU with ≥24 GB (RTX 4090-class; the CatBoost stages require a GPU, the BGE
fine-tunes want the full 24 GB), ~64 GB RAM, ~40 GB free disk.

Training data (all public, official challenge repos — fetched automatically on first
use, or explicitly with):

```bash
uv run python scripts/download_data.py --training   # ~4 GB total
```

| Input | Source |
|---|---|
| conversations (train split + the 8000 held-out dev cases) | `talkpl-ai/TalkPlayData-Challenge-Dataset` |
| track catalog (`all_tracks`, 47,071) | `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` |
| precomputed catalog embeddings (qwen3 / cf-bpr / CLAP columns) | `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings` |
| blind sessions (Blind-B to serve; Blind-A for the adversarial weights, see stage 11) | `talkpl-ai/TalkPlayData-Challenge-Blind-{B,A}` |
| base encoders fine-tuned by stages 4/5/7 | `BAAI/bge-base-en-v1.5`, `BAAI/bge-large-en-v1.5` |

**No external or private data is used anywhere.** Candidate retrieval always operates
over the entire `all_tracks` catalog; no split filtering exists at any stage.

## 1. Exact reproduction of the serving rankers (~20 min)

The shipped rankers are an importance-weighted retrain (see "stage 11" below) over two
per-case training matrices. The weights set ships those **exact matrices**
(`cache/training/case_features.pkl`, 8-source pool → CatBoost;
`case_features_r84pool.pkl`, 9-source pool → LightGBM pair). Re-derive the serving
rankers from them:

```bash
uv run python -c "import sys; sys.path.insert(0,'.'); \
  import training.transfer_weighting as t; t.build(force=True)"
```

This (a) rebuilds the adversarial dev-vs-blind case weights from scratch — expected
output includes `adversarial AUC (dev vs blind_a) = 0.9369`, and the stored weights are
reproduced to `max|Δ| ≤ 4e-15`; (b) retrains the two LightGBM boosters + the CatBoost
ensemble with those weights and **overwrites `cache/rankers/`**.

**Expected result.** The retrained models induce rankings identical to the shipped ones
(we checked top-20 and top-1 on 2000/2000 training cases; saved files differ from the
shipped bytes only in trailing float digits). The end-to-end proof:

```bash
uv run python scripts/verify_inference.py
# → matched 80/80 sessions against reference/prediction_827190.json
```

i.e. **the rankers you just trained reproduce the submitted Blind-B top-20 lists
exactly** (verified). To return to the shipped binaries at any time:
`uv run python scripts/download_weights.py` (re-downloads anything whose sha changed).

**Why the exact matrices are shipped (lineage note, full transparency).** The shipped
matrices were built on 2026-06-12 and their generating stage (`training/case_features.py`,
stage 8 below) is fully deterministic given its inputs. However, on 2026-06-15 the dev
payload was regenerated while being relocated, and its BM25 index rebuild produced
sub-rank-level differences deep in two precomputed source lists — so matrices rebuilt
from today's payload differ microscopically (order flips deep in the 300-candidate
pools) from the June-12 originals. Shipping the originals keeps this path bit-exact;
the regeneration path itself is exercised by path 2. No non-official data is involved
either way.

## 2. Full retrain from scratch (~1 GPU day)

```bash
uv run python run.py train-all --force
```

`train-all` runs an idempotent 11-stage DAG (`training/train_all.py`); `--force`
rebuilds everything, without it each stage skips when its outputs already exist (so on
top of the downloaded weights a plain `train-all` is a no-op).

| # | stage | output | GPU | time* |
|---|---|---|---|---|
| 1 | `base_caches` | BM25 index; qwen3/cf-bpr/CLAP `vectors.npy` repackaged from the official embeddings; metadata/popularity maps | no | ~10 min |
| 2 | `als` | `als_factors.npz` (implicit ALS, seed 42, CPU — deterministic) | no | ~10 min |
| 3 | `dev_payload` | the 8000 parsed dev cases + base source lists | no | ~20 min |
| 4 | `text_retriever` | fine-tuned bi-encoder: 5-fold OOF lists + the serving model/catalog embeddings | yes | ~2-4 h |
| 5 | `struct_base` | 5-fold structured-query BGE-base: OOF + `blind_lists_blind_b.json` | yes | ~2-4 h |
| 6 | `blind_source_cache` | per-session Blind-B base-source cache (read by stage 7's blind encode) | small | ~10 min |
| 7 | `struct_large` | 5-fold BGE-large: per-fold OOF + `blind_lists_blind_b.json` | yes | ~4-7 h |
| 8 | `case_features` | the 8-source training matrix | no | ~2 min |
| 9 | `lgbm_rankers` | plain LightGBM pair (also builds the 9-source matrix) | no | ~15 min |
| 10 | `catboost_ensemble` | plain CatBoost-YetiRank ensemble | yes | ~10 min |
| 11 | `transfer_weighting` | **the final serving rankers** — adversarial dev-vs-blind importance weights (gold-free session descriptors, 5-fold CV) + weighted retrain of the stage-9/10 protocol at `TRANSFER_ALPHA=0.25` → `cache/rankers/` | yes | ~20 min |

\* on 1× RTX 4090; stages 4/5/7 dominate and parallelise across folds/GPUs.

Stage 11 is what serving actually loads: dev and blind sessions are not identically
distributed (session depth, query shape, index coverage of played tracks, popularity),
so the final rankers are trained with a standard covariate-shift correction — dev cases
weighted by `p(blind|x)` from an adversarial classifier that only ever sees **gold-free**
input-side descriptors (`training/transfer_weighting.py`). Its sentinel is
`cache/rankers/transfer_stamp.json`.

**Expected result.** The GPU fine-tunes (stages 4/5/7) are seeded, but CUDA scheduling
makes them only approximately reproducible: a fresh retrain yields slightly different
encoders → slightly different OOF lists, matrices, and rankers. Measured end-to-end
variance of full rebuilds during development was ~±0.01 nDCG@20 on the recommendation
half. Consequently, after a full retrain `scripts/verify_inference.py` reports
*near*-but-not-exact matches to the submitted payload — that is the expected retriever
retrain variance, not a pipeline defect. The bit-level verification is path 1.

## Where each shipped artifact comes from

Every file in `weights_manifest.json` maps to a stage above: `cache/rankers/` ← stage 11;
`cache/retrievers/text_retriever/` ← stage 4; `blind_lists_blind_b.json` ← stages 5/7;
`cache/track_sim/*`, `cache/bm25/`, `metadata_maps.pkl`, `track_popularity.json` ← stage 1;
`als_factors.npz` ← stage 2; `cache/eval/dev_payload.pkl` ← stage 3; the OOF lists ←
stages 4/5/7; `cache/training/*` ← stages 8/9 + 11; `cache/blind_b/source_cache.pkl` ←
stage 6; `cache/gemini/` ← the submitted run's response cache (see INFERENCE.md §4b).
