# Music-CRS — team2_s2 (ACM RecSys Challenge 2026)

Validation release for team **team2_s2**. This repository reproduces our Codabench
**Blind-B submission 827190** end-to-end and rebuilds every model behind it from scratch.

## For reviewers — start here

| Validation point | Guide | Expected result | Effort |
|---|---|---|---|
| **1. Inference** — Blind-Dataset-B → pipeline → `prediction.json` | **[INFERENCE.md](INFERENCE.md)** | the submitted `prediction.json`, **byte-identical** (sha256 `5cbb4216be1b9ba1…`) | ~30 min, CPU is enough |
| **2. Reproducibility** — retrain the weights | **[TRAINING.md](TRAINING.md)** | two paths: **exact reproduction** of the shipped rankers (verified against the submission), or a **full retrain from scratch** of every artifact from official data | ~20 min / ~1 GPU day |

Both guides assume a clean machine and list every command, expected output, and duration.

## What this system is

A conversational music recommender with two independent halves per session:

1. **Recommendation half** (`recommender/`, deterministic — scored by nDCG@20):
   - **L1 candidate sources** over the full 47,071-track catalog: lexical BM25 ×2,
     semantic-metadata Qwen3 ×2, collaborative CF-BPR + ALS, acoustic CLAP, and three
     retrievers fine-tuned on the challenge conversations — `text_retriever` (bi-encoder,
     runs live) and `struct_base`/`struct_large` (5-fold BGE ensembles, precomputed lists);
   - **L2 fusion**: weighted Reciprocal Rank Fusion → 300-candidate pool;
   - **L3 features**: a 300×37 matrix per session (`recommender/features.py`);
   - **L4-L6 ranking**: two LightGBM LambdaRank boosters with margin-based routing,
     blended in z-score space with a CatBoost-YetiRank ensemble → top-20.
2. **Response half** (`response_gen/`, LLM — scored by an LLM judge): a two-pass Gemini
   generator (`gemini-3.1-pro-preview` draft → `gemini-3.5-flash` polish) with track-fact
   grounding. The exact submitted responses replay from a shipped content-addressed cache;
   fresh generations require a Gemini API key.

Candidate retrieval always operates over the **entire `all_tracks` catalog** — no
`track_split_types` or any other catalog restriction exists at any stage (training,
inference, or post-processing).

**→ [PIPELINE.md](PIPELINE.md)** documents the whole system in depth, level by level,
with architecture diagrams (sources map, fusion, feature layout, ranker routing, the
training DAG and the covariate-shift correction).

## Repository layout

```
README.md                  ← you are here (map of the project)
INFERENCE.md               ← validation point 1: reproduce the submission
TRAINING.md                ← validation point 2: rebuild everything from scratch
PIPELINE.md                ← in-depth architecture documentation, with diagrams
run.py                     CLI: `infer` (blind → submission.zip) and `train-all`
pipeline.py                blind sessions → recommender → response_gen → deterministic zip
recommender/               retrieval + fusion + features + GBDT ranking (paths in config.py)
response_gen/              Gemini response generation (prompts/ = the shipped variant)
training/                  the 11-stage from-scratch build DAG (see TRAINING.md)
scripts/
  download_weights.py      fetch + sha-verify the weights/artifact set (~6 GB)
  download_data.py         pre-fetch the official datasets (+ base encoders with --training)
  verify_inference.py      assert the recommendation half matches the submission 80/80
  upload_weights.py        (maintainers) manifest-driven weights upload
reference/
  prediction_827190.json   the submitted payload, verbatim — ground truth for verification
weights_manifest.json      sha256 + size of every artifact file (208 entries)
cache/                     the artifact tree (downloaded; git-ignored)
```

## What gets downloaded, from where

Everything external is public and fetched by script (or automatically on first use):

| Material | Size | Source | How |
|---|---|---|---|
| **Model weights & artifacts** (rankers, fine-tuned retriever, embedding/BM25 indices, precomputed Blind-B retriever lists, training intermediates, the submitted run's response cache) | ~6 GB | HF dataset repo [`lopsandrea/music-crs-team2s2-weights`](https://huggingface.co/datasets/lopsandrea/music-crs-team2s2-weights) | `scripts/download_weights.py` → sha-verified against `weights_manifest.json` |
| **Official challenge datasets** — Track-Metadata (catalog), Blind-B (sessions to score) | ~150 MB | HF `talkpl-ai/…` | auto on first run, or `scripts/download_data.py` |
| **Training data** (only for TRAINING.md) — Challenge-Dataset (conversations), Track-Embeddings (precomputed qwen3/cf-bpr/CLAP columns), Blind-A | ~2.3 GB | HF `talkpl-ai/…` | auto, or `scripts/download_data.py --training` |
| **Base encoder checkpoints** (only for TRAINING.md) — `BAAI/bge-base-en-v1.5`, `BAAI/bge-large-en-v1.5` | ~1.7 GB | HF | auto, or `scripts/download_data.py --training` |
| **Gemini API** (only to generate *fresh* responses; the shipped cache replays the submission without any API call) | — | Google | `GEMINI_API_KEY` env var |

No other external material of any kind is used. All models are trained exclusively on the
official challenge datasets plus the two public BGE checkpoints (TRAINING.md documents this
per stage).

## Verification anchors

- `reference/prediction_827190.json` — the payload of Codabench submission **827190**
  (account `team2_s2`), sha256 `5cbb4216be1b9ba1…`; its deterministic zip is
  `5248ffb08e6b830b…`.
- `weights_manifest.json` — sha256 of all 208 artifact files; `download_weights.py`
  verifies every file after download (`--verify-only` re-checks anytime).

## License

MIT — see `LICENSE`.
