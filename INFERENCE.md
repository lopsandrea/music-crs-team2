# Reproducing the submission (validation point 1)

Goal: starting from a **clean machine with nothing but this repository**, run
Blind-Dataset-B through our pipeline and obtain the submitted `prediction.json`
**byte-for-byte** (Codabench submission 827190).

Total time: **~30 minutes**, most of it downloads. A GPU is *not* required
(it is used automatically if present).

## 0. Requirements

- Linux/macOS, **Python 3.11**, [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- ~12 GB free disk (weights ~6 GB + Python env ~4 GB + HF datasets ~200 MB)
- ~8 GB RAM
- Network access to `huggingface.co` (all downloads are public, no account/token needed)
- A `GEMINI_API_KEY` env var. **For the exact replay below no API request is ever sent**
  (all 160 responses come from the shipped cache) — the variable just needs to be set
  because the client is constructed eagerly; any non-empty value works. A real key is
  only needed for step 5 (fresh generation).

## 1. Environment (~5 min)

```bash
git clone <this-repo> && cd music-crs-team2s2
uv sync          # creates .venv from the committed uv.lock (pinned versions)
```

## 2. Weights & artifacts (~6 GB, ~5 min)

```bash
uv run python scripts/download_weights.py
```

Downloads the artifact tree into `cache/` from the public HF repo recorded in
`weights_manifest.json` and verifies **every file's sha256** against the manifest.
Expected last line:

```
OK: 208 files verified against weights_manifest.json
```

(Re-check anytime with `--verify-only`.)

## 3. Official datasets (~150 MB — optional, they also auto-download)

```bash
uv run python scripts/download_data.py
```

Fetches `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` (catalog) and
`talkpl-ai/TalkPlayData-Challenge-Blind-B` (the 80 sessions to score) into the
HuggingFace cache. Skipping this step is fine — the pipeline downloads them on
first use.

## 4. Reproduce the submission

### 4a. Recommendation half only (no API key involved, ~5-10 min CPU)

```bash
uv run python scripts/verify_inference.py
```

Runs the full retrieval → fusion → ranking pipeline on all 80 Blind-B sessions and
compares the top-20 lists against the submitted payload. Expected:

```
matched 80/80 sessions against reference/prediction_827190.json
OK: recommendation half reproduces the submitted payload exactly.
```

### 4b. Full submission file, responses included (~10 min)

```bash
export GEMINI_API_KEY=anything        # never called on a full cache replay; see §0
uv run python run.py infer --blind blind_b --out submission.zip
```

Expected output ends with:

```
      80 cases | empty responses: 0
      prediction.json sha256 5cbb4216be1b9ba1… | zip 5248ffb08e6b830b… → submission.zip
```

Those are the submitted hashes: `submission.zip` is **byte-identical to Codabench
submission 827190** (the zip packaging is deterministic — fixed timestamps, stable
JSON). The responses come from `cache/gemini/`, the content-addressed response cache
of the submitted run (160 entries, keyed by model + prompt + history + item +
generation parameters), so zero Gemini requests are made.

## 5. (Optional) Generate fresh responses

```bash
rm -rf cache/gemini
export GEMINI_API_KEY=<real key>
uv run python run.py infer --blind blind_b --out submission_fresh.zip
```

With the cache gone, the two-pass generator (`gemini-3.1-pro-preview` draft →
`gemini-3.5-flash` polish, temperature 0.8) is called for real: ~160 requests. The
recommendation half stays byte-identical; the response *text* varies run to run by
nature of LLM sampling while following the same prompts/configuration.

## What makes this reproducible

- **Determinism of the recommendation half**: all retrieval indices are frozen on disk,
  scoring is pure numpy/GBDT, tie-breaks are stable sorts — same inputs, same top-20,
  always.
- **Replay of the response half**: LLM sampling is not deterministic, so we ship the
  exact response cache of the submitted run instead; the pipeline replays it
  transparently through its normal cache layer.
- **Integrity**: every artifact is sha256-pinned in `weights_manifest.json`; the target
  payload ships in `reference/prediction_827190.json`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `download_weights.py` fails mid-way | re-run it (downloads resume); then `--verify-only` |
| HF connection errors on first `infer` | run step 3 explicitly, or set `HF_HOME` to a writable dir |
| `verify_inference.py` mismatch | `scripts/download_weights.py --verify-only` (corrupted/partial weights are the only known cause) |
| slow first run | the text_retriever encodes 80 queries live; on CPU this is minutes, with a GPU seconds |
| `CUDA error: no kernel image is available for execution on the device` | `torch` is pinned to the CUDA 12.6 wheels (`pytorch-cu126`), which cover ~sm_50–sm_90 GPUs. A much newer card (Blackwell / RTX 50-series, needs `torch≥2.7` on the cu128 channel) or a very old one falls outside that range. Inference needs no GPU — just force CPU with `CUDA_VISIBLE_DEVICES="" uv run python run.py infer …`. For training on such a GPU, install a `torch` build matching its CUDA architecture. |
| `no Gemini API key found` | export `GEMINI_API_KEY` (any value for the replay; real key for fresh generation) |
