"""Offline rebuild package: regenerate every cached artifact the recommender consumes.

Each submodule is one stage exposing ``build(force=False, smoke=False)`` and is run, in
dependency order, by ``training.train_all.train_all`` (entry point ``run.py train-all``).
Stages are idempotent — a stage no-ops when its outputs already exist unless ``force`` —
so with the shipped artifacts in place this whole package is a no-op. Stages cover the base
caches (BM25, qwen3 / cf-bpr / CLAP vectors), ALS factors, the fine-tuned retrievers
(text_retriever / struct_base / struct_large), the dev/blind payload caches and per-case
feature matrices, and finally the LightGBM and CatBoost rankers. The heavy retriever stages
are GPU code; see ``training.train_all`` for the full DAG and ``TRAINING.md``.
"""
