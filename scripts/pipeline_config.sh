#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  pipeline_config.sh — single source of truth for machine-specific paths & models
# ══════════════════════════════════════════════════════════════════════════════
# Every script in scripts/ sources this file. Edit the values below (or export the
# same variables in your shell / SLURM job) to point at your environment. Nothing
# here is secret — API keys live in .env (see .env-example).
#
# All values use ${VAR:-default} so an exported environment variable always wins,
# letting you override any single path per-run without editing this file.

# ── Repo + environment ────────────────────────────────────────────────────────
# PROJECT_ROOT defaults to the repo root (parent of this scripts/ dir).
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Python virtualenv to activate on the compute node (uv-managed by default).
VENV="${VENV:-$PROJECT_ROOT/.venv}"
# Hugging Face cache (models + datasets). Point at fast local storage.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Models (HF repo ids or local paths) ───────────────────────────────────────
# Student policy served for rollout / resume / eval (full VLM checkpoint).
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
# KD init for train_kd — the LM-only student (strip the vision tower if present).
STUDENT_LM="${STUDENT_LM:-Qwen/Qwen3.5-9B}"
# OPD teacher scored for logprobs. A LOCAL path is strongly recommended (397B).
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-397B-A17B}"
# Optional slower/NFS copy the teacher is staged FROM onto local NVMe (may be empty).
TEACHER_MODEL_SRC="${TEACHER_MODEL_SRC:-}"
# Embedding model for the qwen3_8b_embedding retriever (only if RETRIEVER=qwen3_8b_embedding).
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}"

# Served-model names (the --served-model-name handed to vLLM / used as the API model).
SERVED_POLICY="${SERVED_POLICY:-qwen3.5-9b-base}"
SERVED_MEMTOOL="${SERVED_MEMTOOL:-qwen3.5-9b-mem_tool}"
SERVED_TEACHER="${SERVED_TEACHER:-qwen3.5-397b-teacher}"
SERVED_STUDENT="${SERVED_STUDENT:-qwen3.5-9b-bcp-opd}"
SERVED_EMBED="${SERVED_EMBED:-qwen3-embedding-8b}"

# ── Data + retrieval ──────────────────────────────────────────────────────────
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/bcp_train_680.json}"   # pass@4 rollout pool
EVAL_DATA="${EVAL_DATA:-$PROJECT_ROOT/data/bcp_eval_150.json}"      # held-out eval
# Retriever backend: "bm25" (local, no server — default) | "qwen3_8b_embedding" (needs embed server).
RETRIEVER="${RETRIEVER:-bm25}"
# BrowseComp-Plus corpus/indexes — clone the upstream repo (see README) and point here.
BCP_REPO="${BCP_REPO:-$PROJECT_ROOT/data/BrowseComp-Plus}"
BM25_INDEX="${BM25_INDEX:-$BCP_REPO/indexes/bm25}"                  # Lucene dir (pyserini)
EMBED_SHARDS_DIR="${EMBED_SHARDS_DIR:-$BCP_REPO/indexes/qwen3-embedding-8b}"
# JAVA_HOME is required for BM25 retrieval (pyserini needs a JDK, e.g. JDK 21).
export JAVA_HOME="${JAVA_HOME:-}"
# Embed server URL (only used when RETRIEVER=qwen3_8b_embedding).
EMBED_API_BASE="${EMBED_API_BASE:-http://localhost:8765}"

# ── Output roots (large; put on fast/local storage, NOT in the repo) ───────────
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"                 # rollout / eval trajectories
CKPT_DIR="${CKPT_DIR:-/scratch/$USER/checkpoints}"                 # trained + merged checkpoints
LOGPROBS_DIR="${LOGPROBS_DIR:-/scratch/$USER/distill/teacher_logprobs}"  # cached teacher npz

# ── Teacher-annotation + grading LLMs (OpenAI-compatible endpoints) ────────────
TEACHER_ANNOT_MODEL="${TEACHER_ANNOT_MODEL:-gpt-5}"                 # annotates failed initial trajectories
TEACHER_ANNOT_API_BASE="${TEACHER_ANNOT_API_BASE:-https://api.openai.com/v1}"
GRADER_MODEL="${GRADER_MODEL:-gpt-5}"                               # BCP answer judge

# ── Serving defaults ──────────────────────────────────────────────────────────
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_ITER="${MAX_ITER:-100}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
