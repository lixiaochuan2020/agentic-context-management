<h1 align="center">
ACM: Agentic Context Management for Long Horizon Tasks
</h1>

<div align="center">
Anonymous Authors (ICLR 2027 submission)
</div>

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](./LICENSE)

</div>

## Data & Checkpoints

The student rollouts, teacher logprob caches, and post-trained checkpoints are hosted on the Hugging Face Hub. The links are withheld during anonymous review and will be added upon publication.

## Intro

This repository contains the code for the anonymous submission "ACM: Agentic Context Management for Long-Horizon Tasks".

We propose a post-training pipeline that teaches a model to manage its own context on long-horizon agentic tasks. It has two key properties. First, it is **agent-native**: the agent decides when to compress context itself, instead of being forced by an external trigger. Second, it is **lossless**: discarded context is written to disk and kept permanently, so the agent can go back and retrieve any earlier information when it needs it.

## Pipeline

```
  1. Student Rollout   Qwen3.5-9B (ReAct mode) rolls out on the training set; results are graded to filter the data.
  2. Teacher Annotate  GPT-5.5 annotates the failed trajectories.
  3. Student Resume    The student resumes from each annotation over 4 trials; results are graded, keeping only questions it does not solve every time.
  4. Teacher Label     The teacher scores top-K logprobs on the kept traces.
  5. KD Train          Train the student on the teacher logprobs.
  6. Evaluation        Evaluate the trained student on the held-out set.
```

Because GPUs are limited, we run stages 4 and 5 separately: first serve the teacher and cache its top-K logprobs, then load that cache to train the student. Strictly speaking this is off-policy — once the student updates, its parameters no longer match the cached rollouts. With enough resources, the two stages can be merged into true on-policy distillation.

## Repository layout

```
src/                  agentic loop, tools, prompts, data loader, evaluator
  teacher_guide/      teacher annotation (annotate) + student resume (resume)
distill/              OPD: teacher logprob scoring + KD loss + KD trainer
train/                trajectory→sample preprocessing + servable-checkpoint merge
scripts/              pipeline_config.sh + the orchestrator + per-stage launchers
data/                 BrowseComp-Plus question sets
configs/              default run config
```

## Setup

Requires `Python ≥ 3.11`.

**1. Environment** (this is a [uv](https://docs.astral.sh/uv/) project):

```bash
uv sync
source .venv/bin/activate
```

**2. API keys** — copy the template and fill in the keys you use (OpenAI for the GPT-5.5 teacher and the GPT-5 grader, an HF token for downloads):

```bash
cp .env-example .env    # then edit
```

**3. BrowseComp-Plus corpus, indexes, and searcher code** — clone the upstream dataset repo and point `BCP_REPO` at it. It provides the BM25 Lucene index, the embedding shards, and the `searchers/bm25_searcher.py` that the runner loads at query time:

```bash
git clone https://huggingface.co/datasets/hltcoe/BrowseComp-Plus data/BrowseComp-Plus
```

BM25 retrieval uses [pyserini](https://github.com/castorini/pyserini), which needs a JDK — set `JAVA_HOME` (e.g. JDK 21) in `scripts/pipeline_config.sh`.

**4. Models** (pulled from public HF, not shipped here):

```bash
hf download Qwen/Qwen3.5-9B          # student policy (rollout / resume / eval)
hf download Qwen/Qwen3.5-397B-A17B   # OPD teacher (served locally)
hf download Qwen/Qwen3-Embedding-8B  # only if RETRIEVER=qwen3_8b_embedding
```

We serve Qwen3.5-397B-A17B in bf16 on 8×B200, or in fp8 on 8×H100.

For the KD init (`STUDENT_LM`), you may strip the 9B to language-model-only — keep the tensors whose names contain `language_model` or `lm_head` — to skip the vision load. By default `STUDENT_LM` just points at the full `Qwen/Qwen3.5-9B`.

## Data

The small question sets ship in `data/`: `bcp_train_680.json` (rollout pool), `bcp_eval_150.json` (held-out eval), and `bcp_full.json`.

The full pass@4 student rollouts (MemTool + ReAct, 4 runs each, ~4.4 GB) are too large for the repo, so they live on the Hugging Face Hub (see **Data & Checkpoints** above). Download them with:

```bash
hf download <anonymized>/acm-browsecompplus-train-rollouts-qwen3.5-9b --repo-type dataset --local-dir <dir>
```

## Running

All machine-specific paths and model ids live in **`scripts/pipeline_config.sh`** — edit it once (or export the same variables). Every script sources it. Retrieval defaults to local **BM25** (no extra server); set `RETRIEVER=qwen3_8b_embedding` to use the dense retriever instead.

Run the whole pipeline in one shot (smoke first, then full):

```bash
SMOKE=1 bash scripts/run_bcp_opd_pipeline.sh    # tiny end-to-end plumbing check
SMOKE=0 bash scripts/run_bcp_opd_pipeline.sh    # full run
```

On a SLURM cluster, add your scheduler header (`#SBATCH --partition/--qos/--nodelist --gres=gpu:8`) and `sbatch` it. Each stage also runs on its own via the per-stage scripts in `scripts/` (e.g. `run_teacher_guided_train680.sh`, `run_distill_score.sh`, `run_distill_train.sh`, `bcp_eval.sh`).
