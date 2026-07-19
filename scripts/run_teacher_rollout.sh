#!/bin/bash
# Full teacher-guided rollout pipeline on bcp_100.
#
# Pipeline:
#   Phase 1: base Qwen3.5-9B + Qwen3-8B-Embedding + new tools → A_0 on bcp_100
#   Phase 2: grade A_0 with gpt-5
#   Phase 3: teacher (gpt-5.5) on wrong A_0 → iter1 rollout
#   Phase 4: grade iter1
#   Phase 5: teacher on wrong iter1 → iter2 rollout
#   Phase 6: grade iter2
#
# All vLLM endpoints stay up across all phases.
# Summarizer + query_memory use the student model itself (self-summarization)
# via the local agent vLLM endpoint. Teacher + grader are GPT-5.5 / GPT-5.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipeline_config.sh"
cd "$PROJECT_ROOT"

source "$VENV/bin/activate"
[[ -f .env ]] && { set -a; source .env; set +a; }
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ── Config ────────────────────────────────────────────────
AGENT_PATH="$BASE_MODEL"
AGENT_SERVED="$SERVED_POLICY"
AGENT_PORT=8003
AGENT_GPUS="0,1,2,3"

EMBED_PATH="$EMBED_MODEL"
EMBED_SERVED="$SERVED_EMBED"
EMBED_PORT=8005
EMBED_GPUS="4,5"

DATA="data/bcp_100.json"                       # 100 training questions
INDEX_PATH="$BM25_INDEX"
QWEN3_8B_EMBEDDING_SHARDS_DIR="$EMBED_SHARDS_DIR"

TAG="${TAG:-acm_qwen3_8b_emb_$(date +%Y%m%d)}"
RUN_DIR_A0="result_${TAG}"
RUN_DIR_ITER1="result_${TAG}_iter1"
RUN_DIR_ITER2="result_${TAG}_iter2"
RUN_ID="run_all"

NUM_SHARDS=16
# annotation/grader LLMs from config (TEACHER_MODEL there is the local OPD 397B — don't shadow it)
ANNOT_MODEL="${TEACHER_ANNOT_MODEL:-gpt-5.5}"
GRADER_MODEL="${GRADER_MODEL:-gpt-5}"
LIMIT="${LIMIT:-0}"  # 0 = all questions. Set env LIMIT=5 to smoke-test on 5.
LIMIT_FLAG=""
[[ "$LIMIT" -gt 0 ]] && LIMIT_FLAG="--limit $LIMIT"
# Self-summarization: summarizer = student model on the same vLLM endpoint
SUMMARIZER_MODEL="openai/${AGENT_SERVED}"
SUMMARIZER_API_BASE="http://localhost:${AGENT_PORT}/v1"

# Embedding endpoint env for the retriever
export BCP_RETRIEVER="qwen3_8b_embedding"
export BCP_QWEN3_8B_EMBEDDING_SHARDS_DIR="${QWEN3_8B_EMBEDDING_SHARDS_DIR}"
export EMBED_API_BASE="http://localhost:${EMBED_PORT}"
export EMBED_MODEL_NAME="${EMBED_SERVED}"
export AGENT_API_BASE="http://localhost:${AGENT_PORT}/v1"

LOG_DIR="logs/teacher_rollout_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

VLLM_PIDS=()

start_vllm_agent() {
    echo "── starting Qwen3.5-9B agent (TP=4) ──"
    CUDA_VISIBLE_DEVICES=$AGENT_GPUS vllm serve "$AGENT_PATH" \
        --tensor-parallel-size 4 \
        --served-model-name "$AGENT_SERVED" \
        --host 0.0.0.0 --port "$AGENT_PORT" \
        --max-model-len 262144 \
        --max-num-seqs 8 \
        --gpu-memory-utilization 0.85 \
        --enable-auto-tool-choice --tool-call-parser qwen3_xml \
        --trust-remote-code \
        > "$LOG_DIR/vllm_agent.log" 2>&1 &
    VLLM_PIDS+=($!)
}
start_vllm_embed() {
    echo "── starting Qwen3-8B-Embedding (TP=2) ──"
    CUDA_VISIBLE_DEVICES=$EMBED_GPUS vllm serve "$EMBED_PATH" \
        --runner pooling --convert embed \
        --tensor-parallel-size 2 \
        --served-model-name "$EMBED_SERVED" \
        --host 0.0.0.0 --port "$EMBED_PORT" \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.85 \
        --trust-remote-code \
        > "$LOG_DIR/vllm_embed.log" 2>&1 &
    VLLM_PIDS+=($!)
}
wait_ready() {
    local pt=$1 nm=$2
    echo -n "waiting for $nm on :$pt"
    for i in {1..120}; do
        if curl -fs "http://localhost:$pt/health" >/dev/null 2>&1; then
            echo " ready"; return 0
        fi
        echo -n .; sleep 5
    done
    echo " TIMEOUT"; return 1
}
cleanup() {
    echo "── cleanup vLLMs ──"
    for pid in "${VLLM_PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
    sleep 5
    for pid in "${VLLM_PIDS[@]:-}"; do kill -KILL "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT

start_vllm_agent
start_vllm_embed
wait_ready "$AGENT_PORT" "agent"
wait_ready "$EMBED_PORT" "embed"

# ────────────────────────────────────────────────────────────
# Phase 1: A_0 — base agent on bcp_100
# ────────────────────────────────────────────────────────────
echo
echo "══ Phase 1: A_0 rollout (16 shards on bcp_100) ══"
A0_PIDS=()
for ((s=0; s<NUM_SHARDS; s++)); do
    python -m src.run --mode run --benchmark browsecomp-plus --client litellm \
        --model "openai/$AGENT_SERVED" --agent_api_base "$AGENT_API_BASE" \
        --data "$DATA" --index_path "$INDEX_PATH" \
        --run_dir "$RUN_DIR_A0" --run_id "$RUN_ID" \
        --shard "$s" --num_shards "$NUM_SHARDS" \
        $LIMIT_FLAG \
        --use_memory_tool \
        --summarizer_model "$SUMMARIZER_MODEL" \
        --summarizer_api_base "$SUMMARIZER_API_BASE" \
        --log_level INFO \
        > "$LOG_DIR/a0_shard_${s}.log" 2>&1 &
    A0_PIDS+=($!)
done
wait "${A0_PIDS[@]}" || echo "[warn] some A_0 shards exited non-zero"

A0_OUT_DIR="results/browsecomp-plus/${AGENT_SERVED}/${RUN_DIR_A0}/${RUN_ID}"
A0_EVAL_DIR="results/browsecomp-plus/${AGENT_SERVED}/eval_bcp/${RUN_DIR_A0}/${RUN_ID}"
echo "Phase 1 done. A_0 trajectories: $A0_OUT_DIR"

# ────────────────────────────────────────────────────────────
# Phase 2: grade A_0
# ────────────────────────────────────────────────────────────
echo
echo "══ Phase 2: grade A_0 ══"
python scripts/grade_bcp_gpt5.py --input_dir "$A0_OUT_DIR" --eval_dir "$A0_EVAL_DIR" \
    --model "$GRADER_MODEL" --workers 16 \
    2>&1 | tee "$LOG_DIR/grade_a0.log"

# ────────────────────────────────────────────────────────────
# Phase 3: iter1 — teacher on wrong A_0
# ────────────────────────────────────────────────────────────
echo
echo "══ Phase 3: iter1 (teacher-guided continuation on wrong A_0) ══"
ITER1_OUT_DIR="results/browsecomp-plus/${AGENT_SERVED}/${RUN_DIR_ITER1}/${RUN_ID}"
python -m src.teacher_guided_rollout \
    --run_dir "$A0_OUT_DIR" --eval_dir "$A0_EVAL_DIR" \
    --out_run_dir "$ITER1_OUT_DIR" \
    --teacher_model "$ANNOT_MODEL" \
    --student_model "openai/$AGENT_SERVED" \
    --index_path "$INDEX_PATH" \
    --src_workspace_root "$A0_OUT_DIR/workspace" \
    --summarizer_model "$SUMMARIZER_MODEL" \
    --summarizer_api_base "$SUMMARIZER_API_BASE" \
    $LIMIT_FLAG \
    --log_level INFO \
    2>&1 | tee "$LOG_DIR/iter1.log"

# ────────────────────────────────────────────────────────────
# Phase 4: grade iter1 (skip if no iter1 outputs)
# ────────────────────────────────────────────────────────────
ITER1_EVAL_DIR="results/browsecomp-plus/${AGENT_SERVED}/eval_bcp/${RUN_DIR_ITER1}/${RUN_ID}"
if compgen -G "$ITER1_OUT_DIR/run_*.json" > /dev/null; then
    echo
    echo "══ Phase 4: grade iter1 ══"
    python scripts/grade_bcp_gpt5.py --input_dir "$ITER1_OUT_DIR" --eval_dir "$ITER1_EVAL_DIR" \
        --model "$GRADER_MODEL" --workers 16 \
        2>&1 | tee "$LOG_DIR/grade_iter1.log"

    # ────────────────────────────────────────────────────────────
    # Phase 5: iter2 — teacher on wrong iter1
    # ────────────────────────────────────────────────────────────
    echo
    echo "══ Phase 5: iter2 (teacher on wrong iter1) ══"
    ITER2_OUT_DIR="results/browsecomp-plus/${AGENT_SERVED}/${RUN_DIR_ITER2}/${RUN_ID}"
    python -m src.teacher_guided_rollout \
        --run_dir "$ITER1_OUT_DIR" --eval_dir "$ITER1_EVAL_DIR" \
        --out_run_dir "$ITER2_OUT_DIR" \
        --teacher_model "$ANNOT_MODEL" \
        --student_model "openai/$AGENT_SERVED" \
        --index_path "$INDEX_PATH" \
        --src_workspace_root "$ITER1_OUT_DIR/workspace" \
        --summarizer_model "$SUMMARIZER_MODEL" \
        --summarizer_api_base "$SUMMARIZER_API_BASE" \
        $LIMIT_FLAG \
        --log_level INFO \
        2>&1 | tee "$LOG_DIR/iter2.log"

    # ────────────────────────────────────────────────────────────
    # Phase 6: grade iter2 (skip if no iter2 outputs)
    # ────────────────────────────────────────────────────────────
    if compgen -G "$ITER2_OUT_DIR/run_*.json" > /dev/null; then
        echo
        echo "══ Phase 6: grade iter2 ══"
        ITER2_EVAL_DIR="results/browsecomp-plus/${AGENT_SERVED}/eval_bcp/${RUN_DIR_ITER2}/${RUN_ID}"
        python scripts/grade_bcp_gpt5.py --input_dir "$ITER2_OUT_DIR" --eval_dir "$ITER2_EVAL_DIR" \
            --model "$GRADER_MODEL" --workers 16 \
            2>&1 | tee "$LOG_DIR/grade_iter2.log"
    else
        echo "── iter2 produced no trajectories — skipping Phase 6 ──"
    fi
else
    echo "── iter1 produced no trajectories (A_0 had no wrong questions) — skipping Phases 4-6 ──"
fi

echo
echo "══ All phases done ══"
echo "A_0    : $A0_OUT_DIR"
echo "iter1  : $ITER1_OUT_DIR"
echo "iter2  : $ITER2_OUT_DIR"
echo "log dir: $LOG_DIR"
