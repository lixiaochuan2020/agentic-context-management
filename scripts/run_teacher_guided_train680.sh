#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Teacher-guided trajectory generation on the training pool (one job)
# ══════════════════════════════════════════════════════════════════════════════
# Produces the teacher-guided student trajectories that feed OPD distillation.
# Three stages run sequentially in a single allocation; GPUs are freed between
# the two GPU-heavy stages so the policy can be re-served under a different tool
# schema.
#
#   Stage 1  initial rollout   — base 9B in pure 2-tool ReAct (search + get_document,
#                            NO memory tools) on the train pool, then GPT-5 grade.
#   Stage 2  teacher annotate — GPT-5 annotates the FAILED initial trajectories
#                            (fixes the wrong reasoning). CPU / API only, no GPU.
#   Stage 3  resume        — student replays each annotation under the 4-tool
#                            MemTool schema (manage_context call message dropped
#                            from live history), then GPT-5 grade.
#
# This is the standalone version of Stages 1–3 of run_bcp_opd_pipeline.sh; the
# outputs (tags below) are consumable by distill/score_teacher_logprobs.py.
#
# Configure paths/models in scripts/pipeline_config.sh (or export overrides).
# Retrieval defaults to local BM25; set RETRIEVER=qwen3_8b_embedding for dense.
#
# Usage (run on the GPU node — NOT the login node):
#   SMOKE=1 bash scripts/run_teacher_guided_train680.sh      # tiny plumbing check
#   SMOKE=0 bash scripts/run_teacher_guided_train680.sh      # full run
#
# SLURM users: add your own #SBATCH header (partition/qos/nodelist/--gres=gpu:N)
# and submit with sbatch. The body below is scheduler-agnostic.

set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipeline_config.sh"
cd "$PROJECT_ROOT"
source "$VENV/bin/activate"
[[ -f .env ]] && { set -a; source .env; set +a; }
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ── SMOKE vs FULL ─────────────────────────────────────────────────────────────
SMOKE="${SMOKE:-1}"
N_GPUS="${N_GPUS:-8}"                       # 1 vLLM replica per GPU (TP=1)
RESUME_REPS="${RESUME_REPS:-1}"             # resume passes (pass@N for downstream filtering)
MAX_WORKERS="${MAX_WORKERS:-48}"            # resume thread pool
if [[ "$SMOKE" == "1" ]]; then
  DATA="$TRAIN_DATA"; ROLL_SHARDS=4; ANNOT_LIMIT=8
else
  DATA="$TRAIN_DATA"; ROLL_SHARDS=64; ANNOT_LIMIT=0
fi

POLICY_PORTS=(); for ((i=0; i<N_GPUS; i++)); do POLICY_PORTS+=($((8001 + i))); done

INIT_TAG="init-react"
RESUME_TAG="resume-teacher-guided"
ANNOT_DIR="src/teacher_guide/results/teacher-guided/annotations"
POLICY_ROOT="$RESULTS_DIR/browsecomp-plus/$SERVED_POLICY"

LD="logs/teacher-guided-$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LD"
{ echo "host=$(hostname) start=$(date -Is) SMOKE=$SMOKE N_GPUS=$N_GPUS";
  echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)";
  echo "RETRIEVER=$RETRIEVER BASE_MODEL=$BASE_MODEL DATA=$DATA";
} | tee "$LD/config.txt"

# ── retrieval flags shared by every src.run / resume invocation ────────────────
RETR_ARGS=(--index_path "$BM25_INDEX")
if [[ "$RETRIEVER" == "qwen3_8b_embedding" ]]; then
  export BCP_RETRIEVER="qwen3_8b_embedding"
  export BCP_QWEN3_8B_EMBEDDING_SHARDS_DIR="$EMBED_SHARDS_DIR"
  export EMBED_API_BASE EMBED_MODEL_NAME="$SERVED_EMBED"
fi
[[ -n "${JAVA_HOME:-}" ]] && export PATH="$JAVA_HOME/bin:$PATH"

# ── sanity: data present ───────────────────────────────────────────────────────
[[ -f "$DATA" ]] || { echo "ERROR: data file not found: $DATA"; exit 1; }

# ── helpers ────────────────────────────────────────────────────────────────────
SERVE_PIDS=()
wait_health(){ local p=$1 n=$2 t=${3:-240}; echo -n "wait $n :$p"; for i in $(seq 1 "$t"); do
  curl -fs "http://localhost:$p/health" >/dev/null 2>&1 && { echo " UP"; return 0; }; echo -n .; sleep 10; done; echo " TIMEOUT"; return 1; }
free_gpus(){
  for x in "${SERVE_PIDS[@]:-}"; do kill -TERM "$x" 2>/dev/null || true; done
  sleep 5; pkill -KILL -f "VLLM::EngineCore" 2>/dev/null || true; pkill -KILL -f "VLLM::Worker" 2>/dev/null || true
  pkill -KILL -f "vllm serve" 2>/dev/null || true; SERVE_PIDS=()
  echo -n "  freeing GPUs"; for i in $(seq 1 24); do
    local mx; mx=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ "${mx:-99999}" -lt 2000 ] 2>/dev/null && { echo " ok (max ${mx}MB used)"; return 0; }; echo -n .; sleep 10; done; echo; }
serve_policy(){  # $1 = model path, $2 = served name — one TP=1 replica per GPU
  local model=$1 served=$2 i p
  for i in "${!POLICY_PORTS[@]}"; do
    p=${POLICY_PORTS[$i]}
    CUDA_VISIBLE_DEVICES=$i vllm serve "$model" --tensor-parallel-size 1 --served-model-name "$served" \
      --host 0.0.0.0 --port "$p" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 16 \
      --gpu-memory-utilization "$GPU_MEM_UTIL" --enable-auto-tool-choice --tool-call-parser qwen3_xml \
      --trust-remote-code > "$LD/vllm_${served}_$p.log" 2>&1 &
    SERVE_PIDS+=($!)
  done
  for p in "${POLICY_PORTS[@]}"; do wait_health "$p" "$served" || { echo "ERROR: $served :$p"; exit 1; }; done
}
cleanup(){ free_gpus >/dev/null 2>&1 || true; }
trap cleanup EXIT

# ══════════════════════════════════════════════════════════════════════════════
echo "════ STAGE 1: initial rollout (base 9B, pure ReAct, ${MAX_MODEL_LEN}/${MAX_ITER}t) ════"
serve_policy "$BASE_MODEL" "$SERVED_POLICY"
pids=()
for ((SH=0; SH<ROLL_SHARDS; SH++)); do
  port=${POLICY_PORTS[$((SH % N_GPUS))]}
  python -m src.run --mode run --benchmark browsecomp-plus --client litellm \
    --model "openai/$SERVED_POLICY" --agent_api_base "http://localhost:$port/v1" \
    --data "$DATA" --results_dir "$RESULTS_DIR" --run_dir "$INIT_TAG" --run_id "run_all" \
    --shard "$SH" --num_shards "$ROLL_SHARDS" \
    --summarizer_model "openai/$SERVED_POLICY" --summarizer_api_base "http://localhost:$port/v1" \
    "${RETR_ARGS[@]}" \
    --override runtime.use_memory_tools=false agent.context_window=$MAX_MODEL_LEN runtime.max_iterations=$MAX_ITER \
    --log_level INFO > "$LD/init_sh${SH}.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}" || echo "  some initial shards non-zero (see $LD/init_sh*.log)"
INIT_DIR="$POLICY_ROOT/$INIT_TAG/run_all"
INIT_GRADE="$POLICY_ROOT/eval_bcp/$INIT_TAG/run_all"
python scripts/grade_bcp_gpt5.py --input_dir "$INIT_DIR" --eval_dir "$INIT_GRADE" --workers 16 2>&1 | tee "$LD/init_grade.log"
free_gpus

echo "════ STAGE 2: teacher annotate FAILED initial trajectories (${TEACHER_ANNOT_MODEL}) ════"
mkdir -p "$ANNOT_DIR"
ANNOT_ARGS=(); [ "$ANNOT_LIMIT" -gt 0 ] && ANNOT_ARGS+=(--limit "$ANNOT_LIMIT")
python -m src.teacher_guide.run \
  --run_dir "$INIT_DIR" --eval_dir "$INIT_GRADE" --out_dir "$ANNOT_DIR" \
  --teacher_model "$TEACHER_ANNOT_MODEL" --teacher_api_base "$TEACHER_ANNOT_API_BASE" \
  --max_workers 4 --skip_existing "${ANNOT_ARGS[@]}" 2>&1 | tee "$LD/annotate.log"
[ "$(ls "$ANNOT_DIR" 2>/dev/null | grep -c '\.json$')" -gt 0 ] || { echo "ERROR: no annotations written"; exit 1; }

echo "════ STAGE 3: resume from annotations under MemTool (mc_drop_call_message=true), pass@${RESUME_REPS} ════"
serve_policy "$BASE_MODEL" "$SERVED_MEMTOOL"
AGENT_BASES=$(printf "http://localhost:%s/v1," "${POLICY_PORTS[@]}"); AGENT_BASES=${AGENT_BASES%,}
for REP in $(seq 1 "$RESUME_REPS"); do
  echo "── resume attempt $REP/$RESUME_REPS ──"
  RTAG="${RESUME_TAG}-run${REP}"; ROUT="$POLICY_ROOT/$RTAG/run_all"
  RES_ARGS=(); [ "$ANNOT_LIMIT" -gt 0 ] && RES_ARGS+=(--limit "$ANNOT_LIMIT")
  python -m src.teacher_guide.resume \
    --annotations_dir "$ANNOT_DIR" --rollouts_dir "$INIT_DIR" --out_dir "$ROUT" \
    --student_model "openai/$SERVED_MEMTOOL" --agent_api_bases "$AGENT_BASES" \
    --summarizer_model "openai/$SERVED_MEMTOOL" --summarizer_api_base "http://localhost:${POLICY_PORTS[0]}/v1" \
    --index_path "$BM25_INDEX" --max_workers "$MAX_WORKERS" --skip_existing \
    --override runtime.mc_drop_call_message=true \
    "${RES_ARGS[@]}" --log_level INFO > "$LD/resume_r${REP}.log" 2>&1
  python scripts/grade_bcp_gpt5.py --input_dir "$ROUT" \
    --eval_dir "$POLICY_ROOT/eval_bcp/$RTAG/run_all" --workers 16 2>&1 | tail -6 | tee "$LD/resume_grade_r${REP}.log"
done
free_gpus

echo
echo "════ DONE $(date -Is) ════"
echo "  initial rollout : $INIT_DIR"
echo "  annotations : $ANNOT_DIR"
echo "  resumed     : $POLICY_ROOT/${RESUME_TAG}-run*/run_all"
echo "  logs        : $LD"
echo
echo "Next: teacher-score the kept traces for KD, e.g."
echo "  python -m distill.score_teacher_logprobs --runs_root $POLICY_ROOT \\"
echo "      --run_tag ${RESUME_TAG}-run --reps $(seq -s' ' 1 "$RESUME_REPS") --keep_solve_counts 0 1 2 3 ..."
