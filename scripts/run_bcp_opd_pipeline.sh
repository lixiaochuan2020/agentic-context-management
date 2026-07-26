#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  BrowseComp-Plus OPD — teacher-guided on-policy distillation, one job (8 GPUs)
# ══════════════════════════════════════════════════════════════════════════════
# Combines teacher ANNOTATION (GPT-5 fixes failed trajectories) with OPD teacher
# DISTILLATION (397B logprob KD). The six stages run sequentially in a single
# allocation; GPU memory is freed between GPU-heavy stages.
#
#   Stage 1  student rollout  — base 9B ReAct initial on train pool, grade       [policy ×8]
#   Stage 2  teacher annotate — GPT-5 annotates the FAILED initial trajectories  [CPU / API]
#   Stage 3  student resume   — student resumes from annotations, pass@N,    [policy ×8]
#            grade → keep the trajectories it now gets right (pass@4 filter)
#   Stage 4  teacher label    — 397B scores top-K logprobs on kept traces    [teacher TP=8]
#   Stage 5  KD train         — distill the student from the teacher logprobs [8 GPU]
#   Stage 6  eval             — merge → serve student → eval held-out → grade [student ×8]
#
# Configure paths/models in scripts/pipeline_config.sh (or export overrides).
#
# SMOKE=1 (default): tiny end-to-end plumbing check on a handful of questions.
# SMOKE=0          : full run.
# Usage (do NOT run login-node; submit or run on the 8-GPU node):
#   SMOKE=1 bash scripts/run_bcp_opd_pipeline.sh      # smoke first
#   SMOKE=0 bash scripts/run_bcp_opd_pipeline.sh      # full, after smoke is clean
#
# SLURM users: add your own #SBATCH header (partition/qos/nodelist/--gres=gpu:8)
# and submit with sbatch. The body below is scheduler-agnostic.

set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipeline_config.sh"
cd "$PROJECT_ROOT"
source "$VENV/bin/activate"
[[ -f .env ]] && { set -a; source .env; set +a; }
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ── SMOKE vs FULL ─────────────────────────────────────────────────────────────
SMOKE="${SMOKE:-1}"
TOP_K="${TOP_K:-20}"; N_POLICY="${N_POLICY:-8}"
POLICY_PORTS=(8001 8002 8003 8004 8005 8006 8007 8008); TEACHER_PORT=8900
if [[ "$SMOKE" == "1" ]]; then
  SFX="-smoke"; DATA="$TRAIN_DATA"; EVAL="$EVAL_DATA"
  ROLL_SHARDS=4; RESUME_REPS=2; EVAL_SHARDS=4; EVAL_REPS=1
  SCORE_SHARDS=2; TEACHER_LIMIT=4; ANNOT_LIMIT=8
  TRAIN_MAXLEN=65536; TRAIN_EXTRA="--max_samples 20 --max_steps 5"; TRAIN_SP=1
else
  SFX=""; DATA="$TRAIN_DATA"; EVAL="$EVAL_DATA"
  ROLL_SHARDS=64; RESUME_REPS=4; EVAL_SHARDS=64; EVAL_REPS=4
  SCORE_SHARDS=8; TEACHER_LIMIT=0; ANNOT_LIMIT=0
  TRAIN_MAXLEN=131072; TRAIN_EXTRA="--epochs 3"; TRAIN_SP=1
fi
TRAIN_DP=$((N_POLICY / TRAIN_SP))

INIT_TAG="init-react${SFX}"
RESUME_TAG="resume-teacher-guided${SFX}"
EVAL_TAG="eval-bcp-opd${SFX}"
ANNOT_DIR="src/teacher_guide/results/opd${SFX}/annotations"
SCORE_OUT="$LOGPROBS_DIR/bcp_teacher_logprobs${SFX}"
OUT_CKPT="$CKPT_DIR/acm-bcp-opd${SFX}"
SERVABLE="${OUT_CKPT}-servable"
POLICY_ROOT="$RESULTS_DIR/browsecomp-plus/$SERVED_POLICY"
STUDENT_ROOT="$RESULTS_DIR/browsecomp-plus/$SERVED_STUDENT"

LD="logs/bcp-opd${SFX}-$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LD/run_metadata"
{ echo "host=$(hostname) start=$(date -Is) SMOKE=$SMOKE"; echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)";
  echo "RETRIEVER=$RETRIEVER BASE_MODEL=$BASE_MODEL TEACHER_MODEL=$TEACHER_MODEL";
  echo "DATA=$DATA EVAL=$EVAL RESUME_REPS=$RESUME_REPS TRAIN_MAXLEN=$TRAIN_MAXLEN";
  echo "SCORE_OUT=$SCORE_OUT OUT_CKPT=$OUT_CKPT";
} | tee "$LD/run_metadata/config.txt"; env | sort > "$LD/run_metadata/env.txt"

# ── retrieval flags shared by every src.run / resume invocation ────────────────
RETR_ARGS=(--index_path "$BM25_INDEX")
if [[ "$RETRIEVER" == "qwen3_8b_embedding" ]]; then
  export BCP_RETRIEVER="qwen3_8b_embedding"
  export BCP_QWEN3_8B_EMBEDDING_SHARDS_DIR="$EMBED_SHARDS_DIR"
  export EMBED_API_BASE EMBED_MODEL_NAME="$SERVED_EMBED"
fi
[[ -n "${JAVA_HOME:-}" ]] && export PATH="$JAVA_HOME/bin:$PATH"

# ── helpers ───────────────────────────────────────────────────────────────────
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
serve_policy(){  # $1 = model path, $2 = served name
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
echo "════ STAGE 1: student rollout (ReAct initial, ${MAX_MODEL_LEN}/${MAX_ITER}t) ════"
serve_policy "$BASE_MODEL" "$SERVED_POLICY"
pids=()
for ((SH=0; SH<ROLL_SHARDS; SH++)); do
  port=${POLICY_PORTS[$((SH % N_POLICY))]}
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
wait "${pids[@]}" || echo "  some initial shards non-zero"
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

echo "════ STAGE 3: student resume from annotations, pass@${RESUME_REPS} + grade ════"
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
    --index_path "$BM25_INDEX" --max_workers 48 --skip_existing \
    --override runtime.mc_drop_call_message=true \
    "${RES_ARGS[@]}" --log_level INFO > "$LD/resume_r${REP}.log" 2>&1
  python scripts/grade_bcp_gpt5.py --input_dir "$ROUT" \
    --eval_dir "$POLICY_ROOT/eval_bcp/$RTAG/run_all" --workers 16 2>&1 | tail -6 | tee "$LD/resume_grade_r${REP}.log"
done
free_gpus

echo "════ STAGE 4: teacher-label kept traces (397B TP=8, top-$TOP_K, keep pass@4 solve-counts 0–3) ════"
# stage teacher onto local NVMe if a slower source copy is configured
if [[ -n "$TEACHER_MODEL_SRC" && "$(ls "$TEACHER_MODEL"/*.safetensors 2>/dev/null | wc -l)" -lt 1 ]]; then
  echo "  staging teacher from $TEACHER_MODEL_SRC → $TEACHER_MODEL"; mkdir -p "$TEACHER_MODEL"
  rsync -a "$TEACHER_MODEL_SRC/" "$TEACHER_MODEL/" 2>&1 | tail -2
fi
mkdir -p "$SCORE_OUT"
vllm serve "$TEACHER_MODEL" --tensor-parallel-size 8 --served-model-name "$SERVED_TEACHER" \
  --host 0.0.0.0 --port "$TEACHER_PORT" --max-model-len 262144 --max-num-seqs "$SCORE_SHARDS" \
  --enable-chunked-prefill --max-num-batched-tokens 8192 --disable-custom-all-reduce \
  --gpu-memory-utilization 0.80 --trust-remote-code > "$LD/vllm_teacher.log" 2>&1 &
SERVE_PIDS+=($!)
wait_health "$TEACHER_PORT" "teacher" 300 || { echo "ERROR: teacher serve failed"; tail -40 "$LD/vllm_teacher.log"; exit 1; }
pids=()
for SH in $(seq 0 $((SCORE_SHARDS-1))); do
  python -m distill.score_teacher_logprobs \
    --runs_root "$POLICY_ROOT" --run_tag "$RESUME_TAG-run" --reps $(seq 1 "$RESUME_REPS") \
    --keep_solve_counts 0 1 2 3 --tokenizer "$BASE_MODEL" --max_tokens 262144 \
    --teacher_api_base "http://localhost:$TEACHER_PORT/v1" --teacher_model "$SERVED_TEACHER" \
    --top_k "$TOP_K" --out_dir "$SCORE_OUT" --limit "$TEACHER_LIMIT" \
    --shard "$SH" --num_shards "$SCORE_SHARDS" > "$LD/score_sh${SH}.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}" || echo "  some score shards non-zero"
NPZ=$(ls "$SCORE_OUT"/*.npz 2>/dev/null | wc -l); echo "  cached npz = $NPZ"
[ "$NPZ" -gt 0 ] || { echo "ERROR: no teacher logprobs cached"; exit 1; }
free_gpus

echo "════ STAGE 5: KD-train (SP=$TRAIN_SP DP=$TRAIN_DP, maxlen=$TRAIN_MAXLEN) ════"
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
ACCEL_CFG="$(mktemp /tmp/accel_bcp_XXXXXX.yaml)"
cat > "$ACCEL_CFG" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
mixed_precision: bf16
num_processes: $N_POLICY
num_machines: 1
use_cpu: false
deepspeed_config:
  zero_stage: 3
  zero3_init_flag: true
  zero3_save_16bit_model: true
  gradient_clipping: 1.0
  gradient_accumulation_steps: 4
EOF
WANDB_FLAG=""; if [[ -n "${WANDB_API_KEY:-}" && "$SMOKE" != "1" ]]; then WANDB_FLAG="--wandb_project acm-bcp-opd"; else export WANDB_DISABLED=true; fi
accelerate launch --config_file "$ACCEL_CFG" -m distill.train_kd \
  --model "$STUDENT_LM" --cache_dir "$SCORE_OUT" --output_dir "$OUT_CKPT" \
  --max_length "$TRAIN_MAXLEN" --sp_size "$TRAIN_SP" --dp_shard_size "$TRAIN_DP" --grad_accum 4 --lr 5e-6 \
  $WANDB_FLAG $TRAIN_EXTRA 2>&1 | tee "$LD/train.log"
rm -f "$ACCEL_CFG"
[[ -f "$OUT_CKPT/model.safetensors" || -f "$OUT_CKPT/model.safetensors.index.json" ]] || { echo "ERROR: no trained model at $OUT_CKPT"; exit 1; }
free_gpus

echo "════ STAGE 6: merge → serve student → eval held-out ×$EVAL_REPS + grade ════"
python -m train.make_servable_ckpt --sft "$OUT_CKPT" --base "$BASE_MODEL" --out "$SERVABLE" 2>&1 | tee "$LD/merge.log"
[[ -f "$SERVABLE/model.safetensors.index.json" ]] || { echo "ERROR: servable merge failed"; exit 1; }
serve_policy "$SERVABLE" "$SERVED_STUDENT"
for REP in $(seq 1 "$EVAL_REPS"); do
  echo "── eval rep $REP/$EVAL_REPS ──"; pids=(); ETAG="${EVAL_TAG}-run${REP}"
  for ((SH=0; SH<EVAL_SHARDS; SH++)); do
    port=${POLICY_PORTS[$((SH % N_POLICY))]}
    python -m src.run --mode run --benchmark browsecomp-plus --client litellm \
      --model "openai/$SERVED_STUDENT" --agent_api_base "http://localhost:$port/v1" \
      --data "$EVAL" --results_dir "$RESULTS_DIR" --run_dir "$ETAG" --run_id "run_all" \
      --shard "$SH" --num_shards "$EVAL_SHARDS" --use_memory_tool --mc_drop_call_message \
      --summarizer_model "openai/$SERVED_STUDENT" --summarizer_api_base "http://localhost:$port/v1" \
      "${RETR_ARGS[@]}" \
      --override agent.context_window=$MAX_MODEL_LEN runtime.max_iterations=$MAX_ITER \
      --log_level INFO > "$LD/eval_r${REP}_sh${SH}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}" || echo "  some eval rep-$REP shards non-zero"
  python scripts/grade_bcp_gpt5.py --input_dir "$STUDENT_ROOT/$ETAG/run_all" \
    --eval_dir "$STUDENT_ROOT/eval_bcp/$ETAG/run_all" --workers 16 2>&1 | tail -8 | tee "$LD/eval_grade_r${REP}.log"
done
free_gpus
echo "════ BCP OPD PIPELINE DONE $(date -Is) — logs: $LD ════"
