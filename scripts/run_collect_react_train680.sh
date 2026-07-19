#!/bin/bash
# ── Collect BASE Qwen3.5-9B ReAct rollouts on bcp_train_680 (the "student" self-rollouts) ──
# Purpose: OPD-1-react experiment. We need base-9B ReAct-mode (NO memory tool) student
# trajectories on the train pool to teacher-score + KD-train an OPD iter-1 in the ReAct regime
# (mirrors memtool-train680 but ReAct). Roll out base Qwen3.5-9B in plain ReAct (search+open
# only, 131K/100t), grade with the GPT-5 judge as we go → solve-count labels for the teacher
# scorer's hard-subset (0-3) filter.
#
# Single node, 4 GPUs (1 embed + 3 policy engines, 24 shards) — sized for a 4-GPU/user cap.
#
# NOTE: add your scheduler's #SBATCH --partition/--qos/--nodelist here
#SBATCH --job-name=collect_react_t680
#SBATCH --gres=gpu:B200:4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48 --mem=400G --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/collect_react_t680-%j.out
#SBATCH --error=logs/collect_react_t680-%j.err

set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipeline_config.sh"
cd "$PROJECT_ROOT"; source "$VENV/bin/activate"; [[ -f .env ]] && { set -a; source .env; set +a; }
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
unset OPENAI_API_BASE 2>/dev/null || true   # grader → api.openai.com (official credit)

REP_LIST="${REP_LIST:-1 2 3 4}"
BASE="$BASE_MODEL"
SERVED="$SERVED_POLICY"
EMBED="$EMBED_MODEL"
EMBED_SERVED="$SERVED_EMBED"; EMBED_PORT=8765; EMBED_GPU=0
POLICY_PORTS=(8001 8002 8003); POLICY_GPUS=(1 2 3)   # 4 GPU: 1 embed + 3 policy
MAX_NUM_SEQS=8; NUM_SHARDS=24

DATA="$TRAIN_DATA"; INDEX_PATH="$BM25_INDEX"
SHARDS_DIR="$EMBED_SHARDS_DIR"
export BCP_RETRIEVER="qwen3_8b_embedding"; export BCP_QWEN3_8B_EMBEDDING_SHARDS_DIR="$SHARDS_DIR"
export EMBED_API_BASE="http://localhost:${EMBED_PORT}"; export EMBED_MODEL_NAME="$EMBED_SERVED"
[[ -n "${JAVA_HOME:-}" ]] && export PATH="$JAVA_HOME/bin:$PATH"
LD="logs/collect-react-t680-$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LD"
echo "job=${SLURM_JOB_ID:-local} node=$(hostname) reps=[$REP_LIST] base ReAct (no memory tool) 131K/100t, $NUM_SHARDS shards on 3 engines" | tee "$LD/meta.txt"

[[ -f "$BASE/config.json" ]] || { echo "ERROR: base model missing on $(hostname): $BASE"; exit 1; }
[[ -d "$EMBED" ]] || { echo "ERROR: embed model missing on $(hostname): $EMBED"; exit 1; }

PIDS=()
wait_for(){ local p=$1 n=$2; echo -n "wait $n :$p"; for i in $(seq 1 300); do curl -fs "http://localhost:$p/health" >/dev/null 2>&1 && { echo " ok"; return 0; }; echo -n .; sleep 5; done; echo " TIMEOUT"; return 1; }
cleanup(){ for x in "${PIDS[@]:-}"; do kill -TERM "$x" 2>/dev/null||true; done; sleep 5; pkill -KILL -f "VLLM::EngineCore" 2>/dev/null||true; pkill -KILL -f "VLLM::Worker" 2>/dev/null||true; }
trap cleanup EXIT

echo "[embed] TP=1 gpu=$EMBED_GPU"
CUDA_VISIBLE_DEVICES=$EMBED_GPU vllm serve "$EMBED" --runner pooling --tensor-parallel-size 1 \
  --served-model-name "$EMBED_SERVED" --host 0.0.0.0 --port "$EMBED_PORT" --max-model-len 8192 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" --trust-remote-code > "$LD/vllm_embed.log" 2>&1 &
PIDS+=($!)
for i in "${!POLICY_PORTS[@]}"; do
  p=${POLICY_PORTS[$i]}; g=${POLICY_GPUS[$i]}; echo "[policy $i] gpu=$g port=$p"
  CUDA_VISIBLE_DEVICES=$g vllm serve "$BASE" --tensor-parallel-size 1 \
    --served-model-name "$SERVED" --host 0.0.0.0 --port "$p" --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --trust-remote-code > "$LD/vllm_policy_$p.log" 2>&1 &
  PIDS+=($!)
done
wait_for "$EMBED_PORT" embed || { echo "ERROR embed"; exit 1; }
for p in "${POLICY_PORTS[@]}"; do wait_for "$p" "policy$p" || { echo "ERROR policy $p"; exit 1; }; done

for REP in $REP_LIST; do
  TAG="react-train680-run${REP}"
  RDIR="$RESULTS_DIR/browsecomp-plus/$SERVED/$TAG/run_all"
  # resume-safe: drop crashed/partial so a requeue redoes only those (completed are skipped)
  [ -d "$RDIR" ] && python3 - "$RDIR" <<'PYC' 2>/dev/null || true
import json,glob,os,sys
for f in glob.glob(os.path.join(sys.argv[1],"run_*.json")):
    if f.endswith(".partial.json"): os.remove(f); continue
    try: st=json.load(open(f)).get("status")
    except Exception: st="crashed"
    if st=="crashed": os.remove(f)
PYC
  echo "── rep=$REP → $TAG ──"; pids=()
  for ((SH=0; SH<NUM_SHARDS; SH++)); do
    port=${POLICY_PORTS[$((SH % ${#POLICY_PORTS[@]}))]}
    python -m src.run --mode run --benchmark browsecomp-plus --client litellm \
      --model "openai/$SERVED" --agent_api_base "http://localhost:$port/v1" \
      --data "$DATA" --index_path "$INDEX_PATH" --results_dir "$RESULTS_DIR" \
      --run_dir "$TAG" --run_id "run_all" --shard "$SH" --num_shards "$NUM_SHARDS" \
      --summarizer_model "openai/$SERVED" --summarizer_api_base "http://localhost:$port/v1" \
      --override runtime.max_iterations=$MAX_ITER agent.context_window=$MAX_MODEL_LEN \
      --log_level INFO > "$LD/shard_r${REP}_${SH}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}" || echo "  some shards non-zero"
  EV="$RESULTS_DIR/browsecomp-plus/$SERVED/eval_bcp/$TAG/run_all"
  python scripts/grade_bcp_gpt5.py --input_dir "$RDIR" --eval_dir "$EV" --workers 16 2>&1 | tee "$LD/grade_r${REP}.log"
done
echo "── DONE $(date -Is) — logs: $LD ──"
