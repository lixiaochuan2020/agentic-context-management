#!/bin/bash
# ── OPD Phase-1: teacher-score student trajectories (serve teacher @131K + cache top-K logprobs) ──
# Serves Qwen3.5-397B-A17B (bf16 TP=8) from LOCAL NVMe, then runs the scorer over the harder
# subset (solve-count 0–3) of the base-9B MemTool rollouts, caching teacher top-K prompt_logprobs
# at each assistant position. Teacher tokenizer == student tokenizer (verified).
#
# Verification run first (cheap):  LIMIT=5 sbatch scripts/run_distill_score.sh
# Full run:                        LIMIT=0 sbatch scripts/run_distill_score.sh   (multi-hour)
# SUBMIT (do NOT run directly).
# NOTE: add your scheduler's #SBATCH --partition/--qos/--nodelist here
#SBATCH --job-name=distill_score
#SBATCH --gres=gpu:B200:8
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48 --mem=512G --time=8:00:00
#SBATCH --output=logs/distill_score-%j.out
#SBATCH --error=logs/distill_score-%j.err

set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipeline_config.sh"
cd "$PROJECT_ROOT"; source "$VENV/bin/activate"; [[ -f .env ]] && { set -a; source .env; set +a; }
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

LIMIT="${LIMIT:-5}"; TOP_K="${TOP_K:-20}"; NUM_SHARDS="${NUM_SHARDS:-4}"
SRC="$TEACHER_MODEL_SRC"    # canonical source copy
T="$TEACHER_MODEL"          # local NVMe (staged by smoke test)
PORT=8900; SERVED="$SERVED_TEACHER"; TP=8; MAXLEN=262144   # 256K: trajectories run long (median ~167K)
# tokenize with the STUDENT (base-9B) template — its chat_template.jinja has {% generation %} for the
# assistant mask; token ids are identical to the teacher (same vocab), so the teacher scores them as-is.
TOKZ="$BASE_MODEL"
RUNS_ROOT="${RUNS_ROOT:-$RESULTS_DIR/browsecomp-plus/$SERVED_POLICY}"
OUT_DIR="${OUT_DIR:-$LOGPROBS_DIR}"
REP_TAG_GLOB="${REP_TAG_GLOB:-memtool-train680-run}"  # iter-2 uses memtool-train680-iter2-run
LD="logs/distill-score-$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LD"
echo "host=$(hostname) start=$(date -Is) LIMIT=$LIMIT TOP_K=$TOP_K out=$OUT_DIR" | tee "$LD/meta.txt"

echo "=== ensure teacher staged on local NVMe ($T) ==="
if [ "$(ls "$T"/*.safetensors 2>/dev/null | wc -l)" -lt 94 ]; then
  echo "  staging from $SRC ..."; mkdir -p "$T"; rsync -a "$SRC/" "$T/" 2>&1 | tail -2
fi
[ "$(ls "$T"/*.safetensors 2>/dev/null | wc -l)" -ge 94 ] || { echo "ERROR: teacher not staged"; exit 1; }

echo "=== serve teacher bf16 TP=$TP @${MAXLEN} ==="
# prompt_logprobs runs full-vocab log_softmax per position → chunk the prefill so the logits spike
# stays small (~1GB/GPU at 8192 vs ~15GB at 120K). KV is tiny (Gated DeltaNet) so low mem-util is fine.
vllm serve "$T" --tensor-parallel-size $TP --served-model-name "$SERVED" \
  --host 0.0.0.0 --port $PORT --max-model-len $MAXLEN --max-num-seqs "$NUM_SHARDS" \
  --enable-chunked-prefill --max-num-batched-tokens 8192 \
  --disable-custom-all-reduce \
  --gpu-memory-utilization 0.80 --trust-remote-code > "$LD/vllm_serve.log" 2>&1 &
SRV=$!; trap 'kill -TERM $SRV 2>/dev/null; sleep 5; pkill -KILL -f VLLM 2>/dev/null||true' EXIT
echo -n "waiting for /health"; ok=0
for i in $(seq 1 240); do
  curl -fs "http://localhost:$PORT/health" >/dev/null 2>&1 && { ok=1; echo " UP"; break; }
  kill -0 $SRV 2>/dev/null || { echo " SERVER DIED"; break; }
  echo -n .; sleep 10
done
[ "$ok" = 1 ] || { echo "=== SERVE FAILED ==="; tail -40 "$LD/vllm_serve.log"; exit 1; }

echo "=== verify prompt_logprobs is returned (the scorer's core mechanism) ==="
curl -s "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"prompt\":[785,6722,315,9625,374],\"max_tokens\":1,\"temperature\":0,\"prompt_logprobs\":5}" \
  | python -c "import json,sys;d=json.load(sys.stdin);plp=d['choices'][0].get('prompt_logprobs');print('  prompt_logprobs present:',plp is not None,'| positions:',len(plp) if plp else 0)" | tee "$LD/prompt_logprobs_check.txt"

echo "=== score: $NUM_SHARDS parallel shards (LIMIT=$LIMIT/shard, TOP_K=$TOP_K) ==="
pids=()
for SH in $(seq 0 $((NUM_SHARDS-1))); do
  python -m distill.score_teacher_logprobs \
    --runs_root "$RUNS_ROOT" --run_tag "$REP_TAG_GLOB" --reps 1 2 3 4 --keep_solve_counts 0 1 2 3 \
    --tokenizer "$TOKZ" --max_tokens "$MAXLEN" \
    --teacher_api_base "http://localhost:$PORT/v1" --teacher_model "$SERVED" \
    --top_k "$TOP_K" --out_dir "$OUT_DIR" --limit "$LIMIT" \
    --shard "$SH" --num_shards "$NUM_SHARDS" > "$LD/score_sh${SH}.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}" || echo "  some shards non-zero"
cat "$LD"/score_sh*.log | grep -E "shard|done: cached|skipped" | tail -12 | tee "$LD/score.log"

echo "=== cached npz so far: $(ls "$OUT_DIR"/*.npz 2>/dev/null | wc -l) | dir size: $(du -sh "$OUT_DIR" 2>/dev/null|cut -f1) ==="
echo "── DONE $(date -Is) — logs: $LD ──"
