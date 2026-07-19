#!/bin/bash
# ── OPD Phase-2: KD-train the 9B student from cached teacher logprobs ──────────
# SMOKE=1 → validate end-to-end on a few trajectories/steps at reduced context (sp=1).
# SMOKE=0 → full run (256K, Ulysses SP). Author NEVER launches full before a clean smoke.
# SUBMIT (do NOT run directly):  SMOKE=1 sbatch scripts/run_distill_train.sh
# NOTE: add your scheduler's #SBATCH --partition/--qos/--nodelist here
#SBATCH --job-name=distill_train
#SBATCH --gres=gpu:B200:8
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48 --mem=512G --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/distill_train-%j.out
#SBATCH --error=logs/distill_train-%j.err

set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipeline_config.sh"
cd "$PROJECT_ROOT"; source "$VENV/bin/activate"; [[ -f .env ]] && { set -a; source .env; set +a; }
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_ROOT/wandb}"
# live wandb logging if a key is present (.env); else disable. SMOKE runs skip wandb.
if [[ -n "${WANDB_API_KEY:-}" && "${SMOKE:-1}" != "1" ]]; then
  WANDB_PROJECT="${WANDB_PROJECT:-acm-opd}"; WANDB_FLAG="--wandb_project $WANDB_PROJECT"
else
  export WANDB_DISABLED=true; WANDB_FLAG=""
fi

SMOKE="${SMOKE:-1}"
MODEL="${MODEL:-$STUDENT_LM}"   # iter-2: init from iter-1 ckpt
CACHE="${CACHE:-$LOGPROBS_DIR}"
NUM_GPUS="${NUM_GPUS:-8}"
if [[ "$SMOKE" == "1" ]]; then
  OUT="$CKPT_DIR/acm-opd_v1-smoke"
  MAXLEN=65536; SP=${SMOKE_SP:-1}; DP=$((NUM_GPUS/SP)); GA=1; EXTRA="--max_samples 20 --max_steps 5"
else
  # FULL. Default SP=4/256K, but Ulysses-SP is incompatible with our per-position teacher
  # side-data (its sample-sharder IndexErrors) → F1 fallback: FULL_SP=1 FULL_MAXLEN=65536
  # trains on the ≤64K subset without SP (validated by the SP=1 smoke). Full-data SP version
  # needs F2/F3 (per-rank loss / gather-to-rank0) — deferred.
  OUT="$CKPT_DIR/acm-opd_v1${FULL_TAG:-}"
  MAXLEN=${FULL_MAXLEN:-262144}; SP=${FULL_SP:-4}; DP=$((NUM_GPUS/SP)); GA=4; EXTRA="--epochs 3"
fi
echo "host=$(hostname) SMOKE=$SMOKE MAXLEN=$MAXLEN SP=$SP DP=$DP cache=$CACHE out=$OUT"
[[ -d "$CACHE" && $(ls "$CACHE"/*.npz 2>/dev/null | wc -l) -gt 0 ]] || { echo "ERROR: no npz cache at $CACHE"; exit 1; }

ACCEL_CFG="$(mktemp /tmp/accel_kd_XXXXXX.yaml)"; trap 'rm -f "$ACCEL_CFG"' EXIT
cat > "$ACCEL_CFG" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
mixed_precision: bf16
num_processes: ${NUM_GPUS}
num_machines: 1
use_cpu: false
deepspeed_config:
  zero_stage: 3
  zero3_init_flag: true
  zero3_save_16bit_model: true
  gradient_clipping: 1.0
  gradient_accumulation_steps: ${GA}
EOF

TOK_FLAG=""; [[ -n "${TOKENIZER:-}" ]] && TOK_FLAG="--tokenizer $TOKENIZER"
accelerate launch --config_file "$ACCEL_CFG" -m distill.train_kd \
  --model "$MODEL" $TOK_FLAG --cache_dir "$CACHE" --output_dir "$OUT" \
  --max_length "$MAXLEN" --sp_size "$SP" --dp_shard_size "$DP" --grad_accum "$GA" --lr 5e-6 $WANDB_FLAG $EXTRA
RC=$?
echo "── End: $(date -Is) rc=$RC ──"
# propagate child failure so afterok dependents don't fire on a masked crash (07-07 bug)
[[ -f "$OUT/model.safetensors" || -f "$OUT/model.safetensors.index.json" ]] || { echo "ERROR: no trained model saved at $OUT"; exit 1; }
exit $RC
