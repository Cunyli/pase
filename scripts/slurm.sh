#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/pase}"
TASK="${TASK:-${1:-train}}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pase}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"
JOB_NAME="${JOB_NAME:-pase-$TASK}"
PARTITION="${PARTITION:-gpu-a100-80g}"
GPU_TYPE="${GPU_TYPE:-a100}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-64G}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"

export ROOT_DIR CONDA_ENV_NAME LOG_DIR SOFTWARE_STACK_MODULE COMPILER_MODULE TASK

mkdir -p "$LOG_DIR"

submit_self() {
  local script_path="$ROOT_DIR/scripts/slurm.sh"
  local sbatch_args=(
    "--job-name=$JOB_NAME"
    "--partition=$PARTITION"
    "--cpus-per-task=$CPUS_PER_TASK"
    "--mem=$MEMORY"
    "--time=$TIME_LIMIT"
    "--output=$LOG_DIR/slurm_%j.out"
    "--error=$LOG_DIR/slurm_%j.err"
  )

  if [[ -n "$GPU_TYPE" ]]; then
    sbatch_args+=("--gres=gpu:${GPU_TYPE}:${GPUS}")
  else
    sbatch_args+=("--gres=gpu:${GPUS}")
  fi

  echo "Submitting $JOB_NAME"
  sbatch "${sbatch_args[@]}" --export=ALL,TASK="$TASK" "$script_path" "$TASK"
}

load_runtime() {
  module load "$SOFTWARE_STACK_MODULE"
  module load "$COMPILER_MODULE"

  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH" | tee -a "$LIVE_LOG"
    exit 1
  fi

  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV_NAME"

  export CC="$(command -v gcc)"
  export CXX="$(command -v g++)"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

best_pase_checkpoint() {
  local ckpt_root="${1:-$ROOT_DIR/checkpoints}"
  local exp_path="${PASE_INFER_EXP_PATH:-}"

  if [[ -z "$exp_path" ]]; then
    exp_path="$(find "$ckpt_root" -maxdepth 1 -type d -name 'exp_vocoder_dual_tau_sim_*' | sort -V | tail -n 1)"
  fi
  if [[ -z "$exp_path" || ! -d "$exp_path" ]]; then
    return 1
  fi
  if [[ -f "$exp_path/model_120.tar" ]]; then
    printf '%s\n' "$exp_path/model_120.tar"
  else
    find "$exp_path" -maxdepth 1 -type f -name 'model_*.tar' | sort -V | tail -n 1
  fi
}

has_complete_output() {
  local out_dir="$1"
  local count
  count="$(find "$out_dir" -maxdepth 2 -type f -name '*.wav' 2>/dev/null | wc -l)"
  [[ "$count" -eq "${EXPECTED_WAV_COUNT:-38}" && -f "$out_dir/inf.scp" && -f "$out_dir/ref.scp" ]]
}

run_model() {
  local name="$1"
  shift

  echo
  echo "===== $name inference started at $(date) =====" | tee -a "$LIVE_LOG"
  echo "Command: $*" | tee -a "$LIVE_LOG"
  "$@" 2>&1 | tee -a "$LIVE_LOG"
  echo "===== $name inference finished at $(date) =====" | tee -a "$LIVE_LOG"
}

run_train() {
  local device_list
  USE_SIMULATION_ROOT="${USE_SIMULATION_ROOT:-/scratch/work/lil14/USE_simulation}"
  CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/train/vocoder_dual_tau_fixed_val_random_start.yaml}"
  PASE_RUN_ROOT="${PASE_RUN_ROOT:-$ROOT_DIR/runs}"
  PASE_GLOBAL_CHECKPOINT_DIR="${PASE_GLOBAL_CHECKPOINT_DIR:-$ROOT_DIR/checkpoints}"
  PASE_DEWAVLM_CKPT="${PASE_DEWAVLM_CKPT:-$ROOT_DIR/checkpoints/pase_hf/DeWavLM.tar}"
  PASE_VOCODER_DUAL_CKPT="${PASE_VOCODER_DUAL_CKPT:-$ROOT_DIR/checkpoints/pase_hf/Vocoder_Dual.tar}"
  PASE_TAU_FIXED_TRAIN_CSV="${PASE_TAU_FIXED_TRAIN_CSV:-/scratch/work/lil14/data/TAU/simulated/phone_room/train/paired.csv}"
  PASE_TAU_FIXED_VALID_CSV="${PASE_TAU_FIXED_VALID_CSV:-/scratch/work/lil14/data/TAU/simulated/phone_room/valid/paired.csv}"
  PASE_NOISE_JSON="${PASE_NOISE_JSON:-$USE_SIMULATION_ROOT/data/train_noise.json}"
  PASE_RIR_JSON="${PASE_RIR_JSON:-$USE_SIMULATION_ROOT/data/train_rir.json}"
  export USE_SIMULATION_ROOT PASE_RUN_ROOT PASE_GLOBAL_CHECKPOINT_DIR
  export PASE_DEWAVLM_CKPT PASE_VOCODER_DUAL_CKPT
  export PASE_TAU_FIXED_TRAIN_CSV PASE_TAU_FIXED_VALID_CSV PASE_NOISE_JSON PASE_RIR_JSON

  test -f "$CONFIG_PATH"
  test -f "$PASE_DEWAVLM_CKPT"
  test -f "$PASE_VOCODER_DUAL_CKPT"

  device_list="$(seq -s, 0 "$((GPUS - 1))")"
  echo "Config: $CONFIG_PATH" | tee -a "$LIVE_LOG"
  if [[ -n "${TRAIN_CMD:-}" ]]; then
    echo "Command: $TRAIN_CMD" | tee -a "$LIVE_LOG"
    $TRAIN_CMD 2>&1 | tee -a "$LIVE_LOG"
  else
    echo "Command: python -m train.train_vocoder_dual -C $CONFIG_PATH -D $device_list" | tee -a "$LIVE_LOG"
    python -m train.train_vocoder_dual -C "$CONFIG_PATH" -D "$device_list" 2>&1 | tee -a "$LIVE_LOG"
  fi
}

run_infer() {
  CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/infer/vocoder_dual_tau_fixed.yaml}"
  PASE_TAU_FIXED_TEST_NOISY_DIR="${PASE_TAU_FIXED_TEST_NOISY_DIR:-/scratch/work/lil14/data/TAU/simulated/phone_room/test/noisy}"
  PASE_TAU_FIXED_TEST_CSV="${PASE_TAU_FIXED_TEST_CSV:-/scratch/work/lil14/data/TAU/simulated/phone_room/test/paired.csv}"
  PASE_TAU_ENHANCED_DIR="${PASE_TAU_ENHANCED_DIR:-${OUTPUT_DIR:-/scratch/work/lil14/data/TAU/enhanced/pase/phone_room/test}}"
  PASE_DEWAVLM_CKPT="${PASE_DEWAVLM_CKPT:-$ROOT_DIR/checkpoints/pase_hf/DeWavLM.tar}"
  PASE_VOCODER_DUAL_CKPT="${PASE_VOCODER_DUAL_CKPT:-$ROOT_DIR/checkpoints/pase_hf/Vocoder_Dual.tar}"
  export PASE_TAU_FIXED_TEST_NOISY_DIR PASE_TAU_FIXED_TEST_CSV PASE_TAU_ENHANCED_DIR
  export PASE_DEWAVLM_CKPT PASE_VOCODER_DUAL_CKPT

  mkdir -p "$PASE_TAU_ENHANCED_DIR"

  if [[ -z "${PASE_INFER_CHECKPOINT:-}" ]]; then
    PASE_INFER_CHECKPOINT="$(best_pase_checkpoint "$ROOT_DIR/checkpoints")"
  fi
  if [[ -z "${PASE_INFER_CHECKPOINT:-}" || ! -f "$PASE_INFER_CHECKPOINT" ]]; then
    echo "No PASE inference checkpoint found. Set PASE_INFER_CHECKPOINT explicitly." | tee -a "$LIVE_LOG"
    exit 1
  fi

  PASE_INFER_EXP_PATH="${PASE_INFER_EXP_PATH:-$(dirname "$PASE_INFER_CHECKPOINT")}"
  PASE_INFER_NETWORK_CONFIG="${PASE_INFER_NETWORK_CONFIG:-$PASE_INFER_EXP_PATH/config.yaml}"
  export PASE_INFER_EXP_PATH PASE_INFER_NETWORK_CONFIG PASE_INFER_CHECKPOINT

  test -f "$CONFIG_PATH"
  test -f "$PASE_INFER_NETWORK_CONFIG"
  test -f "$PASE_DEWAVLM_CKPT"
  test -f "$PASE_VOCODER_DUAL_CKPT"

  echo "Config: $CONFIG_PATH" | tee -a "$LIVE_LOG"
  echo "PASE_INFER_NETWORK_CONFIG=$PASE_INFER_NETWORK_CONFIG" | tee -a "$LIVE_LOG"
  echo "PASE_INFER_CHECKPOINT=$PASE_INFER_CHECKPOINT" | tee -a "$LIVE_LOG"
  python -m inference.infer_vocoder_dual -C "$CONFIG_PATH" -D "${DEVICE_INDEX:-0}" 2>&1 | tee -a "$LIVE_LOG"

  echo "Enhanced files:" | tee -a "$LIVE_LOG"
  find "$PASE_TAU_ENHANCED_DIR/wav" -maxdepth 1 -type f -name "*.wav" | sort | tee -a "$LIVE_LOG"
}

run_infer_all() {
  PASE_ROOT="${PASE_ROOT:-$ROOT_DIR}"
  SEMAMBAPP_ROOT="${SEMAMBAPP_ROOT:-/scratch/work/lil14/SEMambapp-Interspeech}"
  REUSE_ROOT="${REUSE_ROOT:-/scratch/work/lil14/RE-USE}"
  UNISE_ROOT="${UNISE_ROOT:-/scratch/work/lil14/unified-audio/QuarkAudio-UniSE}"
  BSRNN_ROOT="${BSRNN_ROOT:-/scratch/work/lil14/urgent2026_challenge_track1}"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi 2>&1 | tee -a "$LIVE_LOG"
  fi

  if has_complete_output "${PASE_OUTPUT_DIR:-/scratch/work/lil14/data/TAU/enhanced/pase/phone_room/test}"; then
    echo "Skipping PASE; complete output already exists." | tee -a "$LIVE_LOG"
  else
    run_model "PASE" env \
      ROOT_DIR="$PASE_ROOT" \
      LOG_DIR="$PASE_ROOT/logs" \
      CONDA_ENV_NAME="${PASE_CONDA_ENV_NAME:-pase}" \
      CONFIG_PATH="$PASE_ROOT/configs/infer/vocoder_dual_tau_fixed.yaml" \
      SLURM_JOB_ID="$SLURM_JOB_ID" \
      bash "$PASE_ROOT/scripts/slurm.sh" infer
  fi

  if has_complete_output "${SEMAMBAPP_OUT_ROOT:-/scratch/work/lil14/data/TAU/enhanced/semambapp/phone_room/test}"; then
    echo "Skipping SEMamba++; complete output already exists." | tee -a "$LIVE_LOG"
  else
    run_model "SEMamba++" env \
      ROOT_DIR="$SEMAMBAPP_ROOT" \
      LOG_DIR="$SEMAMBAPP_ROOT/logs" \
      CONDA_ENV_NAME="${SEMAMBAPP_CONDA_ENV_NAME:-semambapp}" \
      IN_ROOT="${TAU_TEST_NOISY_DIR:-/scratch/work/lil14/data/TAU/simulated/phone_room/test/noisy}" \
      OUT_ROOT="${SEMAMBAPP_OUT_ROOT:-/scratch/work/lil14/data/TAU/enhanced/semambapp/phone_room/test}" \
      PAIR_CSV="${TAU_TEST_PAIR_CSV:-/scratch/work/lil14/data/TAU/simulated/phone_room/test/paired.csv}" \
      bash "$SEMAMBAPP_ROOT/scripts/slurm_infer_tau.sh"
  fi

  if has_complete_output "${REUSE_OUTPUT_DIR:-/scratch/work/lil14/data/TAU/enhanced/reuse/phone_room/test}"; then
    echo "Skipping RE-USE; complete output already exists." | tee -a "$LIVE_LOG"
  else
    run_model "RE-USE" env \
      ROOT_DIR="$REUSE_ROOT" \
      SEMAMBA_DIR="$REUSE_ROOT/SEMamba" \
      CONDA_ENV_NAME="${REUSE_CONDA_ENV_NAME:-reuse}" \
      INPUT_DIR="${TAU_TEST_NOISY_DIR:-/scratch/work/lil14/data/TAU/simulated/phone_room/test/noisy}" \
      OUTPUT_DIR="${REUSE_OUTPUT_DIR:-/scratch/work/lil14/data/TAU/enhanced/reuse/phone_room/test}" \
      PAIR_CSV="${TAU_TEST_PAIR_CSV:-/scratch/work/lil14/data/TAU/simulated/phone_room/test/paired.csv}" \
      bash "$REUSE_ROOT/scripts/slurm_infer.sh"
  fi

  if has_complete_output "${UNISE_OUTPUT_DIR:-/scratch/work/lil14/data/TAU/enhanced/unise/phone_room/test}"; then
    echo "Skipping UniSE; complete output already exists." | tee -a "$LIVE_LOG"
  else
    run_model "UniSE" env \
      ROOT_DIR="$UNISE_ROOT" \
      LOG_DIR="$UNISE_ROOT/logs" \
      CONDA_ENV_NAME="${UNISE_CONDA_ENV_NAME:-unise}" \
      CONFIG_PATH="$UNISE_ROOT/conf/tau_fixed_unise.yaml" \
      OUTPUT_DIR="${UNISE_OUTPUT_DIR:-/scratch/work/lil14/data/TAU/enhanced/unise/phone_room/test}" \
      PAIR_CSV="${TAU_TEST_PAIR_CSV:-/scratch/work/lil14/data/TAU/simulated/phone_room/test/paired.csv}" \
      bash "$UNISE_ROOT/scripts/slurm_infer_tau.sh"
  fi

  if has_complete_output "${BSRNN_OUTPUT_DIR:-/scratch/work/lil14/data/TAU/enhanced/bsrnn/phone_room/test}"; then
    echo "Skipping BSRNN; complete output already exists." | tee -a "$LIVE_LOG"
  else
    run_model "BSRNN" env \
      ROOT_DIR="$BSRNN_ROOT" \
      LOG_DIR="$BSRNN_ROOT/logs" \
      CONDA_ENV_NAME="${BSRNN_CONDA_ENV_NAME:-/scratch/work/lil14/.conda_envs/urgent2026_baseline_track1}" \
      INPUT_SCP="$BSRNN_ROOT/manifests/tau_fixed/phone_room/test/wav.scp" \
      REF_SCP="$BSRNN_ROOT/manifests/tau_fixed/phone_room/test/ref.scp" \
      OUTPUT_DIR="${BSRNN_OUTPUT_DIR:-/scratch/work/lil14/data/TAU/enhanced/bsrnn/phone_room/test}" \
      bash "$BSRNN_ROOT/scripts/slurm_inference.sh"
  fi

  echo
  echo "All inference jobs finished at $(date)" | tee -a "$LIVE_LOG"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  submit_self
  exit 0
fi

cd "$ROOT_DIR"
LIVE_LOG="$LOG_DIR/${TASK}_${SLURM_JOB_ID}.log"
echo "Live log: $LIVE_LOG"
echo "Job ${SLURM_JOB_ID} started at $(date)" | tee -a "$LIVE_LOG"
echo "Task: $TASK" | tee -a "$LIVE_LOG"
echo "Host: $(hostname)" | tee -a "$LIVE_LOG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$LIVE_LOG"

load_runtime

case "$TASK" in
  train)
    run_train
    ;;
  infer)
    run_infer
    ;;
  infer_all)
    run_infer_all
    ;;
  *)
    echo "Unknown task: $TASK" | tee -a "$LIVE_LOG"
    exit 2
    ;;
esac

echo "Job ${SLURM_JOB_ID} completed at $(date)" | tee -a "$LIVE_LOG"
