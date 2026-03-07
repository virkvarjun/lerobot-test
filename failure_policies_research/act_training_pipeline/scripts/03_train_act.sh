#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Train an ACT policy on a recorded dataset.
#
# Usage:
#   bash scripts/03_train_act.sh                    # train from scratch
#   bash scripts/03_train_act.sh --resume           # resume from checkpoint
#
# Prerequisites:
#   1. Dataset recorded via 02_record_dataset.sh
#   2. configs/robot.env filled in
#   3. Virtual environment activated
#
# The script reads all hyperparameters from configs/robot.env.
# ACT-specific overrides (chunk_size, dim_model, etc.) are only
# passed if explicitly set in the env file.
#
# Docs: https://huggingface.co/docs/lerobot
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Load config ──────────────────────────────────────────────
ENV_FILE="${PROJECT_DIR}/configs/robot.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: ${ENV_FILE} not found."
    echo "Copy the example and fill in your values:"
    echo "  cp configs/robot.env.example configs/robot.env"
    exit 1
fi
source "$ENV_FILE"

# ── Resolve paths ────────────────────────────────────────────
REPO_ID="${HF_USER}/${DATASET_NAME}"
LOCAL_ROOT="${PROJECT_DIR}/data/${DATASET_NAME}"
JOB_NAME="act_${DATASET_NAME}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/train/${JOB_NAME}"

RESUME_FLAG=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FLAG="--resume=true"
    echo ">>> Resuming training from last checkpoint."
fi

echo ">>> Training ACT policy"
echo "    Dataset    : ${REPO_ID}"
echo "    Local root : ${LOCAL_ROOT}"
echo "    Output dir : ${OUTPUT_DIR}"
echo "    Device     : ${DEVICE}"
echo "    Steps      : ${TRAIN_STEPS}"
echo "    Batch size : ${BATCH_SIZE}"
echo ""

# ── Build optional ACT policy overrides ──────────────────────
ACT_OVERRIDES=""
[[ -n "${CHUNK_SIZE:-}" ]]         && ACT_OVERRIDES+=" --policy.chunk_size=${CHUNK_SIZE}"
[[ -n "${DIM_MODEL:-}" ]]          && ACT_OVERRIDES+=" --policy.dim_model=${DIM_MODEL}"
[[ -n "${N_HEADS:-}" ]]            && ACT_OVERRIDES+=" --policy.n_heads=${N_HEADS}"
[[ -n "${N_ENCODER_LAYERS:-}" ]]   && ACT_OVERRIDES+=" --policy.n_encoder_layers=${N_ENCODER_LAYERS}"
[[ -n "${N_DECODER_LAYERS:-}" ]]   && ACT_OVERRIDES+=" --policy.n_decoder_layers=${N_DECODER_LAYERS}"
[[ -n "${KL_WEIGHT:-}" ]]          && ACT_OVERRIDES+=" --policy.kl_weight=${KL_WEIGHT}"
[[ -n "${LEARNING_RATE:-}" ]]      && ACT_OVERRIDES+=" --policy.optimizer_lr=${LEARNING_RATE}"

# ── Train ────────────────────────────────────────────────────
lerobot-train \
    --policy.type=act \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${LOCAL_ROOT}" \
    --policy.device="${DEVICE}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --batch_size="${BATCH_SIZE}" \
    --steps="${TRAIN_STEPS}" \
    --eval_freq="${EVAL_FREQ}" \
    --save_freq="${SAVE_FREQ}" \
    --save_checkpoint=true \
    ${ACT_OVERRIDES} \
    ${RESUME_FLAG}

echo ""
echo ">>> Training complete."
echo "    Checkpoints saved to: ${OUTPUT_DIR}/checkpoints/"
echo "    Final model:          ${OUTPUT_DIR}/checkpoints/last/pretrained_model/"
