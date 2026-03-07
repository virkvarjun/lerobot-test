#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Evaluate a trained ACT policy on the real SO-101 robot.
#
# This uses `lerobot-record --policy.path=...` which runs the
# trained policy on the robot and records the evaluation episodes.
# This is the standard LeRobot approach for real-world eval
# (lerobot-eval is for simulated environments only).
#
# Usage:
#   bash scripts/04_eval_act.sh                         # use last checkpoint
#   bash scripts/04_eval_act.sh /path/to/checkpoint     # use specific checkpoint
#
# Prerequisites:
#   1. Policy trained via 03_train_act.sh
#   2. Robot connected and calibrated
#   3. configs/robot.env filled in
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

# ── Resolve checkpoint path ─────────────────────────────────
JOB_NAME="act_${DATASET_NAME}"
DEFAULT_CHECKPOINT="${PROJECT_DIR}/outputs/train/${JOB_NAME}/checkpoints/last/pretrained_model"
POLICY_PATH="${1:-$DEFAULT_CHECKPOINT}"

if [[ ! -d "$POLICY_PATH" ]]; then
    echo "ERROR: Checkpoint not found at: ${POLICY_PATH}"
    echo "Train a policy first:  bash scripts/03_train_act.sh"
    exit 1
fi

# ── Eval dataset naming ─────────────────────────────────────
EVAL_REPO_ID="${HF_USER}/${DATASET_NAME}_eval"
EVAL_ROOT="${PROJECT_DIR}/data/${DATASET_NAME}_eval"
EVAL_EPISODES="${EVAL_EPISODES:-10}"

echo ">>> Evaluating ACT policy on real robot"
echo "    Checkpoint    : ${POLICY_PATH}"
echo "    Eval episodes : ${EVAL_EPISODES}"
echo "    Eval dataset  : ${EVAL_ROOT}"
echo ""

# ── Build camera arg if configured ───────────────────────────
CAMERA_ARG=""
if [[ -n "${CAMERA_INDEX:-}" ]]; then
    CAMERA_ARG="--robot.cameras={ front: {type: opencv, index_or_path: ${CAMERA_INDEX}, width: ${CAMERA_WIDTH:-640}, height: ${CAMERA_HEIGHT:-480}, fps: ${CAMERA_FPS:-30}} }"
fi

# ── Run policy on robot (record evaluation episodes) ─────────
lerobot-record \
    --robot.type=so101_follower \
    --robot.port="${ROBOT_PORT_FOLLOWER}" \
    --robot.id="${ROBOT_ID}" \
    ${CAMERA_ARG} \
    --policy.path="${POLICY_PATH}" \
    --policy.device="${DEVICE}" \
    --dataset.repo_id="${EVAL_REPO_ID}" \
    --dataset.single_task="${DATASET_TASK}" \
    --dataset.root="${EVAL_ROOT}" \
    --dataset.num_episodes="${EVAL_EPISODES}" \
    --dataset.episode_time_s="${DATASET_EPISODE_TIME_S}" \
    --dataset.fps="${DATASET_FPS}" \
    --dataset.push_to_hub=false \
    --display_data=true

echo ""
echo ">>> Evaluation complete."
echo "    Recorded episodes saved to: ${EVAL_ROOT}"
