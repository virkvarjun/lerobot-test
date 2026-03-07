#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Record a dataset from the SO-101 leader/follower arms.
#
# Usage:
#   bash scripts/02_record_dataset.sh          # record locally only
#   bash scripts/02_record_dataset.sh --push   # record + push to HF Hub
#
# Prerequisites:
#   1. Copy configs/robot.env.example → configs/robot.env and fill in values.
#   2. Both arms connected and calibrated.
#   3. Virtual environment activated.
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

# ── Determine push-to-hub ────────────────────────────────────
PUSH_TO_HUB="false"
if [[ "${1:-}" == "--push" ]]; then
    PUSH_TO_HUB="true"
    echo ">>> Will push dataset to HF Hub after recording."
fi

REPO_ID="${HF_USER}/${DATASET_NAME}"
LOCAL_ROOT="${PROJECT_DIR}/data/${DATASET_NAME}"

echo ">>> Recording dataset: ${REPO_ID}"
echo "    Episodes   : ${DATASET_NUM_EPISODES}"
echo "    Episode len : ${DATASET_EPISODE_TIME_S}s"
echo "    FPS         : ${DATASET_FPS}"
echo "    Local path  : ${LOCAL_ROOT}"
echo ""

# ── Build camera arg if configured ───────────────────────────
CAMERA_ARG=""
if [[ -n "${CAMERA_INDEX:-}" ]]; then
    CAMERA_ARG="--robot.cameras={ front: {type: opencv, index_or_path: ${CAMERA_INDEX}, width: ${CAMERA_WIDTH:-640}, height: ${CAMERA_HEIGHT:-480}, fps: ${CAMERA_FPS:-30}} }"
fi

# ── Record ───────────────────────────────────────────────────
lerobot-record \
    --robot.type=so101_follower \
    --robot.port="${ROBOT_PORT_FOLLOWER}" \
    --robot.id="${ROBOT_ID}" \
    ${CAMERA_ARG} \
    --teleop.type=so101_leader \
    --teleop.port="${ROBOT_PORT_LEADER}" \
    --teleop.id="${TELEOP_ID}" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.single_task="${DATASET_TASK}" \
    --dataset.root="${LOCAL_ROOT}" \
    --dataset.num_episodes="${DATASET_NUM_EPISODES}" \
    --dataset.episode_time_s="${DATASET_EPISODE_TIME_S}" \
    --dataset.fps="${DATASET_FPS}" \
    --dataset.push_to_hub="${PUSH_TO_HUB}" \
    --display_data=true

echo ""
echo ">>> Recording complete. Dataset saved to: ${LOCAL_ROOT}"
if [[ "$PUSH_TO_HUB" == "true" ]]; then
    echo ">>> Dataset pushed to: https://huggingface.co/datasets/${REPO_ID}"
fi
