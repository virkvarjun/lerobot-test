#!/bin/bash
# Train ACT with Karpathy-inspired optimizer/scheduler adaptations.
# EXPERIMENTAL — lives in experiments/karpathy_autoresearch/
#
# Usage (local smoke, 500 steps):
#   cd Research && bash experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh smoke
#
# Usage (RunPod, full ablation):
#   cd /workspace/lerobot-test/Research && bash experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh runpod
#
# Prerequisites: lerobot installed, dataset and policy configured.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root (Research/) — 3 levels up from experiments/.../scripts/
RESEARCH_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$(dirname "$RESEARCH_DIR")"
# lerobot submodule: at repo root (lerobot-test) or inside Research (local dev)
LEROBOT_DIR="${REPO_ROOT}/lerobot"
[ -d "${RESEARCH_DIR}/lerobot" ] && LEROBOT_DIR="${RESEARCH_DIR}/lerobot"
cd "$RESEARCH_DIR"

# Default: transfer cube
DATASET_REPO="${DATASET_REPO:-lerobot/aloha_sim_transfer_cube}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/train}"
STEPS="${STEPS:-100000}"

if [ "$1" = "smoke" ]; then
  STEPS=500
  EVAL_FREQ=200
  SAVE_FREQ=500
  echo "Smoke test: $STEPS steps"
elif [ "$1" = "runpod" ]; then
  export MUJOCO_GL=egl
  STEPS=100000
  EVAL_FREQ=10000
  SAVE_FREQ=20000
  echo "RunPod run: $STEPS steps"
else
  EVAL_FREQ=10000
  SAVE_FREQ=20000
fi

# Karpathy preset: scheduler + per-group LR
# Optional: --policy.optimizer_lr_action_head=2e-5

PYTHONPATH="${LEROBOT_DIR}:${RESEARCH_DIR}" lerobot-train \
  --dataset.repo_id="$DATASET_REPO" \
  --policy.type=act \
  --policy.use_karpathy_scheduler=True \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_warmdown_ratio=0.5 \
  --policy.scheduler_final_lr_frac=0.05 \
  --output_dir="${OUTPUT_BASE}/faact_karpathy_opt_$(date +%Y%m%d_%H%M)" \
  --steps=$STEPS \
  --eval_freq=$EVAL_FREQ \
  --save_freq=$SAVE_FREQ \
  --batch_size=8 \
  --num_workers=4 \
  "$@"
