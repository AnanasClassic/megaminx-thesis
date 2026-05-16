#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$PYTHON" -m megathesis.train_state \
  --train-path "$STATE_TRAIN" \
  --val-path "$STATE_VAL" \
  --epochs "$STATE_EPOCHS" \
  --steps-per-epoch "$STATE_STEPS_PER_EPOCH" \
  --batch-size "$STATE_BATCH" \
  --depths "$DEPTHS" \
  --lr "$STATE_LR" \
  --seed "$SEED" \
  --device "$DEVICE"
