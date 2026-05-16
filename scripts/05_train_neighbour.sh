#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${STATE_CKPT:?set STATE_CKPT or run scripts/02_train_state.sh}"
"$PYTHON" -m megathesis.train_neighbour \
  --state-path "$STATE_TRAIN" \
  --val-path "$NEIGH_VAL" \
  --teacher "$STATE_CKPT" \
  --epochs "$NEIGH_EPOCHS" \
  --steps-per-epoch "$NEIGH_STEPS_PER_EPOCH" \
  --teacher-batch-size "$TEACHER_BATCH" \
  --batch-size "$NEIGH_BATCH" \
  --depths "$DEPTHS" \
  --lr "$NEIGH_LR" \
  --seed "$SEED" \
  --device "$DEVICE"
