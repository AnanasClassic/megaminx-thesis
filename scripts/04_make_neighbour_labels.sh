#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${STATE_CKPT:?set STATE_CKPT or run scripts/02_train_state.sh}"
"$PYTHON" -m megathesis.make_neighbour_labels \
  --state-path "$STATE_VAL" \
  --teacher "$STATE_CKPT" \
  --out "$NEIGH_VAL" \
  --batch-size "$LABEL_BATCH" \
  --teacher-batch-size "$TEACHER_BATCH" \
  --device "$DEVICE"
