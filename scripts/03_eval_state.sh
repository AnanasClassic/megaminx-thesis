#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${STATE_CKPT:?set STATE_CKPT or run scripts/02_train_state.sh}"
"$PYTHON" -m megathesis.eval_state \
  --checkpoint "$STATE_CKPT" \
  --data-path "$STATE_TEST" \
  --batch-size "$TEACHER_BATCH" \
  --device "$DEVICE"
