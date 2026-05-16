#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${NEIGH_CKPT:?set NEIGH_CKPT or run scripts/05_train_neighbour.sh}"
"$PYTHON" -m megathesis.eval_neighbour \
  --checkpoint "$NEIGH_CKPT" \
  --data-path "$NEIGH_VAL" \
  --batch-size "$NEIGH_BATCH" \
  --device "$DEVICE"
