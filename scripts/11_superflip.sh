#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${STATE_CKPT:?set STATE_CKPT or run scripts/02_train_state.sh}"
: "${NEIGH_CKPT:?set NEIGH_CKPT or run scripts/05_train_neighbour.sh}"
SUPERFLIP_PATH="${SUPERFLIP_PATH:-datasets/superflip.pt}"
"$PYTHON" -m megathesis.run_search \
  --checkpoint "$STATE_CKPT" \
  --data-path "$SUPERFLIP_PATH" \
  --beam-width "$BEAM_WIDTH" \
  --max-depth "$MAX_DEPTH" \
  --device "$DEVICE" \
  --out "logs/superflip_state_${METRIC_LC}_B${BEAM_WIDTH}.json"
"$PYTHON" -m megathesis.run_search \
  --checkpoint "$NEIGH_CKPT" \
  --data-path "$SUPERFLIP_PATH" \
  --beam-width "$BEAM_WIDTH" \
  --max-depth "$MAX_DEPTH" \
  --device "$DEVICE" \
  --out "logs/superflip_neighbour_${METRIC_LC}_B${BEAM_WIDTH}.json"
