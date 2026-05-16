#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${STATE_CKPT:?set STATE_CKPT or run scripts/02_train_state.sh}"
: "${NEIGH_CKPT:?set NEIGH_CKPT or run scripts/05_train_neighbour.sh}"
for D in $(search_depth_values); do
  "$PYTHON" -m megathesis.run_search \
    --checkpoint "$STATE_CKPT" \
    --data-path "$SEARCH_TEST" \
    --beam-width "$BEAM_WIDTH" \
    --max-depth "$MAX_DEPTH" \
    --depths "$D" \
    --tests "$TESTS" \
    --device "$DEVICE" \
    --out "logs/depth_state_${METRIC_LC}_d${D}_B${BEAM_WIDTH}.json"
  "$PYTHON" -m megathesis.run_search \
    --checkpoint "$NEIGH_CKPT" \
    --data-path "$SEARCH_TEST" \
    --beam-width "$BEAM_WIDTH" \
    --max-depth "$MAX_DEPTH" \
    --depths "$D" \
    --tests "$TESTS" \
    --device "$DEVICE" \
    --out "logs/depth_neighbour_${METRIC_LC}_d${D}_B${BEAM_WIDTH}.json"
done
"$PYTHON" -m megathesis.summarize_logs
