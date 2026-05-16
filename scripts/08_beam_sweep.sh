#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${STATE_CKPT:?set STATE_CKPT or run scripts/02_train_state.sh}"
: "${NEIGH_CKPT:?set NEIGH_CKPT or run scripts/05_train_neighbour.sh}"
for B in $BEAM_WIDTHS; do
  "$PYTHON" -m megathesis.run_search \
    --checkpoint "$STATE_CKPT" \
    --data-path "$SEARCH_TEST" \
    --beam-width "$B" \
    --max-depth "$MAX_DEPTH" \
    --tests "$TESTS" \
    --device "$DEVICE" \
    --out "logs/beam_sweep_state_${METRIC_LC}_B${B}.json"
  "$PYTHON" -m megathesis.run_search \
    --checkpoint "$NEIGH_CKPT" \
    --data-path "$SEARCH_TEST" \
    --beam-width "$B" \
    --max-depth "$MAX_DEPTH" \
    --tests "$TESTS" \
    --device "$DEVICE" \
    --out "logs/beam_sweep_neighbour_${METRIC_LC}_B${B}.json"
done
"$PYTHON" -m megathesis.plot_sweep \
  --metric "$METRIC" \
  --bootstraps "$PLOT_BOOTSTRAPS" \
  --out "logs/beam_sweep_${METRIC_LC}.png" \
  --csv "logs/beam_sweep_${METRIC_LC}.csv"
