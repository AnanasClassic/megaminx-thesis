#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$PYTHON" -m megathesis.make_datasets \
  --metric "$METRIC" \
  --depths "$DEPTHS" \
  --train-per-depth "$TRAIN_PER_DEPTH" \
  --val-per-depth "$VAL_PER_DEPTH" \
  --test-per-depth "$TEST_PER_DEPTH" \
  --search-per-depth "$SEARCH_PER_DEPTH" \
  --seed "$SEED" \
  --device "$DEVICE"
