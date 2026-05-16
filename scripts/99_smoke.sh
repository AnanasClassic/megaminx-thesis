#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
METRIC="${METRIC:-UTM}"
METRIC_LC="${METRIC,,}"
DEVICE="${SMOKE_DEVICE:-cpu}"
"$PYTHON" -m megathesis.make_datasets \
  --metric "$METRIC" \
  --depths "1,2" \
  --train-per-depth 16 \
  --val-per-depth 8 \
  --test-per-depth 8 \
  --search-per-depth 4 \
  --seed 7 \
  --device "$DEVICE" \
  --out-dir datasets/smoke
"$PYTHON" -m megathesis.train_state \
  --train-path datasets/smoke/state_train.pt \
  --val-path datasets/smoke/state_val.pt \
  --epochs 1 \
  --steps-per-epoch 2 \
  --batch-size 8 \
  --hd1 64 \
  --hd2 32 \
  --nrd 1 \
  --seed 7 \
  --device "$DEVICE" \
  --run-name smoke_state
SMOKE_STATE="$(cat "checkpoints/latest_state_${METRIC_LC}.txt")"
"$PYTHON" -m megathesis.make_neighbour_labels \
  --state-path datasets/smoke/state_train.pt \
  --teacher "$SMOKE_STATE" \
  --out datasets/smoke/neigh_val.pt \
  --batch-size 8 \
  --teacher-batch-size 64 \
  --device "$DEVICE"
"$PYTHON" -m megathesis.train_neighbour \
  --state-path datasets/smoke/state_train.pt \
  --val-path datasets/smoke/neigh_val.pt \
  --teacher "$SMOKE_STATE" \
  --epochs 1 \
  --steps-per-epoch 2 \
  --teacher-batch-size 64 \
  --batch-size 8 \
  --hd1 64 \
  --hd2 32 \
  --nrd 1 \
  --seed 7 \
  --device "$DEVICE" \
  --run-name smoke_neighbour
SMOKE_NEIGH="$(cat "checkpoints/latest_neighbour_${METRIC_LC}.txt")"
"$PYTHON" -m megathesis.run_search \
  --checkpoint "$SMOKE_STATE" \
  --data-path datasets/smoke/search_test.pt \
  --beam-width 8 \
  --max-depth 4 \
  --device "$DEVICE" \
  --out logs/smoke_state.json
"$PYTHON" -m megathesis.run_search \
  --checkpoint "$SMOKE_NEIGH" \
  --data-path datasets/smoke/search_test.pt \
  --beam-width 8 \
  --max-depth 4 \
  --device "$DEVICE" \
  --out logs/smoke_neighbour.json
