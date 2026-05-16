#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f configs/default.env ]]; then
  while IFS='=' read -r key value; do
    [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
    if [[ -z "${!key+x}" ]]; then
      eval "export ${key}=${value}"
    fi
  done < configs/default.env
fi
PYTHON="${PYTHON:-python3}"
METRIC="${METRIC:-UTM}"
METRIC_LC="$(printf '%s' "$METRIC" | tr '[:upper:]' '[:lower:]')"
DEPTHS="${DEPTHS:-${BUCKETS:-1-60}}"
DATA_DIR="datasets/${METRIC_LC}"
STATE_TRAIN="${STATE_TRAIN:-${DATA_DIR}/state_train.pt}"
STATE_VAL="${STATE_VAL:-${DATA_DIR}/state_val.pt}"
STATE_TEST="${STATE_TEST:-${DATA_DIR}/state_test.pt}"
SEARCH_TEST="${SEARCH_TEST:-${DATA_DIR}/search_test.pt}"
NEIGH_VAL="${NEIGH_VAL:-${DATA_DIR}/neigh_val.pt}"
latest_ckpt() {
  local kind="$1"
  local path="checkpoints/latest_${kind}_${METRIC_LC}.txt"
  [[ -f "$path" ]] && cat "$path" || true
}
depth_values() {
  local text="${DEPTHS// /}"
  local part a b
  IFS=',' read -ra parts <<< "$text"
  for part in "${parts[@]}"; do
    [[ -n "$part" ]] || continue
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      a="${BASH_REMATCH[1]}"
      b="${BASH_REMATCH[2]}"
      seq "$a" "$b"
    else
      printf '%s\n' "$part"
    fi
  done
}
STATE_CKPT="${STATE_CKPT:-$(latest_ckpt state)}"
NEIGH_CKPT="${NEIGH_CKPT:-$(latest_ckpt neighbour)}"
