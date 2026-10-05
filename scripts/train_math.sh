#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
exec accelerate launch \
    --config_file "$REPO_DIR/map/configs/accelerate.yaml" \
    --num_processes "${NUM_PROCESSES:-8}" \
    --main_process_port "${MASTER_PORT:-29500}" \
    --module map.train \
    --config "$REPO_DIR/map/configs/math.yaml" "$@"
