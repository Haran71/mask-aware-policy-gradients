#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
exec torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_PROCESSES:-1}" \
    "$REPO_DIR/eval/eval.py" \
    --dataset math \
    --model_path GSAI-ML/LLaDA-8B-Instruct \
    --gen_length 128 \
    --output_dir "results/math" "$@"
