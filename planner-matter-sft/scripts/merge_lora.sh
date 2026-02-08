#!/bin/bash

# Please set the model name and paths according to your environment
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$TRAIN_DIR:$PYTHONPATH"

# Set your checkpoint paths (e.g. after training)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$TRAIN_DIR/checkpoints/lora_qformer_qwen3vl}"
MODEL_PATH="${MODEL_PATH:-$CHECKPOINT_DIR/checkpoint-800}"
SAVE_PATH="${SAVE_PATH:-$CHECKPOINT_DIR/checkpoint-800-merged}"

cd "$TRAIN_DIR" || exit 1
python src_agent/merge_lora_weights.py \
    --model-path "$MODEL_PATH" \
    --model-base "$MODEL_NAME" \
    --save-model-path "$SAVE_PATH" \
    --safe-serialization