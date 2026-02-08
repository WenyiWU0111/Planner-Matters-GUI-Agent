#!/bin/bash
# Example script for building multimodal reasoning bank

# Set paths
INFERENCE_PROJECT_ROOT="${INFERENCE_PROJECT_ROOT:-.}"
cd "$INFERENCE_PROJECT_ROOT"

# Example 1: Build from success trajectories only
echo "Building multimodal reasoning bank from success trajectories..."
python scripts/build_reasoning_bank.py \
  --input_glob "data/downloaded_datasets/webvoyager_memory/Amazon/qwen2.5-vl-32b/test/success/*.jsonl" \
  --dataset webvoyager \
  --domain Amazon \
  --multimodal \
  --max_items_per_traj 2

echo ""
echo "Output files:"
echo "  - memory/reasoning_bank_mm.jsonl"
echo "  - memory_index/reasoning_bank_mm.faiss"
echo "  - media/reasoning_bank/{task_id}/step_*.jpg"

# Example 2: Build from both success and failure trajectories
# Uncomment to run:
# echo ""
# echo "Building from success and failure trajectories..."
# python scripts/build_reasoning_bank.py \
#   --input_glob "data/downloaded_datasets/webvoyager_memory/Amazon/qwen2.5-vl-32b/test/**/*.jsonl" \
#   --dataset webvoyager \
#   --domain Amazon \
#   --multimodal \
#   --max_items_per_traj 2

echo ""
echo "To use the multimodal bank with the agent, run:"
echo "python run.py --use_reasoning_bank True --reasoning_bank_multimodal True --domain Amazon"

