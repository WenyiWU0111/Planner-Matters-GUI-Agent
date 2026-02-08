#!/bin/bash

# Get the absolute path to planner-matter-inference directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(cd "$SCRIPT_DIR/../../planner-matter-inference" && pwd)"
export INFERENCE_DIR="$INFERENCE_DIR"

# Get the absolute path to directory (parent of scripts)
TRAIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEEPSPEED_CONFIG="$TRAIN_DIR/scripts/zero3.json"

# EDIT: Use this script
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
export PYTHONPATH=src:$PYTHONPATH
# Set GPU devices for your environment (e.g. 0,1,2,3 or 4,5,6,7)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export LOCAL_WORLD_SIZE=4

GLOBAL_BATCH_SIZE=32 # Adjusted for 1 GPU
BATCH_PER_DEVICE=1    # Keep same per-device batch size
NUM_DEVICES=4      # Using 1 GPU
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
# You should freeze the the merger also, becuase the merger is included in the vision_tower.

deepspeed --master_port 2048 src_agent/training/train.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_id $MODEL_NAME \
    --image_folder '' \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir checkpoints/lora_qformer_qwen3vl \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 800 \
    --save_total_limit 10 \
    --dataloader_num_workers 4 \
    --max_grad_norm 3.0
    # --report_to  wandb