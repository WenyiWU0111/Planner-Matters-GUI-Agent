# Multimodal Reasoning Bank Scripts

Scripts for building and testing the multimodal reasoning bank.

## Main Scripts

- **`build_reasoning_bank.py`** - Build reasoning bank from trajectories
  - Use `--multimodal` for multimodal mode
  - Use `--visual_stage1` for aggressive visual Stage 1

## Testing Scripts

- **`validate_multimodal.py`** - Validate setup (no VLM needed)
- **`test_stage1_only.py`** - Test Stage 1 text-only mode
- **`test_stage1_visual.py`** - Test Stage 1 visual mode (with all screenshots)
- **`test_multimodal_distill.py`** - Full pipeline test

## Example

- **`example_build_multimodal_bank.sh`** - Example build commands

## Quick Start

```bash
# Test visual Stage 1
python test_stage1_visual.py

# Build bank with visual Stage 1
python build_reasoning_bank.py \
  --input_glob "../../data/downloaded_datasets/webvoyager_memory/Amazon/*/test/**/*.jsonl" \
  --multimodal \
  --visual_stage1 \
  --domain Amazon
```

