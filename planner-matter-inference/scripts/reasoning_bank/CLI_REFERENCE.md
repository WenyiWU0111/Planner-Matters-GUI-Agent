# Multimodal Reasoning Bank - CLI Reference

## Build Reasoning Bank

### Basic Usage
```bash
python scripts/reasoning_bank/build_reasoning_bank.py \
  --input_glob "path/to/trajectories/**/*.jsonl" \
  --dataset webvoyager \
  --domain Amazon
```

### All Options

```bash
python scripts/reasoning_bank/build_reasoning_bank.py \
  --input_glob "data/downloaded_datasets/webvoyager_memory/Amazon/*/test/**/*.jsonl" \
  --dataset webvoyager \
  --domain Amazon \
  --bank_jsonl memory/reasoning_bank.jsonl \
  --prompts_dir agent/prompts \
  --max_items_per_traj 2 \
  --multimodal \
  --visual_stage1
```

**Arguments:**
- `--input_glob` (required) - Glob pattern for trajectory JSON files
- `--dataset` (default: webvoyager) - Dataset name for metadata
- `--domain` (default: Amazon) - Domain name for metadata
- `--bank_jsonl` (default: memory/reasoning_bank.jsonl) - Output JSONL path
- `--prompts_dir` (default: agent/prompts) - Directory with prompt templates
- `--max_items_per_traj` (default: 3) - Max items per trajectory (recommend 2 for multimodal)
- `--multimodal` (flag) - Enable multimodal distillation with screenshots
- `--visual_stage1` (flag) - Include all screenshots in Stage 1 (more aggressive)

**Output files:**
- Text-only: `memory/reasoning_bank.jsonl` + `memory_index/reasoning_bank_text.faiss`
- Multimodal: `memory/reasoning_bank_mm.jsonl` + `memory_index/reasoning_bank_mm.faiss` + `media/reasoning_bank/{task_id}/step_*.jpg`

---

## Testing & Validation

### 1. Validate Setup (No VLM needed)
```bash
python scripts/reasoning_bank/validate_multimodal.py
```
Checks: trajectory parsing, image saving, prompt templates exist

### 2. Test Stage 1 Text-Only
```bash
python scripts/reasoning_bank/test_stage1_only.py
```
Shows which key steps VLM identifies from text trajectory

### 3. Test Stage 1 Visual (with all screenshots)
```bash
python scripts/reasoning_bank/test_stage1_visual.py

# Or with custom trajectory
python scripts/reasoning_bank/test_stage1_visual.py \
  --trajectory "path/to/trajectory.jsonl" \
  --success  # or --failure
```
Shows which key steps VLM identifies with full visual context

### 4. Test Full Pipeline (Stage 1 + Stage 3)
```bash
python scripts/reasoning_bank/test_multimodal_distill.py
```
Runs complete distillation and shows extracted items with images

---

## Run Agent with Reasoning Bank

### Text-Only Mode (Legacy)
```bash
python run.py \
  --use_reasoning_bank True \
  --reasoning_bank_path memory/reasoning_bank.jsonl \
  --reasoning_index_base memory_index/reasoning_bank_text \
  --reasoning_top_k 2 \
  --reasoning_domain_filter True \
  --evaluation_type webvoyager \
  --domain Amazon
```

### Multimodal Mode (Recommended)
```bash
python run.py \
  --use_reasoning_bank True \
  --reasoning_bank_multimodal True \
  --reasoning_bank_path memory/reasoning_bank_mm.jsonl \
  --reasoning_index_base memory_index/reasoning_bank_mm \
  --reasoning_top_k 2 \
  --reasoning_domain_filter True \
  --evaluation_type webvoyager \
  --domain Amazon
```

**Agent Arguments:**
- `--use_reasoning_bank True` - Enable reasoning bank
- `--reasoning_bank_multimodal True` - Use multimodal mode (text + images)
- `--reasoning_bank_path` - Path to bank JSONL
- `--reasoning_index_base` - Base path for FAISS index (without .faiss extension)
- `--reasoning_top_k` (default: 2) - Number of hints to inject
- `--reasoning_domain_filter` (default: True) - Filter by current domain

---

## Common Workflows

### Workflow 1: Build from Success Trajectories Only
```bash
cd $INFERENCE_PROJECT_ROOT

# Build with visual Stage 1
python scripts/reasoning_bank/build_reasoning_bank.py \
  --input_glob "data/downloaded_datasets/webvoyager_memory/Amazon/qwen2.5-vl-32b/test/success/*.jsonl" \
  --dataset webvoyager \
  --domain Amazon \
  --multimodal \
  --visual_stage1 \
  --max_items_per_traj 2
```

### Workflow 2: Build from Success + Failure Trajectories
```bash
# Build with visual Stage 1 (includes both success and failure)
python scripts/reasoning_bank/build_reasoning_bank.py \
  --input_glob "data/downloaded_datasets/webvoyager_memory/Amazon/qwen2.5-vl-32b/test/**/*.jsonl" \
  --dataset webvoyager \
  --domain Amazon \
  --multimodal \
  --visual_stage1 \
  --max_items_per_traj 2
```

### Workflow 3: Build from Multiple Domains
```bash
# Build separate banks per domain
for domain in Amazon Apple GitHub; do
  python scripts/reasoning_bank/build_reasoning_bank.py \
    --input_glob "data/downloaded_datasets/webvoyager_memory/${domain}/*/test/**/*.jsonl" \
    --dataset webvoyager \
    --domain ${domain} \
    --multimodal \
    --visual_stage1 \
    --bank_jsonl "memory/reasoning_bank_mm_${domain}.jsonl" \
    --max_items_per_traj 2
done
```

### Workflow 4: Quick Test Before Full Build
```bash
# 1. Validate setup
python scripts/reasoning_bank/validate_multimodal.py

# 2. Test Stage 1 visual
python scripts/reasoning_bank/test_stage1_visual.py

# 3. Build from small subset (5 trajectories)
python scripts/reasoning_bank/build_reasoning_bank.py \
  --input_glob "data/downloaded_datasets/webvoyager_memory/Amazon/qwen2.5-vl-32b/test/success/Amazon_Amazon_{2,18,51,62,113}.jsonl" \
  --dataset webvoyager \
  --domain Amazon \
  --multimodal \
  --visual_stage1 \
  --max_items_per_traj 2

# 4. Check output
head -1 memory/reasoning_bank_mm.jsonl | python -m json.tool
ls -lh media/reasoning_bank/
```

---

## Inspect Output

### View Bank Items
```bash
# Count items
wc -l memory/reasoning_bank_mm.jsonl

# View first item
head -1 memory/reasoning_bank_mm.jsonl | python -m json.tool

# View all key takeaways
cat memory/reasoning_bank_mm.jsonl | jq -r '.key_takeaway'

# Filter by label
cat memory/reasoning_bank_mm.jsonl | jq 'select(.label == "success")' | jq -r '.key_takeaway'
```

### Check Saved Images
```bash
# List all saved images
ls -lh media/reasoning_bank/*/step_*.jpg

# Count images per task
ls media/reasoning_bank/ | while read task; do 
  echo "$task: $(ls media/reasoning_bank/$task/*.jpg 2>/dev/null | wc -l) images"
done

# View image with eog/feh/etc
eog media/reasoning_bank/Amazon_Amazon_2/step_07.jpg
```

### Check FAISS Index
```bash
# Index metadata
cat memory_index/reasoning_bank_mm.json

# Index size
ls -lh memory_index/reasoning_bank_mm.faiss
```

---

## Troubleshooting

### VLM Server Not Running
```bash
# Check if server is up
curl http://localhost:8000/v1/models

# Start vLLM server (example)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000
```

### Build Fails
```bash
# Check trajectory structure
python -c "
import json
with open('path/to/trajectory.jsonl', 'r') as f:
    obj = json.load(f)
    print('Keys:', list(obj.keys()))
    print('Rounds:', len(obj.get('rounds', [])))
"

# Run validation
python scripts/reasoning_bank/validate_multimodal.py
```

### No Items Generated
- Check VLM is returning valid JSON
- Try with `--max_items_per_traj 3` 
- Test with a single trajectory first
- Check prompt templates exist in `agent/prompts/`

---

## Performance Tuning

### Speed vs Quality Trade-offs

**Fastest (text-only):**
```bash
--multimodal  # No --visual_stage1
# ~8-12 seconds per trajectory
```

**Balanced (multimodal with text Stage 1):**
```bash
--multimodal  # No --visual_stage1
# ~10-15 seconds per trajectory
```

**Best Quality (multimodal with visual Stage 1):**
```bash
--multimodal --visual_stage1
# ~15-30 seconds per trajectory
```

### Batch Processing
```bash
# Process in chunks
for i in {0..4}; do
  python scripts/reasoning_bank/build_reasoning_bank.py \
    --input_glob "data/downloaded_datasets/webvoyager_memory/Amazon/*/test/success/Amazon_Amazon_${i}*.jsonl" \
    --multimodal --visual_stage1 \
    --domain Amazon
done
```

---

## Examples by Use Case

### Research: High Quality Bank
```bash
# Use visual Stage 1 for best quality
python scripts/reasoning_bank/build_reasoning_bank.py \
  --input_glob "data/downloaded_datasets/webvoyager_memory/Amazon/*/test/**/*.jsonl" \
  --multimodal \
  --visual_stage1 \
  --max_items_per_traj 2 \
  --domain Amazon
```

### Production: Fast Build
```bash
# Use text-only Stage 1 for speed
python scripts/reasoning_bank/build_reasoning_bank.py \
  --input_glob "data/downloaded_datasets/webvoyager_memory/Amazon/*/test/**/*.jsonl" \
  --multimodal \
  --max_items_per_traj 2 \
  --domain Amazon
```

### Debugging: Single Trajectory
```bash
# Test on one file
python scripts/reasoning_bank/build_reasoning_bank.py \
  --input_glob "data/downloaded_datasets/webvoyager_memory/Amazon/qwen2.5-vl-32b/test/success/Amazon_Amazon_2.jsonl" \
  --multimodal \
  --visual_stage1 \
  --domain Amazon
```

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `validate_multimodal.py` | Check setup (no VLM) |
| `test_stage1_only.py` | Test text-only Stage 1 |
| `test_stage1_visual.py` | Test visual Stage 1 |
| `test_multimodal_distill.py` | Test full pipeline |
| `build_reasoning_bank.py` | Build the bank |
| `build_reasoning_bank.py --multimodal` | Build with multimodal |
| `build_reasoning_bank.py --multimodal --visual_stage1` | Build with visual Stage 1 (best) |

---

## File Paths

**Input:**
- Trajectories: `data/downloaded_datasets/webvoyager_memory/{domain}/{model}/test/{success,positive,negative,failed}/*.jsonl`

**Output:**
- Bank: `memory/reasoning_bank_mm.jsonl`
- Index: `memory_index/reasoning_bank_mm.faiss`
- Images: `media/reasoning_bank/{task_id}/step_{i:02d}.jpg`
- Metadata: `memory_index/reasoning_bank_mm.json`

**Prompts:**
- Stage 1 text: `agent/prompts/reasoning_bank_mm_identify_steps.md`
- Stage 1 visual: `agent/prompts/reasoning_bank_mm_identify_steps_visual.md`
- Stage 3: `agent/prompts/reasoning_bank_mm_extract.md`

