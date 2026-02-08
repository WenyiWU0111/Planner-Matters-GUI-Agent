# Memory Module

This module implements **discrete takeaway memories** for the **planner** in our multi-agent GUI framework (planner, actor, memory manager). The planner uses these memories when generating and updating plans.

---

## Pipeline (3 steps)

### Step 1: Precompute discrete takeaways

Run over all successful trajectories under a `memory_root` to produce **discrete_summary.json**. Each trajectory is summarized into keywords and short steps via a VLM.

```bash
python -m memory.precompute_takeaways \
  --memory_root /path/to/trajectories \
  --server_url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --out_json discrete_summary.json \
  --overwrite
```

- **Input**: Directory tree containing `success/` folders with `*.jsonl` trajectory files.
- **Output**: `discrete_summary.json` (task_id → intent, keywords, steps, src).

Use `--overwrite` to replace an existing file; omit it to avoid overwriting. Optional: `--max_pairs_per_traj`, `--temperature`, `--max_tokens`.

---

### Step 2: Build planner FAISS index

Build the FAISS index (and optional metadata) from **discrete_summary.json** for fast similarity search at runtime.

```bash
python -m memory.experience_memory_planner \
  --summary_json discrete_summary.json \
  --output_index memory_index/simple_text
```

- **Input**: `discrete_summary.json` from Step 1.
- **Output**: `memory_index/simple_text.faiss`, `.embeddings.npy`, `.json` (or the path you pass to `--output_index`).

If `--output_index` is omitted, the index is still built and saved under `memory_index/simple_text_<N>`. Use `--load_existing` to load an existing index and optionally re-save it elsewhere.

---

### Step 3: Use at agent runtime

When running the agent, the following are used from **plan_with_memory.py**:

- **`generate_plan_with_memory`**: Retrieves similar tasks from the planner memory, formats them, and generates a step-by-step plan (with optional screenshot).
- **`update_plan_with_memory`**: Updates the plan given recent observations and action history; supports adaptive memory focus and history-context hints.
- **`extract_history_context`**: Builds structured context (successes, failures, stuck patterns, avoid list) from action history for plan updates.

The agent loads the planner memory by instantiating **`ExperienceMemorySimple`** with:

- `summary_json_path`: path to `discrete_summary.json`
- `faiss_index_path`: path to the index prefix (e.g. `memory_index/simple_text_152`) produced in Step 2

No need to run Step 1 or 2 again unless you change trajectories or want to rebuild the index.

---

## File roles

| File | Role |
|------|------|
| **precompute_takeaways.py** | Step 1: Trajectories → discrete_summary.json (keywords + steps per task). |
| **experience_memory_planner.py** | Step 2: CLI to build (and optionally save) FAISS index from discrete_summary.json. |
| **experience_memory_planner.py** | **ExperienceMemorySimple** class + Step 2 CLI: loads discrete_summary.json, builds/loads FAISS index, provides `retrieve_similar_tasks()`. |
| **plan_with_memory.py** | Step 3 (runtime): `generate_plan_with_memory`, `update_plan_with_memory`, `extract_history_context`. |
| **experience_memory.py** | **ExperienceMemory** (legacy): original class over raw trajectories; same retrieval idea (CLIP + FAISS), different input (conversations with images). |
| **help_functions.py** | CLIP text/multimodal similarity and embeddings used by both planner memory and legacy Memory. |

---

## Index layout

After Step 2 you typically have:

```
memory_index/
  simple_text_<N>.faiss
  simple_text_<N>.embeddings.npy
  simple_text_<N>.json
```

`<N>` is the number of vectors in the index. At runtime, pass the **prefix** (e.g. `memory_index/simple_text_152`) as `faiss_index_path`; the loader appends `.faiss`, `.embeddings.npy`, `.json`.

---

## Environment / paths

- **DISCRETE_SUMMARY_PATH**: default path to `discrete_summary.json` (used by scripts and examples if not overridden by CLI).
- **FAISS_INDEX_PATH**: default planner index prefix (e.g. `memory_index/simple_text_152`).

CLI flags override these where applicable.
