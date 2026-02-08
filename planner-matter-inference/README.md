# Planner-Matter-Inference

A **GUI agent** that learns from experience: it uses a ReAct-style, function-calling loop to control a browser (click, type, scroll, etc.), with optional **memory** (discrete takeaways, planner-with-memory, reasoning bank) and **planning** to improve over trajectories.

---

## Brief introduction

This folder implements an agent that:

- Interacts with web pages via **Playwright** (browser_env).
- Uses **vision-language models** (e.g. Qwen2.5-VL) to choose the next action from the current screenshot and task intent.
- Can use **experience memory**: similar past trajectories are retrieved and summarized to guide planning and action selection.
- Supports **planner-with-memory**: a planner generates/updates a step-by-step plan using retrieved similar tasks; the actor follows the plan.
- Is evaluated on **WebVoyager**, **MMInA**, and **Mind2Web**-style benchmarks.

Main components: **agent** (ReAct + tools), **browser_env** (Playwright env, observation processors, action execution), **memory** (discrete takeaways, FAISS index, plan_with_memory), **benchmarks** (evaluation runners and LLM-based evaluators).

---

## Dependencies

- **Python 3.10** (recommended)
- **PyTorch**, **transformers**, **vLLM** (or compatible API) for LMs
- **Playwright** (Chromium) for browser automation
- **faiss-cpu** (or faiss-gpu) for memory retrieval
- **python-dotenv**, **openai**, **gymnasium**, and others (see `requirements.txt`)

Install from the project root:

```bash
pip install -r requirements.txt
playwright install chromium
```

---

## Docker

The repo includes a **Dockerfile** and **docker-compose.yml** for a GPU-enabled environment (CUDA 12.6, vLLM, flash_attn, Playwright).

- **Build**: `docker compose build` or `./docker-run.sh build`
- **Start**: `docker compose up -d` or `./docker-run.sh start`
- **Shell**: `docker exec -it gui-agent-container bash` or `./docker-run.sh shell`
- **Stop**: `docker compose down` or `./docker-run.sh stop`

Run `./docker-run.sh` with no arguments for an interactive menu. The project is mounted at `/workspace`; set `.env` and data paths as needed.

---

## Step-by-step setup and run

### 1. Deploy vLLM models

Start the vLLM OpenAI-compatible API servers for the **planner** and (optionally) the **grounding/UI** model. Run in separate terminals or in the background.

**Planner model** (e.g. Qwen2.5-VL-7B) on port 8000:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8000
```

**UI / grounding model** (e.g. UI-Ins-7B) on port 8001:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model Tongyi-MiA/UI-Ins-7B \
  --port 8001
```

Adjust `--model` and `--port` if you use different models or ports; ensure the run scripts and `.env` point to the same URLs (e.g. `http://localhost:8000/v1`, `http://localhost:8001/v1`).

### 2. Prepare memory JSON and FAISS index

Build the **discrete summary** and **planner FAISS index** so the agent can retrieve similar past tasks. See **`memory/README.md`** for full options.

**2a – Put experience trajectories in place**

- Place trajectories under **`data/trajectories`** (or set **`MEMORY_DATA_DIR`** in `.env`).
- For format and example data: **WenyiWU0111/webvoyager_memory** on Hugging Face (WebVoyager-style trajectories).

**2b – Precompute discrete takeaways**

From the **planner-matter-inference** root:

```bash
python -m memory.precompute_takeaways \
  --memory_root data/trajectories \
  --server_url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --out_json discrete_summary.json \
  --overwrite
```

This produces **`discrete_summary.json`** (task intents, keywords, steps).

**2c – Build planner FAISS index**

```bash
python -m memory.experience_memory_planner \
  --summary_json discrete_summary.json \
  --output_index memory_index/simple_text
```

This creates the FAISS index (e.g. `memory_index/simple_text.faiss` and related files). The agent loads this when using `--use_planner_with_memory`.

### 3. Set environment variables and load them

- Edit **`.env`** in the project root and set at least:
  - **`OPEN_ROUTER_API_KEY`** — (or your LM API key) for model API calls.
  - **`INFERENCE_PROJECT_ROOT`** — Inference project root (e.g. `./`).
  - **`MEMORY_DATA_DIR`** — Directory of experience trajectories (e.g. `data/trajectories`).
  - **`FAISS_INDEX_PATH`** — Path to the planner FAISS index (e.g. `memory_index/simple_text`).
  - **`DISCRETE_SUMMARY_PATH`** — Path to `discrete_summary.json` (e.g. `discrete_summary.json`).
  - **`CHECKPOINT_PATH`** — Path to the main agent/actor checkpoint when using a local model.

- **Load** the variables into your shell before running scripts:

```bash
source scripts/load_env.sh
```

Or from the repo root: `source planner-matter-inference/scripts/load_env.sh`. Do **not** use `export $(cat .env | xargs)` (it breaks on comments).

### 4. Run the bash scripts

Execution entry points are in **`scripts/bash/`**. From the **planner-matter-inference** root (after loading `.env`):

```bash
cd scripts/bash
./run_planner_baseline.bash    # Planner with memory (WebVoyager domains)
# or
./run_planner_rl.bash          # Planner + RL agent
./run_planner_rl_comem.bash   # Planner + RL + CoMEM
./run_baseline.bash            # Baseline
./run_baseline_RL.bash         # Baseline RL
./run_continuous.bash          # Continuous memory
```

Each script sets `--domain`, `--evaluation_type`, `--use_planner_with_memory`, and related flags, then calls **`run.py`**. Ensure `.env` and the vLLM URLs (Step 1) match the models used by the scripts.

---

## Project layout (summary)

- **`agent/`** — ReAct agent, tool map, prompts; optional discrete memory, reasoning bank, Reflexion.
- **`browser_env/`** — Playwright env, observation processors, action execution, trajectory HTML render.
- **`memory/`** — Discrete takeaways, FAISS index, plan_with_memory (generate/update plan from similar tasks).
- **`benchmarks/`** — MMInA and WebVoyager evaluation (test runners, evaluators, LLM evaluation).
- **`config/`** — Argument parsing and run configuration.
- **`tools/`** — GUI tools (click, type, scroll, etc.) and analysis tools (map search, content analyzer).
- **`scripts/`** — Reasoning bank build, runners (e.g. run_webvoyager.sh, run_mmina_*.sh).
- **`scripts/bash/`** — Run scripts (planner baseline, RL, continuous, etc.).
- **`run.py`** — Main entry point used by the bash scripts.

For more detail, see the READMEs in **`agent/`**, **`browser_env/`**, **`memory/`**, and **`benchmarks/`**.
