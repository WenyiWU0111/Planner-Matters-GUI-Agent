# Planner Matters! An Efficient and Unbalanced Multi-agent Collaboration Framework for Long-horizon Planning

Official implementation for the paper **"Planner Matters! An Efficient and Unbalanced Multi-agent Collaboration Framework for Long-horizon Planning"**.

---
## Framework overview

<img src="media/framework.png" alt="Framework overview" style="width: 80%;">

Language model (LM)–based GUI agents have shown promising results in automating complex tasks from natural language instructions, but they still struggle with **long-horizon planning and reasoning**. We address this by proposing a **human-like multi-agent framework** that decomposes GUI automation into:

- **Planner** — high-level step-by-step planning
- **Actor** — low-level execution (click, type, scroll, etc.)
- **Memory manager** — retrieval and use of past experience

This decomposition enables structured decision-making over long interaction horizons and substantially improves performance on complex GUI tasks. We find that **planning is the dominant factor**: execution and memory can be handled by smaller models with minimal performance loss. We therefore introduce a **planner-centric reinforcement learning** approach that optimizes only the planner with trajectory-level rewards from a **VLM-as-judge**, while keeping the actor and memory frozen. Concentrating model capacity and learning on high-level planning yields robust, compute-efficient gains in long-horizon GUI automation.

**Repository layout:**

- **`planner-matter-inference/`** — Inference and evaluation (browser env, agent, memory, benchmarks, run scripts).
- **`planner-matter-RL/`** — Planner-centric RL training (Ray, PPO, vLLM rollouts, VLM-as-judge).
- **`planner-matter-sft/`** — SFT / LoRA training utilities.
- **`OSWorld/`** — Experiments on the OSWorld benchmark.
- **`mcp-bench/`** — Experiments on the MCP-Bench benchmark.

---

## Setting up

- **Python 3.10+**
- **PyTorch**, **transformers**, **Playwright** (Chromium), **faiss-cpu** (or faiss-gpu), and dependencies in each subproject’s `requirements.txt`.

**vLLM:** Used for serving the planner and (optionally) the grounding/UI model. Install vLLM in the environment you use for inference or training.

- For **Qwen2.5-VL** and common models: the versions in `planner-matter-inference/requirements.txt` and `planner-matter-RL/requirements.txt` are sufficient.
- For **Qwen3-VL**: install **`vllm>=0.11.0`** (e.g. `pip install "vllm>=0.11.0"`).

**Docker:** Both **`planner-matter-inference/`** and **`planner-matter-RL/`** provide a **Dockerfile** and **docker-compose.yml** (GPU-enabled, CUDA 12.x, vLLM, Playwright). Build and run via `docker compose build` / `docker compose up -d` or the helper script `./docker-run.sh` (build, start, shell, etc.) in each subproject. See the **Training** and **Inference** sections (and each subproject’s README) for Docker usage.

Each subproject can be set up independently; see **Training** and **Inference** below.

---

## Training

Planner-centric RL training lives in **`planner-matter-RL/`**. It uses Ray for distributed rollouts, vLLM for the planner (and actor), and a VLM-as-judge for trajectory-level rewards.

**Quick start:**

1. Install dependencies and (optionally) use Docker — see [`planner-matter-RL/README.md`](planner-matter-RL/README.md).
2. Set `DATA_PATH`, `OPENROUTER_API_KEY` (or `API_KEY`), and optionally `MEMORY_SUMMARY_PATH`, `MEMORY_INDEX_PATH` (discrete planner memory).
3. From `planner-matter-RL/` run:

```bash
bash scripts/train_osworld_test-time/ray_launch.sh
```

This starts a Ray head and runs PPO via `train.sh` (see `scripts/train_osworld_test-time/train.sh` for hyperparameters and `--use_planner_with_memory`). For Slurm clusters, use `scripts/train_osworld_test-time/slurm_launch.sh`.

**Memory for the planner:** Build the discrete summary and FAISS index with the pipeline in **`planner-matter-inference`** (e.g. `memory/precompute_takeaways.py` and `memory/experience_memory_planner.py`), then point `MEMORY_SUMMARY_PATH` and `MEMORY_INDEX_PATH` to the generated files (or copy them into `planner-matter-RL/memory_for_planner/`).

Full details: **[planner-matter-RL/README.md](planner-matter-RL/README.md)**.

---

## Inference

Inference and evaluation are in **`planner-matter-inference/`**: browser env, ReAct agent, planner-with-memory, and benchmarks (WebVoyager, MMInA, Mind2Web-style).

**High-level steps:**

1. **Deploy vLLM** — e.g. planner on port 8000, UI/grounding model on port 8001 (see [`planner-matter-inference/README.md`](planner-matter-inference/README.md)).
2. **Prepare memory** — Build `discrete_summary.json` and the planner FAISS index (see `planner-matter-inference/memory/`).
3. **Set `.env`** — Configure API keys, paths (`FAISS_INDEX_PATH`, `DISCRETE_SUMMARY_PATH`, etc.), then run `source scripts/load_env.sh` from `planner-matter-inference/`.
4. **Run scripts** — From `planner-matter-inference/scripts/bash/` run e.g. `./run_planner_baseline.bash`, `./run_planner_rl.bash`, or `./run_continuous.bash`.

Full step-by-step instructions: **[planner-matter-inference/README.md](planner-matter-inference/README.md)**.

---

## Checkpoints and trajectory dataset

We provide the following on **Hugging Face** for reproducibility:

**Planner / agent checkpoints**

- [**WenyiWU0111/Qwen2_5_7B_RL_planner**](https://huggingface.co/WenyiWU0111/Qwen2_5_7B_RL_planner)
- [**WenyiWU0111/Qwen2_5_3B_RL_planner**](https://huggingface.co/WenyiWU0111/Qwen2_5_3B_RL_planner)
- [**WenyiWU0111/Qwen3_8B_RL_planner**](https://huggingface.co/WenyiWU0111/Qwen3_8B_RL_planner)

Use these as the planner (or actor) checkpoint in `.env` (`CHECKPOINT_PATH`) or in training (`MODEL_PATH` / `--pretrain`).

**Trajectory dataset (experience memory)**

- [**WenyiWU0111/webvoyager_memory**](https://huggingface.co/WenyiWU0111/webvoyager_memory)
- [**WenyiWU0111/CoMEM-agent-memory-trajectories**](https://huggingface.co/datasets/WenyiWU0111/CoMEM-agent-memory-trajectories)

These trajectories can be downloaded to `planner-matter-inference/data/trajectories`, set `MEMORY_DATA_DIR`, and run `memory/precompute_takeaways.py` and `memory/experience_memory_planner.py`).

---

## Experiments on OSWorld and MCP-Bench

Experiments and evaluation runs on the **OSWorld** and **MCP-Bench** benchmarks are provided in the respective folders:
- **`OSWorld/`** — Contains scripts, configs, and results for OSWorld experiments.
- **`mcp-bench/`** — Contains scripts, configs, and results for MCP-Bench experiments.

---

## TODO

- **Continuous memory embedding generation scripts** — Scripts to prepare continuous memory embeddings will be released soon.

---

## Citation

If you use this code or the planner-centric framework, please cite our paper:

```bibtex
@article{planner-matters-2025,
  title   = {Planner Matters! An Efficient and Unbalanced Multi-agent Collaboration Framework for Long-horizon GUI Planning},
  author  = {Wenyi Wu* and Sibo Zhu* and Kun Zhou and Biwei Huang},
  journal = {...},
  year    = {2026}
}
```
