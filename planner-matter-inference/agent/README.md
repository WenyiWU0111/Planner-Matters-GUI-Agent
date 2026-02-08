# Agent

This folder implements the **GUI agent**: a ReAct-style, function-calling agent that takes a task intent and browser trajectory, calls an LLM (optionally with discrete/continuous memory), and returns the next action (click, type, scroll, stop, etc.) for the browser environment.

---

## Overview

- The **agent** receives the current trajectory (states + actions), task intent, and metadata. It builds a prompt (system message + tools + experience memory + current screenshot + reminders), calls the main LLM, then parses the response into a structured **Action** (e.g. `create_click_action`, `create_stop_action`).
- A separate **tool LLM** is used for page description, login detection, natural-language action parsing, and (in `agent.py`) discrete memory summarization/digestion.
- **llm_config** provides the model backends: vLLM API, CoMEM (continuous memory), or local Transformers checkpoints.

---

## Files

| File | Purpose |
|------|--------|
| **agent.py** | Full **FunctionCallAgent**: discrete memory (ExperienceMemory), reasoning bank, training data collector, verifier, Reflexion-style history summary, stuck detection, AWM workflow memory, hybrid/concept memory hooks. Uses **VerifierTool** and a rich system prompt built from `prompts/`. Entry point: **construct_agent(args)**. |
| **agent_new.py** | Lighter **FunctionCallAgent**: same ReAct + function-calling loop and tools, but no discrete memory, reasoning bank, or Reflexion. Single `_reset_task_state()` and simpler init. Suited for WebVoyager-style runs. Also exposes **construct_agent(args)**. |
| **llm_config.py** | **create_model(args)** → main agent model (vLLM, CoMEM, or Transformers from checkpoint). **load_tool_llm(args, model_name=None)** → tool/auxiliary model (e.g. for verification, page description). Defines **VLLMModel**, **CoMEMModel**, **TransformersModel**, **LLMEvaluator**-compatible chat interface. |

---

## Prompts (`prompts/`)

- **system_prompt.txt** – Base system prompt for the GUI agent (used in `agent.py`).
- **reminders.txt** – Injected each turn with the current task reminder.
- **examples.txt** – Fallback “experience” text when discrete memory is disabled.
- **tool_selection.txt** – Tool list/descriptions for the agent (used in `agent_new.py`; `agent.py` builds tool section from `tool_specs`).
- **reasoning_bank_*.md** – Prompts for reasoning-bank distillation (success/failure, identify steps, visual, etc.).

---

## Main interface

- **construct_agent(args)**  
  Returns a **FunctionCallAgent** instance (from `agent.py` or `agent_new.py` depending on which module is used by the runner).

- **FunctionCallAgent**
  - **next_action_custom(trajectory, intent, meta_data)** → `(action, meta_data)`. Builds messages (including optional discrete memory, reflection, reasoning bank), calls `llm.chat(...)`, then **\_process_response** to get an Action; supports fallback **\_parse_natural_language_with_llm** when the response is not a function call.
  - **reset(test_config_file)** – Clears per-task state (experience_*, last_analysis_result, etc.) before a new task.
  - **check_login(state_info)** – Uses tool_llm to detect login/CAPTCHA pages from the observation image.

- **Tools** (from **tools.gui_tools** and **tools.analysis_tools**) are registered in **NAME_TO_CLS** and instantiated in **function_map**; the agent passes **tool_llm** and **page** where needed (e.g. ContentAnalyzer, MapSearch, PageGoto).

---

## Dependencies

- **browser_env** (Trajectory, Action, ActionTypes)
- **actions** (create_*_action, parse_action_json)
- **tools** (gui_tools, analysis_tools, helper_tools for VerifierTool)
- **memory** (experience_memory, reasoning_bank; optional)
- **utils** (training_data_collector, llm_wrapper; optional)
- **llm_config** (create_model, load_tool_llm)
- **openai**-compatible client (vLLM), **transformers** / **torch** (for CoMEM/Transformers backends)

---

## Usage note

Runners (e.g. under **benchmarks/**) typically call **construct_agent(args)** to build the agent, then run a loop: **env.reset(options={"config_file": ...})**, and repeatedly **agent.next_action_custom(trajectory, intent, meta_data)** and **env.step(action)** until a stop action or termination. Call **agent.reset(config_file)** before each new task config.
