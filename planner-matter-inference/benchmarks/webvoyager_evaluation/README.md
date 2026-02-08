# WebVoyager Evaluation

This folder contains the **WebVoyager-style evaluation** for the GUI agent: run tasks in the browser, render trajectories to HTML, and score success/failure with an LLM that looks at the last 5 screenshots and the final answer.

---

## Overview

- **Test runners** drive the browser env with task configs (one intent per task), collect trajectories, and write a render HTML per task. After each task they call the **LLM evaluator** to get a success/failure score from the render file (no live page).
- **LLMEvaluator** reads `result_dir/render_{task_id}.html`, extracts images and the `finished(answer=...)` text, sends the last 5 screenshots + task + answer to a VL model, and parses `<result>SUCCESS</result>` or `NOT SUCCESS` to return `(score, answer_text, ori_answer)`.
- **llm_evaluation.py** is a standalone script to run the same evaluator in batch over a folder of configs and existing render HTML (e.g. for re-scoring or offline evaluation).

---

## Files

| File | Purpose |
|------|--------|
| **test_runner.py** | **TestRunner**: reset env from config, run agent loop (with optional retries and login reset), step with `tool_llm` for page-change check. After the run, call **LLMEvaluator(config_file, result_dir)** and close render with score/answer. Optional: **Reasoning Bank** distillation, **workflow memory** (AWM) distillation, training-data collection. |
| **test_runner_new.py** | Extended **TestRunner** with: **planner-with-memory** (generate/update plan), **fail_reasons** load/save, **history context** for planner, **adaptive memory**. Same evaluation and render flow. |
| **evaluator.py** | **LLMEvaluator(vllm_client)**. `__call__(config_file, html_folder)` → reads config for task_id/intent, reads `html_folder/render_{task_id}.html`, extracts images and answer, builds VL messages (last 5 images + task + answer), returns `(score, answer_text, ori_answer)`. Helpers: `extract_and_validate_images`, `extract_answer`. |
| **llm_evaluation.py** | **LLMEvaluation(html_folder, args)**: loads existing `llm_evaluation.json` for seen task_ids, iterates over **config_folder** JSONs, skips already-evaluated tasks, runs **LLMEvaluator** for each and appends to results, writes `llm_evaluation.json` after each task. Run as script with `--html_folder`, `--config_folder`, `--model`. |

---

## Task config (per task)

Config JSON typically includes:

- **intent**: natural-language task (e.g. “Find the price of …”).
- **task_id** (or **id**): used for `render_{task_id}.html` and evaluator.
- **site** (WebVoyager) or **sites** (single-element list for Mind2Web-style): domain/site name.
- **start_url** (or similar): used by the env for initial navigation.

---

## Evaluation flow

1. During a run, **RenderHelper** (from `browser_env`) appends each step to `result_dir/render_{task_id}.html` (screenshots, actions, etc.).
2. After the trajectory ends, the test runner calls  
   `score, answer_text, ori_answer = evaluator(config_file, self.args.result_dir)`  
   so **html_folder** is the same as **result_dir**.
3. **LLMEvaluator** opens the render HTML, takes the last 5 images and the `finished(answer=...)` string, sends them to the VL client with a fixed system/user prompt, and parses the model output for `<result>SUCCESS</result>` or `NOT SUCCESS` to set score to 1.0 or 0.0.

---

## Usage

- **From the main benchmark/runner**: build an agent and args (e.g. `result_dir`, `viewport_width`, `max_steps`, `render_screenshot`, `slow_mo`). Get a list of task config paths. Instantiate `TestRunner(args, agent)` or the new runner and call `run(config_file_list)`. Results and render HTML are under `args.result_dir`; the evaluator is called automatically after each task.
- **Offline batch evaluation**: ensure render HTML files exist under `html_folder` (e.g. `results/webvoyager/.../render_*.html`). Run  
  `python llm_evaluation.py --html_folder <path> --config_folder <path> --model <name>`  
  Results are written to `html_folder/llm_evaluation.json`.

---

## Dependencies

- **browser_env** (ScriptBrowserEnv, RenderHelper, get_action_description, create_stop_action)
- **agent** (llm_config for load_tool_llm / create_qwenvl_model)
- **utils** (early_stop, etc.)
- **memory** (reasoning_bank, plan_with_memory, experience_memory_planner) for test_runner_new and optional features in test_runner
- **playwright**; VL model server (e.g. Qwen2.5-VL) used by the env for page-change check and by **LLMEvaluator**
