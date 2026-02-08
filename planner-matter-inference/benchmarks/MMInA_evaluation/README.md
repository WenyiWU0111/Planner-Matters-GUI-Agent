# MMInA Evaluation

This folder contains the **evaluation pipeline** for the GUI agent on the MMInA benchmark: test runners that drive the browser env with task configs, and evaluators that score trajectories (string match, URL match, HTML content).

---

## Overview

- **Test runner**: loads task configs (intent, start URL, eval criteria), resets the browser env, runs the agent in a loop until STOP or termination, then runs the evaluator and optionally saves fail reasons and training conversations.
- **Evaluator**: router picks one or more evaluator types from the config (`string_match`, `url_match`, `program_html`); each returns a score and optional answer text; scores are combined by multiplication.

---

## Files

| File | Purpose |
|------|--------|
| **test_runner.py** | **TestRunner**: simple loop — reset env from `config_file`, get actions from `agent.next_action_custom`, step env, then evaluate. Uses `evaluator_router` and `RenderHelper` for HTML trajectory. Exposes `run(config_file_list)`. |
| **test_runner_new.py** | Extended **TestRunner** with: planner-with-memory (plan generation/update), fail-reasons load/save, and training-data collection. Same core loop and evaluation. |
| **evaluator.py** | **evaluator_router(config_file, vllm_client)** builds an **EvaluatorComb** from config `eval.eval_types`. **StringEvaluator**: exact/match must_include/fuzzy (LLM) over last action answer + optional page URL. **URLExactEvaluator**: compare `page.url` to `reference_url` (EXACT or GOLD in PRED). **HTMLContentEvaluator**: navigate to URLs from `program_html`, run locators, check required_contents. All evaluators return `(score, answer_text)`. |
| **helper_functions.py** | **clean_url**, **clean_answer**: normalize URL/answer for comparison. **encode_image**: PIL image → base64 JPEG (for optional VL eval). |

---

## Task config (per task)

Config JSON typically includes:

- **intent**: natural-language task description (and optionally normalized, e.g. Kiwix Wikipedia → wikipedia.org).
- **task_id** / **id**: unique task id (used for render file name and fail reasons).
- **sites**: list of site names (e.g. for logging / training summary).
- **eval**: **eval_types** (e.g. `["string_match", "url_match"]`), **reference_answers** (for string_match), **reference_url** (for url_match), **program_html** (for program_html).

---

## Evaluator types

1. **string_match**  
   Uses **reference_answers** with:
   - **exact_match**: compare cleaned predicted answer to a single reference string.
   - **must_include**: require each of a list of strings to appear in (predicted answer + current page URL).
   - **fuzzy_match**: use the provided **vllm_client** to do binary yes/no semantic match against each reference; combines with multiplication.

2. **url_match**  
   Compares `page.url` (cleaned) to **reference_url** (single URL or several split by ` |OR| `). **url_note**: `EXACT` (pred in ref list) or `GOLD in PRED` (any ref substring of pred).

3. **program_html**  
   For each entry in **program_html**: resolve **url** (or `func:...` with `__last_url__`), optionally goto that URL, then run **locator** (empty = full `page.content()`, or `document....` JS). Check that **required_contents** (literal or ` |OR| ` list) appears in the selected content.

---

## Usage

- Entry point is typically from the parent benchmark/runner that:
  - Builds an **agent** and **args** (result_dir, viewport, max_steps, render_screenshot, etc.).
  - Gets a list of task config paths (e.g. under `MMInA/...`).
  - Instantiates **TestRunner(args, agent)** or the new runner and calls **run(config_file_list)**.

- **TestRunner** is imported from this package; **evaluator_router** is used to get the combined evaluator. Results are logged; fail reasons (new runner) and render HTML are written to **args.result_dir**.

---

## Dependencies

- **browser_env** (ScriptBrowserEnv, RenderHelper, get_action_description, create_stop_action, etc.)
- **agent** and **memory** modules (for the new runner: planner, experience memory, training collector)
- **utils** (early_stop, training_data_collector, etc.)
- **playwright**, **openai** (or the LLM client used by the agent and by StringEvaluator fuzzy match)
