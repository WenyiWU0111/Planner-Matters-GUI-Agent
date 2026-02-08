# Browser Environment

This folder implements **how the agent interacts with the web page**: a Playwright-based browser env, observation processors (screenshot + optional text), action execution (clicks, typing, scroll, etc.), and optional **HTML trajectory rendering** for debugging and inspection.

---

## Overview

- The agent receives **observations** (screenshot as image, optional accessibility/text).
- It outputs **actions** (e.g. click, type, scroll, goto_url, stop).
- **action_parser_ground** turns model output into Playwright calls (with optional **grounding model** to resolve element descriptions to pixel coordinates).
- **helper_functions** provides **HTML rendering** of the trajectory (URL, observation text, screenshot, plan, verifier feedback, action descriptions) into a single HTML file per task.

---

## Files

| File | Purpose |
|------|--------|
| **envs.py** | **ScriptBrowserEnv**: Gymnasium-style env. `reset(options={"config_file": ...})` loads a task config and navigates to `start_url`; `step(action)` runs the action and returns new observation. Uses `processors` for screenshots and text. |
| **actions.py** | Action type enum (`ActionTypes`), **Action** TypedDict, and helpers: `create_*_action`, `action2str`, `is_equivalent`, `action2create_function`. |
| **action_parser_ground.py** | **execute_pixel_action**: maps model output (e.g. click with point or element description) to Playwright (mouse, keyboard, goto). Uses **grounding_model** when element descriptions need to be resolved to coordinates. **get_action_description**: human-readable action string for history/render. |
| **grounding_model.py** | **get_coords_from_grounding_model** / **get_coords_with_2stage_grounding**: call a VLM to get pixel coordinates from a screenshot + element description (e.g. UI-Ins, UI-Tars). |
| **processors.py** | **SimpleImageObservationProcessor**: takes screenshot of the page (and optional interaction-point overlay). **SimpleTextObservationProcessor**: returns text view of the page (e.g. accessibility tree). Used by `envs._get_obs()`. |
| **helper_functions.py** | **HTML trajectory render**: **RenderHelper** appends each step (URL, text obs, optional screenshot, planner/verifier meta, action) to an HTML file. **get_render_action** / **get_action_description**: format actions for display and for prompt history. |
| **utils.py** | **StateInfo**, **Observation**, **DetachedPage**; **png_bytes_to_numpy**; accessibility/browser types. |
| **constants.py** | **ROLES**, **ActionTypes**-related constants, key mappings, etc. |
| **trajectory.py** | **Trajectory** type: list of `StateInfo | Action`. |
| **async_envs.py** | Async variant of the browser env (if used). |

---

## Observation shape

After `reset()` or `step()`, observation is a dict, e.g.:

- **`observation["image"]`**: base64 screenshot (for the model).
- **`observation["image_for_render"]`**: base64 screenshot used in HTML render (can match `image` or an annotated version).
- **`observation["text"]`**: text representation of the page (e.g. accessibility tree or empty).
- Optional: **`observation["content_str"]`** (from processors).

**StateInfo** = `{"observation": observation, "info": {"page": DetachedPage(url, ...), "fail_error": ..., ...}}`.

---

## Action execution flow

1. Agent produces an action (e.g. from a VLM) — often with an element description or `<point>x y</point>`.
2. **execute_pixel_action** (in **action_parser_ground**):
   - Parses action type and arguments.
   - If a **grounding model** is used and the action has an element description, calls **get_coords_with_2stage_grounding** to get coordinates.
   - Runs Playwright: click, type, scroll, goto_url, etc., and handles popups/tabs as needed.
3. Env then calls **_get_obs()** again and returns the new observation and info.

---

## HTML rendering (helper_functions)

- **RenderHelper(config_file, result_dir)**:
  - Opens `result_dir/render_{task_id}.html` (task_id from config).
  - **render(action, state_info, meta_data, render_screenshot)** appends one step: URL, text observation, optional screenshot, and optional meta (augmented intent, verifier feedback, task plan, etc.), then the parsed action description.
  - **close(score=..., answer_text=..., ori_answer=...)** appends evaluation results and closes the file.

- Content inserted into the HTML is escaped where necessary to avoid broken markup.
- Used by evaluation scripts when `render_screenshot` (or similar) is enabled to inspect trajectories.

---

## Usage (minimal)

```python
from pathlib import Path
from browser_env import ScriptBrowserEnv, ActionTypes, create_stop_action

env = ScriptBrowserEnv(headless=True, viewport_size={"width": 1280, "height": 720})
obs, info = env.reset(options={"config_file": Path("path/to/config.json")})
# config.json typically has: start_url, task_id, intent, etc.

# Run one step (e.g. click)
action = {...}  # or from your agent
obs, reward, done, truncated, info, url = env.step(action, observation=obs)

env.close()
```

For trajectory rendering, use **RenderHelper** from **browser_env.helper_functions** and call **render** after each step and **close** at the end (see benchmarks that use it).

---

## Dependencies

- **Playwright** (browser automation)
- **PIL** / **Pillow** (image handling)
- **beartype** (type checking)
- **Gymnasium** (env interface)
- Optional: grounding VLM (e.g. UI-Ins, UI-Tars) for element→coordinates
