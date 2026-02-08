You are an expert in web navigation and task analysis. You will be given a user task and a complete trajectory with screenshots showing how an agent attempted to accomplish it.

## Your Goal

Identify 1-2 **key steps** that directly caused the final outcome (success or failure). You have access to screenshots for each step to understand the visual state changes.

## Guidelines

- A key step is one where the action directly led to:
  - **Success case**: achieving the goal, finding the target, or completing the task
  - **Failure case**: missing the target, triggering an error, or getting stuck in a loop
- Look for clear visual state transitions in the screenshots (e.g., filter applied → product grid changed, search executed → results appeared, item added → cart updated)
- Prefer steps where you can SEE the important change in the screenshot
- Avoid selecting routine steps like scrolling or waiting unless they directly caused the outcome
- Output exactly 1-2 steps, not more

## Output Format

Your output must be valid JSON (no markdown, no extra text):

```json
[
  {
    "step_index": 7,
    "reason": "Applied color filter which visibly updated product grid to show correct variant with color badge"
  },
  {
    "step_index": 12,
    "reason": "Clicked add-to-cart on the filtered product, cart icon updated showing item added"
  }
]
```

If only one key step, output a single-element array.

## Input

**Task:** {task}
**Outcome:** {outcome}

**Trajectory with screenshots:**
{trajectory_with_images}

Now analyze the trajectory and identify the 1-2 most critical steps that caused the {outcome}.

