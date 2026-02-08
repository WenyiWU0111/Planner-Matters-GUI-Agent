You are an expert in web navigation and task analysis. You will be given a user task and a complete trajectory showing how an agent attempted to accomplish it.

## Your Goal

Identify 1-2 **key steps** that directly caused the final outcome (success or failure). Focus on causal steps, not every action.

## Guidelines

- A key step is one where the action directly led to:
  - **Success case**: achieving the goal, finding the target, or completing the task
  - **Failure case**: missing the target, triggering an error, or getting stuck in a loop
- Prefer steps with clear state transitions (e.g., filter applied, search executed, item added to cart, error appeared)
- Avoid selecting routine steps like scrolling or waiting unless they directly caused the outcome
- Output exactly 1-2 steps, not more

## Output Format

Your output must be valid JSON (no markdown, no extra text):

```json
[
  {
    "step_index": 7,
    "reason": "Applied color filter which updated product grid to show correct variant"
  },
  {
    "step_index": 12,
    "reason": "Clicked add-to-cart on the filtered product, completing the task"
  }
]
```

If only one key step, output a single-element array.

## Input

Task: {task}
Outcome: {outcome}

Trajectory (text only):
{trajectory_text}

