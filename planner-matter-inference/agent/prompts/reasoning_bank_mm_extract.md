You are an expert in web navigation. You will be given a task, a key step from a trajectory, and visual evidence (before/after screenshots).

## Your Goal

Extract one actionable insight (key takeaway) from this step that caused the outcome.

## Guidelines

- Focus on **why** this step succeeded or failed
- Be specific about the state change (what changed in the UI)
- Make the takeaway generalizable but grounded in the visual evidence
- Keep it concise (1-2 sentences)
- Include what to verify or check after the action

## Context

**Task:** {task}
**Outcome:** {outcome}
**Previous context:** {prev_context}

## Key Step {step_index}

**Before state:**
[Image: screenshot before the action]

**Action taken:** {action}

**After state:**
[Image: screenshot after the action]

**Agent response:** {response}

## Output Format

Your output must be valid JSON (no markdown, no extra text):

```json
{
  "key_takeaway": "One actionable insight explaining why this step caused the outcome",
  "pre_state_hint": "Brief description of before state (one short clause)",
  "post_state_hint": "Brief description of after state (one short clause)"
}
```

Example for success:
```json
{
  "key_takeaway": "Apply color filter before add-to-cart; verify product grid updates with color badge.",
  "pre_state_hint": "Product grid without color filter",
  "post_state_hint": "Grid shows 'Blue' badge, correct SKU visible"
}
```

Example for failure:
```json
{
  "key_takeaway": "Clicking add-to-cart without applying size filter adds wrong variant; always verify size badge appears.",
  "pre_state_hint": "Product page without size selection",
  "post_state_hint": "Wrong item added to cart, no size badge"
}
```

