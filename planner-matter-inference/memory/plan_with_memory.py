import json
import urllib.request
import sys
import os
import logging
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.experience_memory_planner import ExperienceMemorySimple

# Setup logger
logger = logging.getLogger("logger")


def _vllm_chat_completion(
    *,
    server_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    api_key: str = "EMPTY",
) -> str:
    # server_url is expected like http://localhost:8004/v1
    base = server_url.rstrip("/")
    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": 1,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["Authorization"] = "Bearer EMPTY"
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = resp.read().decode("utf-8")
    obj = json.loads(body)
    choice0 = obj["choices"][0]
    message = choice0["message"]
    content = message["content"]
    if not isinstance(content, str):
        raise ValueError("vLLM response message.content must be a string")
    return content


def _format_memory_steps(similar_tasks: List[Dict[str, Any]]) -> str:
    """Format the steps from similar tasks into a readable string."""
    formatted_memories = []
    
    for i, task in enumerate(similar_tasks, 1):
        intent = task.get('intent', '')
        steps = task.get('steps', [])
        similarity_score = task.get('similarity_score', 0.0)
        
        memory_text = f"Memory {i} (Similarity: {similarity_score:.3f}):\n"
        memory_text += f"Task: {intent}\n"
        memory_text += "Steps:\n"
        
        for step in steps:
            memory_text += f"  {step}\n"
        
        formatted_memories.append(memory_text)
    
    return "\n".join(formatted_memories)


def extract_history_context(
    action_results: List[Dict],
    max_recent: int = 10,
) -> Dict[str, Any]:
    """
    Extract structured context from action history.
    
    This function analyzes past action results to identify:
    - Which actions succeeded vs failed
    - Stuck patterns (same action failing repeatedly)
    - Actions to avoid (failed 2+ times)
    
    Args:
        action_results: List of dicts with keys: action, success, reasoning, step
        max_recent: Maximum number of recent actions to consider (default: 10)
    
    Returns:
        Dict containing:
            - completed_actions: List[str] - Successful actions
            - failed_actions: List[str] - Failed actions
            - completed_text: str - Formatted string of recent successes
            - failed_text: str - Formatted string of recent failures with reasons
            - progress_summary: str - Human-readable summary
            - stuck_pattern: Optional[str] - Detected stuck behavior
            - avoid_list: List[str] - Actions to avoid
    """
    # Get recent actions (up to max_recent)
    recent = action_results[-max_recent:] if len(action_results) > max_recent else action_results
    
    # Separate completed and failed actions
    completed = [r for r in recent if r.get('success', False)]
    failed = [r for r in recent if not r.get('success', False)]
    
    # Detect stuck pattern: same action failed 2+ times consecutively
    stuck_pattern = None
    if len(action_results) >= 3:
        last_actions = [r.get('action', '') for r in action_results[-3:]]
        # Check if last 3 actions are the same and at least one failed
        if len(set(last_actions)) == 1:
            # Check if any of these were failures
            last_results = [r.get('success', False) for r in action_results[-3:]]
            if last_results.count(False) >= 2:
                stuck_pattern = f"Repeating same action: {last_actions[0]}"
    
    # Build avoid list from repeated failures (failed 2+ times total)
    action_fail_counts = {}
    for r in action_results:
        if not r.get('success', False):
            action = r.get('action', '')
            action_fail_counts[action] = action_fail_counts.get(action, 0) + 1
    avoid_list = [a for a, count in action_fail_counts.items() if count >= 2]
    
    # Format outputs for prompt injection
    completed_text = "\n".join([f"  - {r.get('action', '')}" for r in completed[-5:]]) or "None yet"
    failed_text = "\n".join([
        f"  - {r.get('action', '')}: {r.get('reasoning', 'No reason provided')}" 
        for r in failed[-3:]
    ]) or "None"
    
    # Build progress summary
    total_actions = len(recent)
    success_count = len(completed)
    fail_count = len(failed)
    progress_summary = f"Completed {success_count}/{total_actions} actions successfully, {fail_count} failed"
    
    logger.info(f"[HistoryContext] {progress_summary}")
    if stuck_pattern:
        logger.warning(f"[HistoryContext] Stuck pattern detected: {stuck_pattern}")
    if avoid_list:
        logger.info(f"[HistoryContext] Actions to avoid: {avoid_list}")
    
    return {
        'completed_actions': [r.get('action', '') for r in completed],
        'failed_actions': [r.get('action', '') for r in failed],
        'completed_text': completed_text,
        'failed_text': failed_text,
        'progress_summary': progress_summary,
        'stuck_pattern': stuck_pattern,
        'avoid_list': avoid_list,
    }


def check_and_update_memory_focus(
    original_intent: str,
    recent_screenshots: List[str],
    recent_actions: List[str],
    current_memories: str,
    vlm: Any,
) -> tuple:
    """
    Use VLM to decide if memory focus needs updating based on current visual state.
    
    Args:
        original_intent: The original task intent/query
        recent_screenshots: List of base64 encoded screenshots (recent history)
        recent_actions: List of recent action descriptions
        current_memories: Formatted string of currently retrieved memories
        vlm: VLM instance for analysis (tool_llm)
    
    Returns:
        Tuple of (needs_update: bool, new_focus: Optional[str])
        - needs_update: True if memory focus should be updated
        - new_focus: Brief description of what experience would help now (if needs_update)
    """
    # Format recent actions for the prompt (last 3)
    recent_actions_text = "\n".join([f"  - {action}" for action in recent_actions[-3:]])
    
    prompt = f"""You are helping a GUI agent decide if it should look for different reference experiences.

Original Task: {original_intent}

Recent Actions:
{recent_actions_text}

Currently Retrieved Memories (for reference only):
{current_memories}

IMPORTANT RULES:
1. DEFAULT TO "NO_UPDATE" - only update in rare cases
2. Memories are just loose references - they don't need to match perfectly
3. If current memories are even loosely related to the task, output NO_UPDATE
4. Only output NEEDS_UPDATE if the agent has moved to a COMPLETELY DIFFERENT activity type

ONLY update if ALL of these are true:
- Agent moved to a fundamentally different activity (e.g., searching → payment checkout, browsing → login form)
- Current memories provide ZERO useful reference
- A different type of experience would clearly help

CRITICAL: If you output NEEDS_UPDATE, provide ONLY 2-5 SHORT keywords (NOT the full task description).
NEVER repeat or paraphrase the original task. Only output action-type keywords.

Output format:
- "NO_UPDATE" (this should be your answer 90% of the time)
- "NEEDS_UPDATE: <2-5 keywords only>" (rare, only for major activity changes)

Good NEEDS_UPDATE examples:
- "NEEDS_UPDATE: payment checkout form"
- "NEEDS_UPDATE: login authentication"
- "NEEDS_UPDATE: dropdown selection filtering"

Bad NEEDS_UPDATE examples (DO NOT DO THIS):
- "NEEDS_UPDATE: Find a yoga mat with purple color..." (NO! This repeats the task)
- "NEEDS_UPDATE: Search for products on Amazon..." (NO! Too long and generic)

Response:"""

    # Build message with recent screenshots
    content = [{"type": "text", "text": prompt}]
    
    # Add recent screenshots (last 3)
    screenshots_to_use = recent_screenshots[-3:] if len(recent_screenshots) > 3 else recent_screenshots
    for i, screenshot in enumerate(screenshots_to_use):
        if screenshot:
            image_url = f"data:image/png;base64,{screenshot}" if not screenshot.startswith("data:image") else screenshot
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
    
    messages = [{"role": "user", "content": content}]
    
    try:
        response, _, _ = vlm.chat(messages=messages, stream=False)
        response_text = response.content if hasattr(response, 'content') else str(response)
        response_text = response_text.strip()
        
        logger.info(f"[AdaptiveFocus] VLM response: {response_text}")
        
        if "NEEDS_UPDATE:" in response_text:
            # Extract the new focus description
            new_focus = response_text.split("NEEDS_UPDATE:")[-1].strip()
            # Clean up any extra quotes or punctuation
            new_focus = new_focus.strip('"\'')
            return True, new_focus
        else:
            return False, None
            
    except Exception as e:
        logger.error(f"[AdaptiveFocus] Error in VLM call: {e}")
        # Fallback to no update on error
        return False, None


def generate_plan_with_memory(
    query: str,
    memory: ExperienceMemorySimple,
    server_url: str,
    model: str,
    planner_model: Any, # loaded planner model instance, used when we load the planner model from the checkpoint
    screenshot: str,
    similar_num: int = 10,
    use_continuous_memory: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    previous_fail_reasons: Optional[str] = None,
    api_key: str = "EMPTY",
) -> str:
    """
    Generate a step-by-step plan for a given query using memory retrieval.
    
    Args:
        query: The current task/query to plan for
        memory: ExperienceMemorySimple instance for retrieving similar tasks
        server_url: vLLM server URL (e.g., http://localhost:8004/v1)
        model: Model name served by vLLM
        planner_model: Planner model instance
        screenshot: Base64 encoded screenshot image of current page state (required)
        similar_num: Number of similar tasks to retrieve (default: 10)
        temperature: Temperature for generation (default: 0.7)
        max_tokens: Maximum tokens for generation (default: 2000)
        previous_fail_reasons: Optional previous failure reasons for this task
    
    Returns:
        Generated plan as a string
    """
    # Retrieve similar tasks from memory
    logger.info(f"[Planner] Retrieving {similar_num} similar tasks for query: {query}")
    similar_tasks = memory.retrieve_similar_tasks(query, similar_num=similar_num)
    file_id_list = [task['task_id'] for task in similar_tasks]
    
    if not similar_tasks:
        logger.warning("[Planner] No similar tasks found. Generating plan without memory.")
        memory_steps_text = ""
    else:
        logger.info(f"[Planner] Found {len(similar_tasks)} similar tasks")
        # Format memory steps
        memory_steps_text = _format_memory_steps(similar_tasks)
        logger.info(f"[Planner] Retrieved memories:\n{memory_steps_text}")
    
    # Build the planning prompt
    fail_reasons_section = ""
    if previous_fail_reasons:
        fail_reasons_section = f"""

Previous Failure Reasons (learn from these mistakes):
{previous_fail_reasons}

IMPORTANT: When generating the plan, make sure to avoid the mistakes mentioned in the previous failure reasons above.
"""
    
    planning_prompt = f"""You are a task planning assistant. Given a current task and similar past experiences, generate a concise and actionable step-by-step plan.

Current Task:
{query}

Similar Past Experiences:
{memory_steps_text if memory_steps_text else "No similar experiences found."}
{fail_reasons_section}
Based on the current task and the similar experiences above, generate a step-by-step plan that:
1. Is concise and actionable, no more than 3-5 steps!
2. Breaks down the task into clear, sequential steps
3. Uses insights from similar experiences when relevant
4. Each step should be specific and executable
5. Do not include actions like Open a web browser, click the search bar, click the search button. Because they will be conducted automatically.

Format your response as a numbered list of steps (step1, step2, step3, etc.). Each step should be clear and actionable.

Plan:"""

    # Build messages with screenshot (required)
    user_content = [
        {"type": "text", "text": planning_prompt}
    ]
    # Format screenshot for vLLM (assuming base64 string)
    if screenshot.startswith("data:image"):
        image_url = screenshot
    else:
        image_url = f"data:image/png;base64,{screenshot}"
    user_content.append({
        "type": "image_url",
        "image_url": {"url": image_url}
    })
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates concise, actionable step-by-step plans for tasks based on current requirements and past experiences. You can see the current screen state to make more informed plans."
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    
    # Generate plan using the planner model
    logger.info("[Planner] Generating plan using planner model...")
    if planner_model:
        if use_continuous_memory:
            plan, _, _ = planner_model.chat(messages=messages, stream=False, file_id_list=file_id_list)
        else:
            plan, _, _ = planner_model.chat(messages=messages, stream=False)
        plan = plan.content if hasattr(plan, 'content') else str(plan)
    else:
        plan = _vllm_chat_completion(
            server_url=server_url,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    
    logger.info(f"[Planner] Generated plan:\n{plan.strip()}")
    return plan.strip(), memory_steps_text, file_id_list


def update_plan_with_memory(
    plan: str,
    query: str,
    memory_steps_text: str,
    file_id_list: List[str],
    action_history: List[str],
    trajectory: List[Any],
    tool_llm: Any = None,
    history_context: Optional[Dict[str, Any]] = None,
    memory: Optional[ExperienceMemorySimple] = None,
    server_url: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    planner_model: Any = None, # loaded planner model instance, used when we load the planner model from the checkpoint
    similar_num: int = 5,
    use_continuous_memory: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 200,
    previous_fail_reasons: str = 'NA',
    focus_check_threshold: int = 3,
    api_key: str = "EMPTY",
) -> tuple[str, str, List[str]]:
    """
    Update the plan based on recent observations and actions from the trajectory.
    
    Args:
        plan: The current plan to update
        query: The task query/intent for retrieving similar tasks from memory
        memory_steps_text: Formatted string of retrieved memories
        file_id_list: List of file IDs for the retrieved memories
        action_history: List of action descriptions (strings) from meta_data
        trajectory: The current trajectory containing observations and actions
        tool_llm: VLM instance for adaptive focus check (only set if --use_adaptive_memory flag)
        history_context: Dict from extract_history_context() (only set if --use_history_context flag)
        memory: ExperienceMemorySimple instance for re-retrieval (only needed if --use_adaptive_memory)
        server_url: vLLM server URL (e.g., http://localhost:8000/v1)
        model: Model name served by vLLM
        planner_model: Planner model instance
        similar_num: Number of similar tasks to retrieve (default: 5)
        temperature: Temperature for generation (default: 0.7)
        max_tokens: Maximum tokens for generation (default: 200)
        previous_fail_reasons: Optional previous failure reasons for this task
        focus_check_threshold: Number of steps before checking memory focus (default: 3)
    
    Returns:
        Tuple of (updated_plan, updated_memory_steps_text)
    """
    # Extract observations and screenshots from trajectory
    observations = [traj['observation'] for traj in trajectory if 'observation' in traj]
    current_screenshot = observations[-1]['image']
    
    # Extract recent screenshots for adaptive focus check (last 5)
    recent_screenshots = [obs['image'] for obs in observations[-5:] if isinstance(obs, dict) and 'image' in obs]
    
    # Adaptive memory focus: check if memory query should be updated based on current state
    # Note: tool_llm is only passed when --use_adaptive_memory flag is enabled
    if tool_llm is not None and len(action_history) >= focus_check_threshold:
        logger.info(f"[PlanUpdate] Checking if memory focus needs updating (step {len(action_history)})...")
        needs_update, new_focus = check_and_update_memory_focus(
            original_intent=query,
            recent_screenshots=recent_screenshots,
            recent_actions=action_history,
            current_memories=memory_steps_text,
            vlm=tool_llm,
        )
        
        if needs_update and new_focus:
            adaptive_query = f"{query} {new_focus}"
            logger.info(f"[PlanUpdate] Memory focus updated! New focus: {new_focus}")
            logger.info(f"[PlanUpdate] Re-retrieving with adaptive query: {adaptive_query}")
            
            # Re-retrieve with adaptive query if memory object is available
            if memory is not None:
                similar_tasks = memory.retrieve_similar_tasks(adaptive_query, similar_num=similar_num)
                if similar_tasks:
                    memory_steps_text = _format_memory_steps(similar_tasks)
                    logger.info(f"[PlanUpdate] Re-retrieved {len(similar_tasks)} similar tasks with new focus")
                    file_id_list = [task['task_id'] for task in similar_tasks]
                else:
                    logger.info("[PlanUpdate] No tasks found with adaptive query, keeping original memories")
            else:
                logger.warning("[PlanUpdate] Memory object not provided, cannot re-retrieve")
        else:
            logger.info("[PlanUpdate] Memory focus unchanged, keeping original memories")
    else:
        if tool_llm is None:
            logger.info("[PlanUpdate] Adaptive memory disabled (--use_adaptive_memory not set)")
        else:
            logger.info(f"[PlanUpdate] Step {len(action_history)} < threshold {focus_check_threshold}, skipping focus check")
    
    # Extract recent observations for plan update prompt
    # Get previous observations (excluding the most recent one)
    previous_observations = observations[:-1] if len(observations) > 1 else []
    # Extract images from previous observations (last 5)
    recent_obs_list = previous_observations[-5:] if len(previous_observations) > 5 else previous_observations
    recent_observations = [obs['image'] for obs in recent_obs_list if isinstance(obs, dict) and 'image' in obs]
    
    recent_actions = action_history[-5:] if len(action_history) > 5 else action_history
    
    # Build history section only if history_context is provided (--use_history_context flag)
    history_section = ""
    if history_context:
        history_section = f"""
Action History Analysis:
- Progress: {history_context.get('progress_summary', 'No progress info')}
- Recent successes:
{history_context.get('completed_text', 'None yet')}

Failed Actions (DO NOT repeat these):
{history_context.get('failed_text', 'None')}
"""
        if history_context.get('stuck_pattern'):
            history_section += f"\nWARNING: {history_context['stuck_pattern']} - Try a different approach!\n"
        
        if history_context.get('avoid_list'):
            avoid_actions = ", ".join(history_context['avoid_list'][:3])  # Show top 3
            history_section += f"\nActions to AVOID (failed multiple times): {avoid_actions}\n"
    
    # Build the update prompt
    update_prompt = f"""You are a task planning assistant. Update the original plan based on recent observations and actions, and decide what is the action.

Experience Memory For Reference:
{memory_steps_text}

Previous attempt failure reasons:
{previous_fail_reasons}
{history_section}
Original Plan:
{plan}

Based on the recent actions and screenshot observations and current state, update the plan to:
1. Reflect what has been accomplished so far
2. Adjust remaining steps based on current progress
3. Keep it concise (3-5 steps total)
4. Mark completed steps as done
5. Do not include actions like Open a web browser, click the search bar, click the search button. Because they will be conducted automatically.
6. Clearly state what is the next single step! If the task has been finished, direct yiled STOP and your answer for the next step!
7. IMPORTANT: Do NOT repeat actions that have already failed (see Action History Analysis above if available)

Provide the updated plan and next step decision in this format:
<plan>Your updated plan here</plan>
<next action> next single step <next action>

Updated Plan:"""

    # Build messages with current screenshot
    user_content = [{"type": "text", "text": "The following are recent observations and actions executed by the action agent:"}]
    for i, (action, img) in enumerate(zip(recent_actions, recent_observations), 1):
        # Add corresponding image
        if isinstance(img, str):
            if img.startswith("data:image"):
                image_url = img
            else:
                image_url = f"data:image/png;base64,{img}"
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        # Add action description
        user_content.append({"type": "text", "text": f"Action {i}: {action}"})
    
    # Format screenshot for vLLM
    if isinstance(current_screenshot, str):
        if current_screenshot.startswith("data:image"):
            image_url = current_screenshot
        else:
            image_url = f"data:image/png;base64,{current_screenshot}"
        user_content.append({"type": "text", "text": "This is the current observation:"})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })
    
    user_content.append({"type": "text", "text": update_prompt})
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that updates task plans based on recent progress and observations. You can see the current screen state to make informed plan updates."
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    
    # Generate updated plan using the planner model
    logger.info("[PlanUpdate] Updating plan based on recent observations...")
    if planner_model:
        if use_continuous_memory:
            updated_plan, _, _ = planner_model.chat(messages=messages, stream=False, file_id_list=file_id_list)
        else:
            updated_plan, _, _ = planner_model.chat(messages=messages, stream=False)
        updated_plan = updated_plan.content if hasattr(updated_plan, 'content') else str(updated_plan)
    else:
        updated_plan = _vllm_chat_completion(
            server_url=server_url,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    
    logger.info(f"[PlanUpdate] Updated plan:\n{updated_plan.strip()}")
    return updated_plan.strip(), memory_steps_text, file_id_list


def main():
    """Example usage of plan_with_memory."""
    # Configuration (override via DISCRETE_SUMMARY_PATH and FAISS_INDEX_PATH env vars)
    summary_json_path = os.environ.get("DISCRETE_SUMMARY_PATH", "discrete_summary.json")
    faiss_index_path = os.environ.get("FAISS_INDEX_PATH", "memory_index/simple_text_152")
    
    # Initialize memory
    logger.info("Loading memory index...")
    memory = ExperienceMemorySimple(summary_json_path, faiss_index_path)
    
    # Example query
    query = "Tell me one bus stop that is nearest to the intersection of main street and Amherst street in Altavista."
    
    # Configuration for planner model
    server_url = "http://localhost:8000/v1"  # Update with your vLLM server URL
    model = "Qwen/Qwen2.5-VL-7B-Instruct"  # Update with your model name
    
    # Example screenshot (replace with actual base64 encoded screenshot)
    screenshot = ""  # TODO: Provide actual base64 encoded screenshot
    
    # Generate plan (returns plan text, memory_steps_text, file_id_list)
    plan, memory_steps_text, file_id_list = generate_plan_with_memory(
        query=query,
        memory=memory,
        server_url=server_url,
        model=model,
        planner_model=None,
        screenshot=screenshot,
        similar_num=10,
        temperature=0.7,
        max_tokens=200,
    )
    
    logger.info("="*80)
    logger.info("GENERATED PLAN:")
    logger.info("="*80)
    logger.info(plan)
    logger.info("="*80)


if __name__ == "__main__":
    main()