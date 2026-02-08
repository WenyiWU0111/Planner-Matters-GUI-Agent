"""Function Call Agent for GUI Agent with ReAct paradigm."""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import base64

from browser_env import Trajectory, Action
from browser_env.actions import ActionTypes
from actions import (
    create_click_action,
    create_selection_action,
    create_type_action,
    create_scroll_action,
    create_wait_action,
    create_stop_action,
    create_none_action,
    create_key_press_action,
    create_goto_url_action,
    create_go_back_action,
    parse_action_json,
)
from .llm_config import create_model, load_tool_llm
from tools.gui_tools import (
    ClickTool,
    TypeTool,
    ScrollTool,
    WaitTool,
    StopTool,
    PressKeyTool,
    PageGotoTool,
    SelectionTool,
    GoBackTool,
)
from tools.analysis_tools import MapSearchTool, ContentAnalyzerTool

from memory.reasoning_bank import ReasoningBank

NAME_TO_CLS = {
    "click": ClickTool,
    "selection": SelectionTool,
    "type": TypeTool,
    "scroll": ScrollTool,
    "wait": WaitTool,
    "stop": StopTool,
    "go_back": GoBackTool,
    "press_key": PressKeyTool,
    "goto_url": PageGotoTool,
    "map_search": MapSearchTool,
    "content_analyzer": ContentAnalyzerTool
}


class FunctionCallAgent:
    """Custom function call agent for GUI interactions using ReAct paradigm with direct model calls"""
    
    def __init__(self, args: argparse.Namespace, **kwargs):
        """Initialize the function call agent"""
        # Store args first so it's available to other initialization methods
        self.args = args
        
        # Initialize model
        self.llm = create_model(args)
        # Initialize tool LLM for tools that need it
        self.tool_llm = load_tool_llm(args)
        
        self.logger = logging.getLogger("logger")
        self.current_step = 0
        self.discrete_memory_cache: Dict[str, str] = {}
        self.function_map = self._build_function_map()
        self.tool_specs = self._build_tool_specs(self._define_functions())

        training_data_dir = getattr(args, "training_data_dir", "training_data")
        memory_data_dirs = getattr(args, 'memory_data_dir', ['training_data'])
        # Ensure memory_data_dirs is a list
        if isinstance(memory_data_dirs, str):
            memory_data_dirs = [memory_data_dirs]
        self.memory_data_dirs = memory_data_dirs
        # Initialize training data collector if enabled
        if hasattr(args, 'collect_training_data') and args.collect_training_data:
            from utils.training_data_collector import TrainingDataCollector, get_collector, set_collector
            from utils.llm_wrapper import wrap_llm
            self.training_collector = TrainingDataCollector(
                output_dir=training_data_dir,
                enabled=True
            )
            set_collector(self.training_collector)
            wrapped_llm = wrap_llm(self.llm)
            self.llm = wrapped_llm
        else:
            self.training_collector = None
        
        # Initialize discrete memory system if enabled
        if (hasattr(args, 'use_discrete_memory') and args.use_discrete_memory) or (hasattr(args, 'use_continuous_memory') and args.use_continuous_memory) or (hasattr(args, 'use_verifier') and args.use_verifier):
            from memory.experience_memory import ExperienceMemory
            # Determine if multimodal memory should be used
            multimodal = True
            # Check if there's a saved index path
            faiss_index_path = getattr(args, 'faiss_index_path', None)
            print(f"Initializing Discrete Memory system (multimodal: {multimodal})...")
            self.memory = ExperienceMemory(training_data_path=self.memory_data_dirs[0], multimodal=multimodal, faiss_index_path=faiss_index_path, agent=self, bank_size=args.bank_size)
            print("Discrete Memory system initialized successfully")
            self.experience_memory = None
            self.experience_texts, self.experience_images, self.file_id_list = None, None, None
        else:
            self.memory = None
            self.experience_memory = None
            self.experience_texts = None
            self.experience_images = None
            self.file_id_list = None

        # Initialize reasoning bank (optional)
        self.reasoning_bank = None
        if hasattr(self.args, 'use_reasoning_bank') and self.args.use_reasoning_bank:
            try:
                use_mm = getattr(self.args, 'reasoning_bank_multimodal', False)
                bank_path = getattr(self.args, 'reasoning_bank_path', 'memory/reasoning_bank.jsonl')
                index_base = getattr(self.args, 'reasoning_index_base', 'memory_index/reasoning_bank_text')
                
                # Use multimodal paths if enabled
                if use_mm:
                    bank_path = getattr(self.args, 'reasoning_bank_path', 'memory/reasoning_bank_mm.jsonl')
                    index_base = getattr(self.args, 'reasoning_index_base', 'memory_index/reasoning_bank_mm')
                
                self.reasoning_bank = ReasoningBank(
                    bank_path=bank_path,
                    index_base_path=index_base,
                    use_multimodal=use_mm
                )
            except Exception as e:
                self.reasoning_bank = None
        
        # Store analysis results and map search context for next steps
        self.last_analysis_result = None
        self.last_map_search_query = None
        self.last_map_search_result = None
        self.last_page_goto_name = None
        self.last_page_goto_result = None
        
        # Dynamic memory update state (for use_dynamic_memory_update feature)
        self.current_raw_takeaways = None  # List[TaggedTrajectory] - for checkpoint comparison
        self.memory_update_count = 0        # int - track how many updates occurred
        self.clean_intent = None            # str - original intent without instructions/errors for retrieval
        
        if getattr(self.args, 'use_dynamic_memory_update', False):
            self.logger.info("[DynamicMemory] Dynamic memory update ENABLED")

    def _define_functions(self) -> List[str]:
        """Function names exposed to the agent."""
        return [
            "click",
            "type",
            "press_key",
            "scroll",
            "wait",
            "go_back",
            "stop",
            "map_search",
            "content_analyzer",
            "goto_url",
        ]

    def _build_tool_specs(self, function_list: List[str]) -> List[Dict[str, Any]]:
        """Build tool specs for prompt (no mutation of tool.parameters)."""
        specs: List[Dict[str, Any]] = []
        for name in function_list:
            cls = NAME_TO_CLS.get(name)
            if cls is None:
                self.logger.warning("Tool %s not in NAME_TO_CLS", name)
                continue
            try:
                tool = cls()
            except Exception as e:
                self.logger.warning("Failed to instantiate tool %s: %s", name, e)
                continue
            raw_params = getattr(tool, "parameters", None)
            params = None
            if raw_params is not None:
                properties = dict(raw_params.get("properties", {}))
                params = {**raw_params, "properties": properties}
            specs.append({
                "name": getattr(tool, "name", name),
                "description": getattr(tool, "description", "No description"),
                "parameters": params,
            })
        return specs

    def _build_function_map(self) -> Dict[str, Any]:
        """Build name -> tool instance map."""
        function_map: Dict[str, Any] = {}
        for name, cls in NAME_TO_CLS.items():
            try:
                tool = cls()
                tool.llm = self.tool_llm
                function_map[name] = tool
            except Exception as e:
                self.logger.warning("Failed to initialize tool %s: %s", name, e)
        return function_map
    
    def _get_system_message(self, intent, trajectory, reflection=None, status_note=None) -> str:
        """Get the system message for the agent using ReAct paradigm"""
        tools_section = ""
        lines = []
        for spec in self.tool_specs:
            desc = spec.get('description') or ''
            name = spec.get('name')
            params = spec.get('parameters', {})
            
            # Build parameter description
            param_desc = ""
            if params and 'properties' in params:
                param_list = []
                for param_name, param_info in params['properties'].items():
                    param_type = param_info.get('type', 'string')
                    param_desc_text = param_info.get('description', '')
                    if 'enum' in param_info:
                        enum_values = ', '.join(param_info['enum'])
                        param_list.append(f"`{param_name}` ({param_type}: {enum_values})")
                    else:
                        param_list.append(f"`{param_name}` ({param_type}): {param_desc_text}")
                param_desc = f" - Parameters: {', '.join(param_list)}"
            
            lines.append(f"- **{name}**: {desc}{param_desc}")
        tools_section = "\n".join(lines)
        # Resolve prompt file paths relative to this file's location
        agent_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Build discrete memory (only once per task - cached in self.experience_memory)
        if (getattr(self.args, "use_discrete_memory", False) or getattr(self.args, "use_continuous_memory", False)) or getattr(self.args, "use_verifier", False) and self.memory is not None and self.experience_memory is None:
            first_image = self._get_first_screenshot(trajectory)
            print(f'[Discrete Memory] Retrieving similar trajectories with similar_num: {self.args.similar_num}')
            _, self.experience_texts, self.experience_images, self.file_id_list = self.memory.construct_experience_memory(
                intent, self, current_image=first_image,
                dataset=self.args.evaluation_type, domain=self.args.domain,
                similar_num=self.args.similar_num
            )
            self.experience_memory = self._build_discrete_memory_block(
                intent=intent,
                file_id_list=self.file_id_list,
                experience_actions=self.experience_texts,
                experience_images=self.experience_images,
                current_image=first_image,  # Pass current screenshot for digestion
            )
        elif self.experience_memory is None:
            examples_path = os.path.join(agent_dir, "prompts", "examples.txt")
            if os.path.exists(examples_path):
                with open(examples_path, "r", encoding="utf-8") as f:
                    self.experience_memory = f.read()
            else:
                self.experience_memory = ""

        if getattr(self.args, "use_awm", False):
            self.logger.info("[AWM] Using AWM for experience memory")
            workflow_prompt_path = f"workflow_memory/{self.args.domain}.txt"
            if os.path.exists(workflow_prompt_path):
                with open(workflow_prompt_path, "r", encoding="utf-8") as f:
                    self.experience_memory = f.read()
            else:
                self.experience_memory = ""
                Path(workflow_prompt_path).parent.mkdir(parents=True, exist_ok=True)
                with open(workflow_prompt_path, "w", encoding="utf-8") as f:
                    f.write("")

        system_prompt_path = os.path.join(agent_dir, "prompts", "system_prompt.txt")
        if os.path.exists(system_prompt_path):
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        else:
            # Fallback: basic system prompt if file doesn't exist
            system_prompt = """You are a helpful GUI automation agent. Use the available tools to complete tasks.
            
Available tools:
{tools_section}

{experience_memory}
"""
        system_prompt = system_prompt.format(experience_memory=self.experience_memory, tools_section=tools_section)
        
        # Inject reflection/status into system prompt if available
        if reflection or status_note:
             system_prompt += "\n\n*** SELF-REFLECTION & CORRECTION ***"
             if reflection:
                 system_prompt += f"\nThe following is an analysis of your recent performance. You must prioritize this feedback over your previous plan.\n\n{reflection}"
             # Only add hard constraint and system check when stuck is indicated (status_note present)
             if status_note:
                 system_prompt += "\n\nCONSTRAINT: If status is 'Stuck' or 'Regressing', you MUST change strategy. DO NOT REPEAT THE PREVIOUS ACTION."
                 system_prompt += f"\n\nSystem check: {status_note}"
                 
        return system_prompt

    def _load_discrete_memory_cache(self) -> Dict[str, str]:
        cache_path = getattr(self.args, "discrete_memory_cache_path", None)
        if cache_path is None:
            raise ValueError("discrete_memory_cache_path must not be None when use_discrete_memory is enabled")
        cache_file = Path(cache_path)
        if not cache_file.exists():
            return {}
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid discrete memory cache format in {cache_path}: expected JSON object")
        summaries = payload.get("summaries")
        if summaries is None:
            raise ValueError(f"Invalid discrete memory cache format in {cache_path}: missing 'summaries'")
        if not isinstance(summaries, dict):
            raise ValueError(f"Invalid discrete memory cache format in {cache_path}: 'summaries' must be an object")
        for k, v in summaries.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError(f"Invalid discrete memory cache entry in {cache_path}: keys/values must be strings")
        return summaries

    def _save_discrete_memory_cache(self, summaries: Dict[str, str]) -> None:
        cache_path = getattr(self.args, "discrete_memory_cache_path", None)
        if cache_path is None:
            raise ValueError("discrete_memory_cache_path must not be None when use_discrete_memory is enabled")
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summaries": summaries}
        cache_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _format_actions_for_summary(self, actions: List[Dict], max_actions: int) -> str:
        if not isinstance(actions, list):
            raise ValueError(f"actions must be a list, got {type(actions)}")
        lines: List[str] = []
        for a in actions[:max_actions]:
            if not isinstance(a, dict):
                raise ValueError(f"action must be a dict, got {type(a)}")
            name = a.get("name")
            args = a.get("arguments")
            if not isinstance(name, str):
                raise ValueError("action['name'] must be a string")
            if not isinstance(args, dict):
                raise ValueError("action['arguments'] must be a dict")
            reasoning = args.get("reasoning", "")
            if not isinstance(reasoning, str):
                raise ValueError("action['arguments']['reasoning'] must be a string")
            lines.append(f"- {name}: {reasoning}")
        return "\n".join(lines)

    def _summarize_trajectory_with_vlm(
        self,
        task: str,
        actions_text: str,
        image_b64: Optional[str],
        experience_texts: Optional[List[List[Dict]]] = None,
        experience_images: Optional[List[List[str]]] = None,
        file_id_list: Optional[List[str]] = None,
    ) -> str:
        discrete_llm = self.tool_llm
        if discrete_llm is None:
            raise ValueError("LLM is required for trajectory summary generation")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        if not isinstance(actions_text, str) or not actions_text.strip():
            raise ValueError("actions_text must be a non-empty string")

        system = (
            "You extract actionable heuristics from SUCCESSFUL GUI agent trajectories.\n"
            "Return EXACTLY 1 sentence starting with 'takeaway:' in this format:\n"
            "takeaway: <ONE concise actionable heuristic>\n"
            "Constraints:\n"
            "- Focus on WHAT strategy worked (not what the agent did).\n"
            "- Must start with 'takeaway:' (exact substring).\n"
            "- Keep it under 20 words.\n"
            "- No quotes, no bullet points, no step-by-step narration, no coordinates/IDs."
        )
        user_text = f"Task: {task}\nActions:\n{actions_text}"
        if image_b64 is None:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ]
        else:
            if not isinstance(image_b64, str) or not image_b64.startswith("data:image"):
                raise ValueError("image_b64 must be a data:image base64 URL when provided")
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_b64}},
                ]},
            ]

        resp, _, _ = discrete_llm.chat(messages=messages, stream=False, temperature=0.0, max_tokens=128)
        if not hasattr(resp, "content"):
            raise ValueError("LLM response missing content")
        summary = resp.content.strip()
        if not summary:
            raise ValueError("Empty summary returned by LLM")
        if "\n" in summary:
            raise ValueError(f"Summary must be a single line, got: {summary!r}")
        if summary.lstrip().startswith(("-", "•", "\"")):
            raise ValueError(f"Summary must not start with bullets/quotes, got: {summary!r}")
        if not summary.lower().startswith("takeaway:"):
            raise ValueError(f"Summary must start with 'takeaway:', got: {summary!r}")
        self.logger.info(f"[Trajectory Summary] ({len(summary.split())} words): {summary}")
        return summary

    def _digest_discrete_memory(
        self,
        current_task: str,
        current_image: Optional[str],
        trajectory_summaries: List[str],
        experience_texts: Optional[List[List[Dict]]] = None,
        experience_images: Optional[List[List[str]]] = None,
        file_id_list: Optional[List[str]] = None,
    ) -> str:
        """
        Digest multiple trajectory summaries into a single, task-specific guidance.
        
        Instead of injecting all summaries directly, we ask the VLM to analyze them
        in the context of the current task and screenshot, producing focused guidance.
        """
        discrete_llm = self.tool_llm
        if discrete_llm is None:
            raise ValueError("LLM is required for discrete memory digestion")
        if not trajectory_summaries:
            raise ValueError("trajectory_summaries cannot be empty")
        
        summaries_text = "\n".join(f"- {s}" for s in trajectory_summaries)
        
        system = (
            "You are an expert at analyzing past GUI agent experiences to help with a new task.\n"
            "Given the current task, current screenshot, and retrieved experience summaries,\n"
            "synthesize them into focused, actionable guidance.\n\n"
            "Output format: ONE concise paragraph (2-3 sentences) that answers:\n"
            "1. Which strategies from past experiences are MOST relevant to this specific task?\n"
            "2. What key actions or filters should be prioritized?\n\n"
            "IMPORTANT RULES:\n"
            "- Focus ONLY on navigation/search strategies, NOT on when to stop.\n"
            "- Do NOT mention stopping, completing, or finishing the task.\n"
            "- Do NOT give instructions about providing answers or explanations.\n"
            "- Be specific to the current task. Do NOT just repeat the summaries.\n"
            "- Do NOT use bullet points. Write as a coherent paragraph."
        )
        
        user_text = f"Current Task: {current_task}\n\nRetrieved Experience Summaries:\n{summaries_text}"
        
        if current_image is None:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ]
        else:
            if not isinstance(current_image, str):
                raise ValueError("current_image must be a string when provided")
            # Handle both raw base64 and data URL formats
            if current_image.startswith("data:image"):
                image_url = current_image
            else:
                # Assume raw base64, add PNG data URL prefix
                image_url = f"data:image/png;base64,{current_image}"
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ]
        
        self.logger.info("[Discrete Memory] Digesting summaries into task-specific guidance...")
        resp, _, _ = discrete_llm.chat(messages=messages, stream=False, temperature=0.0, max_tokens=256)
        if not hasattr(resp, "content"):
            raise ValueError("LLM response missing content")
        
        guidance = resp.content.strip()
        if not guidance:
            raise ValueError("Empty guidance returned by VLM")
        
        self.logger.info(f"[Discrete Memory] Digested guidance ({len(guidance.split())} words): {guidance}")
        return guidance

    def _build_discrete_memory_block(
        self,
        intent: str,
        file_id_list: List[str],
        experience_actions: List[List[Dict]],
        experience_images: List[List[str]],
        current_image: Optional[str] = None,
    ) -> str:
        """
        Build discrete memory block with two-stage processing:
        1. Summarize each retrieved trajectory (cached)
        2. Digest all summaries into task-specific guidance using current task + image
        """
        self.logger.info(f"[Discrete Memory] Building summaries for {len(file_id_list)} trajectories...")
        if self.memory is None:
            raise ValueError("memory must be initialized to build discrete memory")
        if not isinstance(file_id_list, list) or not isinstance(experience_actions, list) or not isinstance(experience_images, list):
            raise ValueError("file_id_list/experience_actions/experience_images must be lists")
        if not (len(file_id_list) == len(experience_actions) == len(experience_images)):
            raise ValueError("file_id_list, experience_actions, and experience_images must have the same length")

        # Load cache once per run (lazy)
        if not self.discrete_memory_cache:
            self.discrete_memory_cache = self._load_discrete_memory_cache()

        max_actions = int(getattr(self.args, "discrete_memory_max_actions", 8))
        selected_files = getattr(self.memory, "selected_conversations", None)
        if not isinstance(selected_files, list):
            raise ValueError("memory.selected_conversations must be a list of file paths")

        # Stage 1: Collect all trajectory summaries
        summaries: List[str] = []
        updated = False
        for file_id, actions, imgs in zip(file_id_list, experience_actions, experience_images):
            if not isinstance(file_id, str) or not file_id:
                raise ValueError("file_id must be a non-empty string")
            cached = self.discrete_memory_cache.get(file_id)
            if cached is None:
                matches = [fp for fp in selected_files if isinstance(fp, str) and fp.endswith(f"/{file_id}.jsonl")]
                if len(matches) != 1:
                    raise ValueError(f"Could not uniquely map file_id={file_id!r} to a selected conversation file")
                memory_file = json.loads(Path(matches[0]).read_text())
                task = memory_file.get("task_description")
                if not isinstance(task, str) or not task.strip():
                    raise ValueError(f"Missing/invalid task_description in {matches[0]}")
                actions_text = self._format_actions_for_summary(actions, max_actions=max_actions)
                first_img = imgs[0] if (isinstance(imgs, list) and len(imgs) > 0) else None
                summary = self._summarize_trajectory_with_vlm(
                    task=task,
                    actions_text=actions_text,
                    image_b64=first_img,
                    experience_texts=experience_actions,
                    experience_images=experience_images,
                    file_id_list=file_id_list,
                )
                self.discrete_memory_cache[file_id] = summary
                updated = True
            else:
                summary = cached
                self.logger.info(f"[Discrete Memory] (cached) {file_id}: {summary}")
            summaries.append(summary)
        self.discrete_memory_summaries = summaries
        if updated:
            self._save_discrete_memory_cache(self.discrete_memory_cache)

        # Stage 2: Digest summaries into task-specific guidance
        digested_guidance = self._digest_discrete_memory(
            current_task=intent,
            current_image=current_image,
            trajectory_summaries=summaries,
            experience_texts=experience_actions,
            experience_images=experience_images,
            file_id_list=file_id_list,
        )

        return f"[Experience Guidance]\n{digested_guidance}"

    def next_action_custom(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: Dict[str, Any],
    ):
        """Generate the next action using function calling with ReAct paradigm"""
        
        self.current_step += 1
        print('*'*50, 'current step: ', self.current_step, '*'*50)
        
        # Step 3: Prepare messages
        messages, meta_data = self._prepare_messages(
            trajectory, 
            intent, 
            meta_data, 
        )
        
        # Continuous memory (Q-Former) should use whatever experience has been populated
        if self.args.use_continuous_memory and (self.experience_texts is not None or self.experience_images is not None):
            responses, original_inputs, original_outputs = self.llm.chat(messages=messages, stream=False, 
                                        experience_texts=self.experience_texts, experience_images=self.experience_images,
                                        file_id_list=self.file_id_list)
        else:
            # Call the LLM with function calling
            responses, original_inputs, original_outputs = self.llm.chat(messages=messages, stream=False)
        meta_data['original_inputs'] = original_inputs
        meta_data['original_outputs'] = original_outputs
        meta_data['original_responses'] = responses
        if self.training_collector:
            self.llm.save_conversation(messages, responses.content)
        meta_data["response_history"].append(responses.content)
                
        # Extract page if available in meta_data or elsewhere; pass explicitly
        page_for_tools = meta_data.get('page')
            
        action = self._process_response(responses.content, trajectory, page_for_tools, intent, meta_data)
        return action, meta_data
    
    def _get_current_screenshot(self, trajectory: Trajectory) -> Optional[str]:
        """Extract the current screenshot from the trajectory for multimodal memory retrieval."""
        recent_obs = trajectory[-1]
        if isinstance(recent_obs, dict) and 'observation' in recent_obs:
            obs = recent_obs['observation']
            if 'image' in obs:
                return obs['image']
        return None
    
    def _get_first_screenshot(self, trajectory: Trajectory) -> Optional[str]:
        """Extract the first screenshot from the trajectory."""
        if not trajectory or not isinstance(trajectory[0], dict):
            return None
        obs = trajectory[0].get("observation") if isinstance(trajectory[0], dict) else {}
        return obs.get("image") if isinstance(obs, dict) else None
    
    def _detect_repetition_and_no_progress(self, trajectory: Trajectory, meta_data: Dict[str, Any]) -> Optional[str]:
        """Detect if the last action made no progress (same URL and screenshot) or is repeating."""
        try:
            if not trajectory or len(trajectory) < 2:
                return None
            last_state = trajectory[-1]
            prev_state = trajectory[-2]
            # URL comparison (from info.page.url if available)
            last_url = None
            prev_url = None
            try:
                if isinstance(last_state, dict) and 'info' in last_state and 'page' in last_state['info']:
                    last_url = getattr(last_state['info']['page'], 'url', None)
                if isinstance(prev_state, dict) and 'info' in prev_state and 'page' in prev_state['info']:
                    prev_url = getattr(prev_state['info']['page'], 'url', None)
            except Exception:
                pass
            # Image comparison (base64 exact match)
            last_img = last_state.get('observation', {}).get('image') if isinstance(last_state, dict) else None
            prev_img = prev_state.get('observation', {}).get('image') if isinstance(prev_state, dict) else None
            url_same = (last_url is not None and prev_url is not None and last_url == prev_url)
            img_same = (last_img is not None and prev_img is not None and last_img == prev_img)
            
            # Log detection details
            self.logger.debug(f"[StuckDetection] URL same: {url_same}, Image same: {img_same}")
            if last_url and prev_url:
                self.logger.debug(f"[StuckDetection] URLs: {prev_url[:80]} -> {last_url[:80]}")
            
            # Repetition detection from textual action history
            repeated_count = 0
            last_action_text = None
            if meta_data and 'action_history' in meta_data:
                ah = meta_data['action_history']
                if isinstance(ah, list) and len(ah) >= 2:
                    last_action_text = ah[-1]
                    # Count consecutive same action strings from the end
                    ref = ah[-1]
                    for s in reversed(ah):
                        if s == ref:
                            repeated_count += 1
                        else:
                            break
                    if repeated_count >= 2:
                        self.logger.debug(f"[StuckDetection] Action repeated {repeated_count} times: {last_action_text[:100]}")
            
            # If no progress or repeated actions, craft feedback
            if (url_same and img_same) or repeated_count >= 2:
                parts = []
                if url_same and img_same:
                    parts.append("The last action did not change the page (URL and screenshot unchanged).")
                if repeated_count >= 2 and last_action_text:
                    parts.append(f"The action was repeated {repeated_count} times: {last_action_text}")
                if meta_data and 'error_message' in meta_data and meta_data['error_message']:
                    parts.append(f"Environment feedback: {meta_data['error_message']}")
                parts.append("Do NOT repeat the same action/target. Try a different strategy (e.g., go_back to the previous page, click a different element, scroll, type into inputs, press Enter, or use content_analyzer).")
                return " ".join(parts)
        except Exception as e:
            self.logger.error(f"[StuckDetection] Error in detection: {e}")
            return None
        return None
    
    def _generate_action_history_summary(self, intent: str, action_history: List[str], trajectory: Trajectory) -> Optional[str]:
        """Generate a summary and reflection on the last 5 actions using tool LLM."""
        if not action_history or not self.tool_llm or not trajectory:
            return None
        recent_obs = [i['observation']['image'] for i in trajectory if 'observation' in i][-5:]
        recent_actions = action_history[-5:]
        # IMPROVED PROMPT: Focus on Quality and Progress
        prompt = f"""You are a strict coach speaking DIRECTLY to a GUI agent.
        
        USER INTENT: {intent}
        
        Analyze YOUR recent actions (Screenshots + Actions).
        
        Evaluate YOUR WORK QUALITY:
        1. **Effectiveness**: Did your actions actually change the state or move closer to the goal?
        2. **Errors**: Are you hallucinating elements or getting stuck?
        3. **Guidance**: If your work quality is low, what SPECIFIC distinct action should you take next?
        
        CRITICAL FORMATTING RULE:
        - You MUST use "You" or "Your".
        - NEVER use "The agent".
        
        Response Format:
        "Status: [Progressing / Stuck / Regressing]
        Critique: [Your evaluation, e.g., 'You have clicked...']
        Correction: [Specific instruction, e.g., 'You must go back...']"
        """
        messages = [{'role': 'system', 'content': prompt}]
        for obs, action in zip(recent_obs, recent_actions):
            messages.append(
                {'role': 'user', 'content': [
                    {"type": "text", "text": "The following is the task for the agent to accomplish: " + intent},
                    {"type": "text", "text": "The following is the screenshot: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{obs}"}},
                    {"type": "text", "text": "The following is the corresponding action: " + action}
                ]
            })
        response, _, _ = self.tool_llm.chat(messages=messages, stream=False)
        return response.content
    
    def _prepare_messages(self, trajectory: Trajectory, intent: str, meta_data: Dict[str, Any]) -> Tuple[List[Dict], Dict[str, Any]]:
        """Prepare messages for the LLM with ReAct context
        
        Args:
            trajectory: Current trajectory
            intent: Task intent
            meta_data: Metadata dictionary
            verifier_feedback: Optional verifier feedback from action verification
            trajectory_verifier_feedback: Optional verifier feedback from trajectory verification
            
        Returns:
            Tuple of (messages, meta_data)
        """
        messages = []
        
        # Extract and store clean_intent for retrieval (avoid instruction/error pollution)
        if self.clean_intent is None and meta_data and 'clean_intent' in meta_data:
            self.clean_intent = meta_data['clean_intent']
            self.logger.info(f"[DynamicMemory] Stored clean intent: {self.clean_intent[:80]}...")
        
        # Generate reflection early to inject into system prompt
        history_summary = None
        if meta_data and 'action_history' in meta_data and self.args.use_history:
            self.logger.info("[Reflexion] use_history=True, generating action history reflection...")
            action_history = meta_data['action_history']
            history_summary = self._generate_action_history_summary(intent, action_history, trajectory)
            if history_summary:
                meta_data['step_history_reflection'] = history_summary
                self.logger.info(f"[Reflexion] Generated reflection:\n{history_summary}")
            else:
                self.logger.info("[Reflexion] No reflection generated (insufficient history)")

        # Early stuck detection and flag for system prompt injection
        is_stuck = False
        status_note = None
        early_feedback = None
        try:
            early_feedback = self._detect_repetition_and_no_progress(trajectory, meta_data)
            if early_feedback:
                status_note = early_feedback
                is_stuck = True
        except Exception:
            pass

        # Also consider the reflection content for stuck indications
        try:
            if (not is_stuck) and history_summary and any(k in history_summary.lower() for k in ['status: stuck', 'status: regressing', 'stuck', 'regressing']):
                is_stuck = True
        except Exception:
            pass

        # Add system message; always include reflection if available, but only include stuck status/constraint when stuck
        messages.append({
            'role': 'system',
            'content': self._get_system_message(
                intent,
                trajectory,
                reflection=history_summary,
                status_note=status_note if is_stuck else None,
            )
        })
        
        # Add current intent with ReAct prompt
        current_task = intent
        
        # Add analysis results context if available
        if self.last_analysis_result:
            analysis_summary = self.last_analysis_result
            messages.append({
                'role': 'user',
                'content': f"**Content analysis results:** {analysis_summary}"
            })
            # Clear the analysis result after using it
            self.last_analysis_result = None
        
        # Add web search results context if available
        if self.last_web_search_result:
            web_search_summary = self.last_web_search_result
            # Create content with text and screenshots if available
            if self.last_web_search_screenshots:                
                content_items = [
                    {"type": "text", "text": f"**Web search results:** {web_search_summary}"}
                ]
                # Add screenshot images and collect paths for cleanup
                screenshot_files_to_delete = []
                for screenshot_path in self.last_web_search_screenshots:
                    if os.path.exists(screenshot_path):
                        with open(screenshot_path, "rb") as img_file:
                            img_data = img_file.read()
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                            content_items.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            })
                        screenshot_files_to_delete.append(screenshot_path)
                
                messages.append({
                    'role': 'user',
                    'content': content_items
                })
                
                # Delete screenshot files after adding them to messages
                for screenshot_path in screenshot_files_to_delete:
                    try:
                        os.remove(screenshot_path)
                    except Exception as e:
                        continue
            else:
                messages.append({
                    'role': 'user',
                    'content': f"**Web search results:** {web_search_summary}"
                })
            # Clear the web search result after using it
            self.last_web_search_result = None
            self.last_web_search_screenshots = None
        
        # Add page goto results context if available
        if self.last_page_goto_result:
            page_goto_summary = f"Successfully navigated to {self.last_page_goto_name} at {self.last_page_goto_result}"
            messages.append({
                'role': 'user',
                'content': f"**Page navigation results:** {page_goto_summary}"
            })
            
            # Clear the page goto result after using it
            self.last_page_goto_result = None
            self.last_page_goto_name = None
        
        # Add action history (Reflection already generated and injected into system prompt, just check actions left)
        if meta_data and 'action_history' in meta_data:
            action_number_left = getattr(self.args, 'max_steps', 15) - len(meta_data['action_history'])
            if action_number_left > 0:
                messages.append({
                    'role': 'user',
                    'content': f"ACTION NUMBER LEFT: You have **{action_number_left} actions left**, You MUST finish the task within the remaining actions! If the left action number is 1, YOU MUST yield the STOP action and provide the answer!"
                })
            # Inject stuck/repetition feedback when no progress or repeated actions are detected
            feedback = early_feedback
            if feedback and not is_stuck:
                self.logger.warning(f"[StuckDetection] Detected issue: {feedback}")
                messages.append({
                    'role': 'user',
                    'content': f"Feedback: {feedback}"
                })
            else:
                if not feedback:
                    # self.logger.info("[StuckDetection] No stuck/repetition detected")
                    pass
                
            # Note: visual history (image, action) pairs are intentionally not injected to reduce context.
                
                # Add recent trajectory information
        if trajectory:
            recent_obs = trajectory[-1]
            if isinstance(recent_obs, dict) and 'observation' in recent_obs:
                obs = recent_obs['observation']
                if 'image' in obs:
                    # Generate a description of the current page using LLM
                    page_description = self._generate_page_description(obs["image"])
                    
                    # Inject reasoning bank hints at the first turn when enabled
                    if getattr(self.args, 'use_reasoning_bank', False) and self.reasoning_bank is not None:
                        try:
                            query_text = f"{self.args.domain}: {current_task}\n{page_description}"
                            top_k = getattr(self.args, 'reasoning_top_k', 2)
                            domain_filter = self.args.domain if getattr(self.args, 'reasoning_domain_filter', True) else None
                            
                            # Use multimodal query if bank supports it
                            query_image = obs['image'] if self.reasoning_bank.use_multimodal else None
                            
                            # Pure top-k retrieval without label balancing
                            idx_scores = self.reasoning_bank.retrieve(
                                query_text=query_text, top_k=top_k, domain=domain_filter,
                                query_image_base64=query_image
                            )
                            # If nothing returned, no-op; fallback handled below if needed
                            if not idx_scores:
                                idx_scores = []
                            # Log retrieved indices and quick labels for traceability
                            try:
                                retrieved_info = []
                                for i, score in idx_scores[:top_k]:
                                    it = self.reasoning_bank.items[i] if 0 <= i < len(self.reasoning_bank.items) else {}
                                    retrieved_info.append({
                                        "index": int(i),
                                        "score": float(score),
                                        "label": it.get("label", ""),
                                        "title": it.get("title", "")[:80] if "title" in it else it.get("key_takeaway", "")[:80],
                                        "task_id": it.get("task_id", "")
                                    })
                                self.logger.info(f"[ReasoningBank] retrieved={retrieved_info}")
                            except Exception:
                                pass
                            
                            # Use multimodal hints if available, otherwise text-only
                            if self.reasoning_bank.use_multimodal:
                                hints_content = self.reasoning_bank.format_hints_multimodal(
                                    idx_scores[:top_k], max_images_per_hint=2
                                )
                                if hints_content:
                                    self.logger.info(f"[ReasoningBank] injected multimodal hints (count={len(hints_content)})")
                                    # Log full key takeaways and image paths for debugging
                                    try:
                                        for i, score in idx_scores[:top_k]:
                                            it = self.reasoning_bank.items[i] if 0 <= i < len(self.reasoning_bank.items) else {}
                                            takeaway = it.get("key_takeaway", "")
                                            img_path = it.get("after_image_path") or it.get("state_image_path", "")
                                            self.logger.info(f"  [{i}] {it.get('label', '')} | {takeaway}")
                                            self.logger.info(f"       image: {img_path}")
                                    except Exception:
                                        pass
                                    messages.append({'role': 'user', 'content': hints_content})
                            else:
                                hints_text = self.reasoning_bank.format_hints(idx_scores[:top_k])
                                if hints_text:
                                    # Log the final hint text injected to the prompt
                                    self.logger.info(f"[ReasoningBank] injected hints:\n{hints_text}")
                                    messages.append({'role': 'user', 'content': hints_text})
                        except Exception as e:
                            self.logger.warning(f"[ReasoningBank] injection failed: {e}")
                            pass
                    
                    # Add SoM legend if available
                    if 'content_str' in obs:
                        messages.append({
                            'role': 'user',
                            'content': f"**SoM Element Legend** (use these numeric IDs for element_id when possible):\n{obs['content_str']}"
                        })
                    
                    # Add the current screenshot with generated description
                    messages.append({
                        'role': 'user',
                        'content': [
                            {"type": "text", "text": "The following is the current page description and screenshot: " + page_description},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{obs['image']}"
                                }
                            } 
                        ]
                    })
            
            reminders_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "reminders.txt")
            reminders_content = ""
            if os.path.exists(reminders_path):
                with open(reminders_path, "r", encoding="utf-8") as f:
                    reminders_content = f.read()
            messages.append({
                'role': 'user',
                'content': reminders_content + f"""**Current task:** {current_task}\nWhat would you like to do next?"""})
        
        return messages, meta_data
    
    def _process_response(self, response: str, trajectory: Trajectory, page: Optional[Any] = None, intent: Optional[str] = None, meta_data: Optional[Dict[str, Any]] = None) -> Action:
        """Process the LLM response and convert to Action."""
        start_url = ""
        if trajectory and isinstance(trajectory[0], dict):
            start_url = trajectory[0].get("current_url") or ""
            if not start_url and isinstance(trajectory[0].get("info"), dict):
                p = trajectory[0]["info"].get("page")
                if p is not None and hasattr(p, "url"):
                    start_url = getattr(p, "url", "") or ""
        # ############### Case for UI-Ins-7B ###############
        # if '<think>' in response and '</think>' in response:
        #     # Find positions
        #     first_end_think = response.find('</think>')
        #     second_start_think = response.find('<think>', first_end_think)
        #     # Extract content between first </think> and second <think>
        #     if first_end_think != -1 and second_start_think != -1:
        #         response = response[first_end_think + len('</think>'):second_start_think].strip()
        #     # Handle case where there's no second <think>
        #     elif first_end_think != -1:
        #         response = response[first_end_think + len('</think>'):].strip()
        # ############### Case for UI-Ins-7B ###############
        parsed_response = parse_action_json(response)
        try:
            func_name = parsed_response['function_call']['name']
            func_args = parsed_response['function_call']['arguments']
            
            # Heuristic: if the intent is to navigate back but model returned a click with "go back" semantics, coerce to go_back
            try:
                desc_lower = str(func_args.get("description", "")).lower()
                if func_name in ["click", "selection"] and any(
                    p in desc_lower for p in ["go back", "back to", "navigate back", "return to previous page", "previous page"]
                ):
                    return create_go_back_action(url=start_url)
            except Exception:
                pass
            
            # Handle different function calls using the actions module
            if func_name == 'click':
                return create_click_action(
                    element_id=func_args.get('element_id', ''),
                    # coords=func_args.get('coords', ''),
                    coords='',
                    description=func_args.get('description', ''),
                    reasoning=func_args.get('reasoning', '')
                )
            elif func_name in ['selection', 'select']:
                return create_selection_action(
                    element_id=func_args.get('element_id', ''),
                    # coords=func_args.get('coords', ''),
                    coords='',
                    description=func_args.get('description', ''),
                    reasoning=func_args.get('reasoning', '')
                )
            elif func_name in ['type', 'search']:
                return create_type_action(
                    text=func_args.get('text', ''),
                    element_id=func_args.get('element_id', ''),
                    # coords=func_args.get('coords', ''),
                    coords='',
                    field_description=func_args.get('field_description', ''),
                    reasoning=func_args.get('reasoning', '')
                )
            elif func_name == 'scroll':
                return create_scroll_action(
                    direction=func_args.get('direction', 'down'),
                    reasoning=func_args.get('reasoning', '')
                )
            elif func_name == 'wait':
                return create_wait_action(
                    seconds=2.0,  # Default as requested
                    reasoning=func_args.get('reasoning', '')
                )
            elif func_name == "go_back":
                return create_go_back_action(url=start_url)
            elif func_name == 'press_key':
                return create_key_press_action(
                    key_comb=func_args.get('key', 'enter'),
                    reasoning=func_args.get('reasoning', '')
                )
            elif func_name == 'stop':
                # Check for 'answer', 'reason', or 'reasoning' fields
                answer = func_args.get('answer') or func_args.get('reason') or func_args.get('reasoning') or 'Task completed'
                self.logger.info(f"Agent answer: {answer}")
                return create_stop_action(
                    answer=answer,
                    reasoning=func_args.get('reasoning', '')
                )
            
            elif func_name == 'content_analyzer':
                # Execute content analyzer and store results for next step
                # Get the tool from function_map
                tool = self.function_map.get(func_name)
                if tool:
                    # Add trajectory context, page (if available), and LLM to kwargs
                    kwargs = {'page': page}
                    tool.llm = self.tool_llm
                    result = tool.call(json.dumps(func_args), **kwargs)
                    # self.logger.info(f"Content analyzer result: {result}")
                    
                    # Store the analysis result for next step context
                    # ContentAnalyzerTool returns JSON string, so store it directly
                    self.last_analysis_result = result
                    
                    # Return a wait action to allow the agent to process the analysis result
                    return create_wait_action(
                        seconds=1.0,
                        reasoning=f"Content analysis completed. Analysis results will be available for the next step."
                    )
                else:
                    self.logger.error(f"Tool {func_name} not found in function_map")
                    return create_none_action()
                
            elif func_name == "map_search":
                tool = self.function_map.get(func_name)
                if tool:
                    result = tool.call(json.dumps(func_args))
                    url = result.strip() if result else ""
                    self.last_map_search_query = func_args.get("query", "")
                    self.last_map_search_result = result
                    return create_goto_url_action(url)
                self.logger.warning("map_search tool not in function_map")
                return create_none_action()

            elif func_name == "goto_url":
                tool = self.function_map.get(func_name)
                if tool:
                    tool.llm = self.tool_llm
                    result = tool.call(json.dumps(func_args))
                    url = result.strip() if result else ""
                    self.last_page_goto_name = func_args.get("page_name", "")
                    self.last_page_goto_result = result
                    return create_goto_url_action(url)
                self.logger.warning("goto_url tool not in function_map")
                return create_none_action()

        except Exception as e:
            self.logger.debug("No function call in response: %s", e)

        action = self._parse_natural_language_with_llm(response, page, start_url=start_url)
        if action and action.get("action_type") != ActionTypes.NONE:
            return action
        return create_none_action()
    

    def _parse_natural_language_with_llm(self, content: str, page: Optional[Any] = None, pure_text: bool = False, start_url: Optional[str] = None) -> Action:
        """Use LLM to parse natural language content and extract action information"""
        try:
            # Create a prompt for the LLM to parse the content
            system_prompt = """You are an expert at parsing natural language responses and converting them into structured actions for a GUI automation agent.

Available actions:
- click: Click on elements by describing what you want to click
- selection: Select an option from a dropdown menu by describing what you want to select
- type: Type text into input fields by describing the field
- press_key: Press specific keys (enter, delete, space, etc.)
- go_back: Navigate back to the previous page
- scroll: Scroll the page in different directions (up, down, left, right)
- wait: Wait for a specified number of seconds
- stop: Stop the task and provide final answer
- map_search: Navigate to Google Maps for geographical searches
- content_analyzer: Analyze page content and images (results will be available for next step)

Parse the given content and return a key-value list with the following structure:

action_type: click|selection|type|press_key|go_back|scroll|wait|stop|map_search|content_analyzer
element_id: id of the element to interact with (for click, selection and type action)
coords: coordinates of the element to interact with (for click, selection and type action), in the format of "<point>x1 y1</point>", and it should be valid with two numbers, without any other text!
description: description of what to click or select (for click and selection action)
text: text to type (for type action)
field_description: description of the field (for type action)
key: key to press (for press_key action)
direction: scroll direction (for scroll action)
seconds: number of seconds (for wait action)
answer: final answer (for stop action)
query: query to search (for map_search action)
reasoning: why this action is needed

EXAMPLES:

For a click action:

action_type: click
element_id: x
coords: <point>x1 y1</point>
description: the search button
reasoning: Need to click the search button to submit the query

For a selection action:

action_type: selection
element_id: x
coords: <point>x1 y1</point>
description: the price option of the dropdown menu want to select from
reasoning: Need to select the price option of the dropdown menu to select the option

For a type action:

action_type: type
element_id: x
coords: <point>x1 y1</point>
text: Sydney Opera House
field_description: the search input field
reasoning: Need to type the search query into the search field

For a press_key action:

action_type: press_key
key: enter
reasoning: Need to press the enter key to submit the query

For a scroll action:

action_type: scroll
direction: down
reasoning: Need to scroll down to load more content


For a wait action:

action_type: wait
seconds: 2.0
reasoning: Need to wait for 2 seconds to load the page


For a stop action (task completion):

action_type: stop
answer: Yes, there is a Ferris wheel in the center of Shanghai. It is the Sky Ring Ferris Wheel in Joy City Shanghai.
reasoning: The information confirms the presence of the Sky Ring Ferris Wheel in the center of Shanghai

For a map_search action:

action_type: map_search
query: Sydney Opera House
reasoning: Need to search for the Sydney Opera House on Google Maps

For a content_analyzer action:

action_type: content_analyzer
reasoning: Need to analyze the page content and images


IMPORTANT: 
1. If the content indicates that a task is complete or provides a final answer, use action_type "stop" with the answer.
2. Ensure all key-value pairs are on separate lines.
3. Values should not contain extra quotes.
4. If there are multiple actions, only output the first action.

REMEMBER: Output ONLY valid key-value pairs, nothing else."""

            user_prompt = f"Parse this content and extract the action:\n\n{content.split('assistant')[0]}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            msg, _, _ = self.tool_llm.chat(messages=messages, stream=False)
            result = getattr(msg, "content", None) or ""
            if not result:
                return create_none_action()

            # Extract individual fields
            action_data = {}
            key_list = ['action_type', 'element_id', 'reasoning', 'coords',
                        'description', 
                        'text', 'field_description', 
                        'page_name', 'url',
                        'query',
                        'key', 'direction', 'seconds', 'answer']
            for line in result.strip().splitlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    if key.strip() in key_list:
                        action_data[key.strip()] = value.strip()
            
            # Check if we found any action data
            if not action_data or "action_type" not in action_data:
                try:
                    parsed = json.loads(result)
                    action_data = dict(parsed.get("arguments", {}))
                    action_data["action_type"] = parsed.get("name", "")
                except Exception:
                    self.logger.debug("No valid action data in LLM response")
                    return create_none_action()
            
            # Convert to Action based on action_type
            action_type = action_data.get('action_type', '').lower()
            reasoning = action_data.get('reasoning', '')
            
            if pure_text:
                return {'name': action_data['action_type'], 'arguments': {'reasoning': action_data['reasoning'],
                                                                          'description': action_data.get('description', '')}}
            
            try:
                desc_lower = str(action_data.get("description", "")).lower()
                if action_type in ["click", "selection"] and any(
                    p in desc_lower for p in ["go back", "back to", "navigate back", "return to previous page", "previous page"]
                ):
                    return create_go_back_action(url=start_url or "")
            except Exception:
                pass
            
            if action_data['action_type'] == 'goto_url' and 'url' not in action_data:
                page_name = (action_data.get("page_name") or "").lower()
                page_urls = {
                    "wiki": "https://www.wikipedia.org/",
                    "map": "https://www.google.com/maps",
                }
                if "wiki" in page_name:
                    action_data["url"] = page_urls["wiki"]
                elif "map" in page_name:
                    action_data["url"] = page_urls["map"]
                else:
                    action_data["url"] = page_urls.get(page_name, "")
            
            if action_type == 'click':
                return create_click_action(
                    element_id=action_data.get('element_id', ''),
                    coords=action_data.get('coords', ''),
                    description=action_data.get('description', ''),
                    reasoning=reasoning
                )
            elif action_type in ['selection', 'select']:
                return create_selection_action(
                    element_id=action_data.get('element_id', ''),
                    coords=action_data.get('coords', ''),
                    description=action_data.get('description', ''),
                    reasoning=reasoning
                )
            elif action_type in ['type', 'search']:
                return create_type_action(
                    text=action_data.get('text', ''),
                    element_id=action_data.get('element_id', ''),
                    coords=action_data.get('coords', ''),
                    field_description=action_data.get('field_description', ''),
                    reasoning=reasoning
                )
            elif action_type == 'press_key':
                return create_key_press_action(
                    key_comb=action_data.get('key', 'enter'),
                    reasoning=reasoning
                )
            elif action_type == 'scroll':
                return create_scroll_action(
                    direction=action_data.get('direction', 'down'),
                    reasoning=reasoning
                )
            elif action_type == 'wait':
                return create_wait_action(
                    seconds=float(action_data.get('seconds', 2.0)),
                    reasoning=reasoning
                )
            elif action_type == "go_back":
                return create_go_back_action(url=start_url or "")
            elif action_type == 'stop':
                # Check for 'answer', 'reason', or 'reasoning' fields
                answer = action_data.get('answer') or action_data.get('reason') or action_data.get('reasoning') or 'Task completed'
                return create_stop_action(
                    answer=answer,
                    reasoning=reasoning
                )
            elif action_type == 'map_search':
                tool = self.function_map.get(action_type)
                if tool:
                    func_args = {
                        'query': action_data.get('query', action_data.get('reasoning', '')),
                        'reasoning': action_data.get('reasoning', '')
                    }
                    result = tool.call(json.dumps(func_args))
                    # Expect result to be a URL string; try to extract
                    url = result.strip()
                    # Store context
                    self.last_map_search_query = func_args.get('query', '')
                    self.last_map_search_result = result
                    # If we got a URL, emit a goto action so the env updates the page
                    return create_goto_url_action(url)
            elif action_type == 'goto_url':
                return create_goto_url_action(
                    url=action_data.get('url', '')
                )
            
            elif action_type == 'content_analyzer':
                tool = self.function_map.get(action_type)
                func_args = {
                    'query': action_data.get('query', ''),
                    'reasoning': action_data.get('reasoning', '')
                }
                if tool:
                    # Add trajectory context, page (if available), and LLM to kwargs
                    kwargs = {'page': page}
                    tool.llm = self.tool_llm
                    result = tool.call(json.dumps(func_args), **kwargs)
                    # self.logger.info(f"Content analyzer result: {result}")
                    
                    # Store the analysis result for next step context
                    # ContentAnalyzerTool returns JSON string, so store it directly
                    self.last_analysis_result = result
                    
                    # Return a wait action to allow the agent to process the analysis result
                    return create_wait_action(
                        seconds=1.0,
                        reasoning=f"Content analysis completed. Analysis results will be available for the next step."
                    )
            else:
                return create_none_action()
            
        except Exception as e:
            self.logger.warning("Error in LLM parsing: %s", e)
            return create_none_action()

    
    def _generate_page_description(self, image_base64: str) -> str:
        """Generate a description of the current page using the LLM"""
        try:
            # Create a prompt for the LLM to describe the page
            messages = [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant that analyzes web page screenshots and provides clear, concise descriptions of what you see. Focus on the main content, interactive elements, and overall purpose of the page.'
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Please describe this web page screenshot. Include the main content, any visible buttons, forms, or interactive elements, and the overall purpose of the page."
                        }
                    ]
                }
            ]
            
            # Get LLM response
            response, _, _ = self.tool_llm.chat(messages=messages, stream=False)
            if hasattr(response, 'content'):
                description = response.content
            else:
                description = str(response)
            
            description = description.replace("\"text\": \"{'role': 'assistant', 'content': '", "")
            description = description.replace("'}\"", "")
            description = description[:2000]
            return description if description else "Current page state - analyze this and decide what to do next"
            
        except Exception as e:
            self.logger.warning(f"Error generating page description: {e}")
            return "Current page state - analyze this and decide what to do next"
    
    def check_login(self, state_info: dict) -> bool:
        """Return True if the current page appears to be a login or CAPTCHA page."""
        obs = state_info.get("observation") or {}
        image_base64 = obs.get("image")
        if not image_base64:
            return False
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that checks if the current page is a login or CAPTCHA page. Answer only yes or no.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": "Is this a login or CAPTCHA page? Answer only yes or no."},
                ],
            },
        ]
        msg, _, _ = self.tool_llm.chat(messages=messages, stream=False)
        content = getattr(msg, "content", None) or str(msg)
        return "yes" in (content or "").lower()
        
    def reset(self, test_config_file: Optional[str] = None) -> None:
        """Reset per-task state for a new task. Call before each new task/config."""
        self.logger.info("Resetting agent for new task (config: %s)", test_config_file or "N/A")
        self.experience_memory = None
        self.experience_texts = None
        self.experience_images = None
        self.file_id_list = None
        self.last_analysis_result = None
        self.last_map_search_query = None
        self.last_map_search_result = None
        self.last_page_goto_name = None
        self.last_page_goto_result = None


def construct_agent(args: argparse.Namespace) -> FunctionCallAgent:
    """Construct a function call agent"""
    agent = FunctionCallAgent(args)
    return agent 