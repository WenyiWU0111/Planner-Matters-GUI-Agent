"""Function Call Agent for GUI Agent with ReAct paradigm."""
import argparse
import json
import logging
import os
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

# Tool name -> class for building specs and function map (single source of truth)
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
    "content_analyzer": ContentAnalyzerTool,
}

def resize_image_base64(base64_string: str) -> str:
    """Simple image resize to reduce token count."""
    try:
        from PIL import Image
        import io

        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        if image.width > 1024 or image.height > 1024:
            image = image.resize(
                (max(image.width // 2, 512), max(image.height // 2, 512)),
                Image.Resampling.LANCZOS,
            )
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=50)
        output.seek(0)
        return base64.b64encode(output.getvalue()).decode("utf-8")
    except Exception:
        return base64_string


class FunctionCallAgent:
    """Function-call agent for GUI interactions using ReAct with model calls."""

    def __init__(self, args: argparse.Namespace, **kwargs) -> None:
        """Initialize model, tools, and per-session state. Call reset() before each task."""
        self.args = args
        self.logger = logging.getLogger("logger")

        # Models
        self.llm = create_model(args)
        self.tool_llm = load_tool_llm(args)

        # Tool definitions and instances (shared across tasks)
        self.function_map = self._build_function_map()
        self.tool_specs = self._build_tool_specs()

        # Optional memory (not used in this implementation)
        self.memory = None

        # Per-task state (must be reset before each new task)
        self._reset_task_state()

    def _reset_task_state(self) -> None:
        """Reset all state that should not carry over between tasks."""
        self.current_step = 0
        self.last_analysis_result = None
        self.last_map_search_query = None
        self.last_map_search_result = None
        self.last_page_goto_name = None
        self.last_page_goto_result = None

    def _define_functions(self) -> List[str]:
        """Function names exposed to the agent (subset of NAME_TO_CLS)."""
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

    def _build_tool_specs(self) -> List[Dict[str, Any]]:
        """Build tool specs for the prompt from defined functions (no mutation of tool.parameters)."""
        specs: List[Dict[str, Any]] = []
        for name in self._define_functions():
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
            if raw_params is None:
                parameters = None
            else:
                properties = dict(raw_params.get("properties", {}))
                parameters = {**raw_params, "properties": properties}
            specs.append({
                "name": getattr(tool, "name", name),
                "description": getattr(tool, "description", "No description"),
                "parameters": parameters,
            })
        return specs

    def _build_function_map(self) -> Dict[str, Any]:
        """Build name -> tool instance map and inject tool_llm."""
        function_map: Dict[str, Any] = {}
        for name, cls in NAME_TO_CLS.items():
            try:
                tool = cls()
                tool.llm = self.tool_llm
                function_map[name] = tool
            except Exception as e:
                self.logger.warning("Failed to initialize tool %s: %s", name, e)
        return function_map
    
    def _get_system_message(self, intent: str, plan: str) -> str:
        """Build system message for the agent (ReAct + tools)."""
        prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
        tool_selection_path = os.path.join(prompt_dir, "tool_selection.txt")
        if os.path.exists(tool_selection_path):
            with open(tool_selection_path, "r", encoding="utf-8") as f:
                tools_section = f.read()
        else:
            tools_section = "(Tool list not found; use click, type, scroll, wait, go_back, stop, map_search, content_analyzer, goto_url as needed.)"

        # basic system prompt if file doesn't exist
        system_prompt = """You are a helpful GUI execution agent following the given plan to complete tasks. 

The task is as follows:
{intent}

The plan is as follows:
{plan}

Please use the following tools to complete tasks:
{tools_section}

Constraints:
- Strictly follow the plan step by step, follow the next step decision in the plan if it exists.
- Please specify the number label of the item you want to interact with, in the description of the action.
- Don't repeat the same action multiple times - try different approaches if something doesn't work.
- Pay attention to images, for many questions about the shape, color, location, etc., you can answer according to the images.
- If the current search term yields no results, you must adjust your search term and try again.
- You must provide an answer within the remaining steps.
- When you answer STOP, you must provide a Detailed answer to describe what you have found and what you have done.
- If you want to type, no need to click first. Directly yield TYPE action with the search bar location and the text you want to type.
- Only output one action at a time, do not output multiple actions at once
"""
        system_prompt = system_prompt.format(intent=intent, tools_section=tools_section, plan=plan)
        return system_prompt

    def next_action_custom(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: Dict[str, Any],
    ):
        """Generate the next action using function calling with ReAct paradigm"""
        
        self.current_step += 1
        print('*'*50, 'current step: ', self.current_step, '*'*50)
        
        messages, meta_data = self._prepare_messages(
            trajectory, 
            intent, 
            meta_data, 
        )

        responses, original_inputs, original_outputs = self.llm.chat(messages=messages, stream=False)
        meta_data['original_inputs'] = original_inputs
        meta_data['original_outputs'] = original_outputs
        meta_data['original_responses'] = responses
        meta_data['response_history'].append(responses.content)
        print('*'*50, 'responses', '*'*50)
        print(responses.content)
        print('*'*50, 'responses', '*'*50)
        
        # Extract page if available in meta_data or elsewhere; pass explicitly
        page_for_tools = meta_data.get('page')
            
        # Process the response
        action = self._process_response(responses.content, trajectory, page_for_tools, intent, meta_data)
        print('*'*50, 'action', '*'*50)
        print(action)
        print('*'*50, 'action', '*'*50)
        
        return (action, meta_data)
    
    def _get_current_screenshot(self, trajectory: Trajectory) -> Optional[str]:
        """Extract the current screenshot from the trajectory for multimodal memory retrieval."""
        recent_obs = trajectory[-1]
        if isinstance(recent_obs, dict) and 'observation' in recent_obs:
            obs = recent_obs['observation']
            if 'image' in obs:
                return obs['image']
        return None
    
    def _prepare_messages(self, trajectory: Trajectory, intent: str, meta_data: Dict[str, Any]) -> Tuple[List[Dict], Dict[str, Any]]:
        """Prepare messages for the LLM with ReAct context
        
        Args:
            trajectory: Current trajectory
            intent: Task intent
            meta_data: Metadata dictionary
            
        Returns:
            Tuple of (messages, meta_data)
        """
        messages = []
        plan = meta_data.get('task_plan', {}).get('plan', '')
        # Add system message
        messages.append({
            'role': 'system',
            'content': self._get_system_message(intent=intent, plan=plan)
        })
        
        # Add analysis results context if available
        if self.last_analysis_result:
            analysis_summary = self.last_analysis_result
            messages.append({
                'role': 'user',
                'content': f"**Content analysis results:** {analysis_summary}"
            })
            # Clear the analysis result after using it
            self.last_analysis_result = None
        
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
            recent_actions = '\n'.join([f"{i+1}. {action}" for i, action in enumerate(meta_data.get('action_history')[-5:])])
            if action_number_left > 0:
                messages.append({
                    'role': 'user',
                    'content': f"ACTION NUMBER LEFT: You have **{action_number_left} actions left**, You MUST finish the task within the remaining actions! If the left action number is 1, YOU MUST yield the STOP action and provide the answer!"
                })
                messages.append({
                    'role': 'user',
                    'content': f"The following are recent actions you have taken: {recent_actions}"
                })
    
        # Add recent trajectory information
        if trajectory:
            recent_obs = trajectory[-1]
            if isinstance(recent_obs, dict) and 'observation' in recent_obs:
                obs = recent_obs['observation']
                if 'image' in obs:
                    # Generate a description of the current page using LLM
                    page_description = self._generate_page_description(obs["image"])
                    
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
            messages.append({
                'role': 'user',
                'content': f"When you answer STOP, you must provide a **Detailed answer to describe what you have found and what you have done**."
            })
            messages.append({
                'role': 'user',
                'content': f"""***Task:** {intent}\n***Plan:** {plan}\nStrictly follow the given plan, what is the next action and corresponding parameters you will take?"""})
        
        return messages, meta_data
    
    def _process_response(self, response: str, trajectory: Trajectory, page: Optional[Any] = None, intent: Optional[str] = None, meta_data: Optional[Dict[str, Any]] = None) -> Action:
        """Process the LLM response and convert to Action"""
        start_url = trajectory[0]['current_url']
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
                desc_lower = str(func_args.get('description', '')).lower()
                if func_name in ['click', 'selection'] and any(phrase in desc_lower for phrase in [
                    'go back', 'back to', 'navigate back', 'return to previous page', 'previous page'
                ]):
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
            elif func_name == 'go_back':
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
                
            elif func_name == 'map_search':
                # Execute map search tool which now returns a Google Maps URL
                tool = self.function_map.get(func_name)
                if tool:
                    result = tool.call(json.dumps(func_args))
                    url = result.strip() if result else ""
                    self.last_map_search_query = func_args.get("query", "")
                    self.last_map_search_result = result
                    return create_goto_url_action(url)
                else:
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
                else:
                    self.logger.warning("goto_url tool not in function_map")
                    return create_none_action()

        except Exception as e:
            self.logger.debug("No function call in response: %s", e)
        action = self._parse_natural_language_with_llm(response, page, start_url=start_url)
        if action and action.get("action_type") != ActionTypes.NONE:
            return action
        return create_none_action()
    

    def _parse_natural_language_with_llm(self, content: str, page: Optional[Any] = None, pure_text=False, start_url: Optional[str] = None) -> Action:
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
            
            # print(f"Extracted action data: {action_data}")
            
            # Convert to Action based on action_type
            action_type = action_data.get('action_type', '').lower()
            reasoning = action_data.get('reasoning', '')
            
            if pure_text:
                return {'name': action_data['action_type'], 'arguments': {'reasoning': action_data['reasoning'],
                                                                          'description': action_data.get('description', '')}}
            
            # Heuristic: coerce to go_back when NL indicates back navigation but action parsed as click/selection
            try:
                desc_lower = str(action_data.get('description', '')).lower()
                if action_type in ['click', 'selection'] and any(phrase in desc_lower for phrase in [
                    'go back', 'back to', 'navigate back', 'return to previous page', 'previous page'
                ]):
                    return create_go_back_action(url=start_url)
            except Exception:
                pass
            
            if action_data.get("action_type") == "goto_url" and not action_data.get("url"):
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
            elif action_type == 'go_back':
                return create_go_back_action(url=start_url)
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
            print(f"Error in LLM parsing: {e}")
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
                "content": "You are a helpful assistant that checks if the current page is a login page or CAPTCHA verification page. Answer only yes or no.",
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
        self._reset_task_state()


def construct_agent(args: argparse.Namespace) -> FunctionCallAgent:
    """Construct a function call agent"""
    agent = FunctionCallAgent(args)
    return agent 