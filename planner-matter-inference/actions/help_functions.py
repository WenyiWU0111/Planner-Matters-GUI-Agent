import json
import re
from typing import Optional, Dict
from agent.llm_config import VLLMModel

def parse_action_json(message: str) -> Optional[Dict]:
    """
    Parses the action JSON from a ChatCompletionMessage content string.

    Relaxed parsing rules:
      - Strip <think> ... </think> blocks
      - Accept code-fenced ```json ... ``` blocks
      - Try to extract the first balanced JSON object from mixed text
      - Accept {"action": {...}} or {"name":..., "arguments":...} and normalize to {"function_call": {...}}
      - Normalize coords from [x, y] to "<point>x y</point>"

    Returns a dict with key "function_call" on success; otherwise returns the original message (to trigger NL fallback).
    """
    def _strip_think_blocks(text: str) -> str:
        return re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>', '', text or '', flags=re.IGNORECASE | re.DOTALL)

    def _extract_first_json_object(text: str) -> Optional[str]:
        start = text.find('{')
        while start != -1:
            depth = 0
            for idx in range(start, len(text)):
                ch = text[idx]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:idx + 1]
            start = text.find('{', start + 1)
        return None

    def _normalize_function_call_object(obj: Dict) -> Optional[Dict]:
        fc = None
        if isinstance(obj, dict):
            if "function_call" in obj and isinstance(obj["function_call"], dict):
                fc = obj["function_call"]
            elif "action" in obj and isinstance(obj["action"], dict):
                fc = obj["action"]
            elif "name" in obj and "arguments" in obj:
                fc = {"name": obj.get("name"), "arguments": obj.get("arguments", {})}
        if not isinstance(fc, dict):
            return None
        args = fc.get("arguments", {})
        if isinstance(args, dict):
            coords_val = args.get("materials", None)  # placeholder to keep indentation consistent
        # Normalize coords if provided in different forms
        if isinstance(fc.get("arguments"), dict):
            cval = fc["arguments"].get("coords")
            if isinstance(cval, list) and len(cval) >= 2:
                try:
                    x = int(cval[0])
                    y = int(cval[1])
                    fc["arguments"]["coords"] = f"<point>{x} {y}</point>"
                except Exception:
                    pass
            elif isinstance(cval, str):
                m = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', cval)
                if m:
                    fc["arguments"]["coords"] = f"<point>{m.group(1)} {m.group(2)}</point>"
        return {"function_call": {"name": fc.get("name"), "arguments": fc.get("arguments", {})}}

    text = _strip_think_blocks(message or "")

    # Try explicit 'Action: {...}'
    m = re.search(r'Action:\s*(\{.*\})', text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            normalized = _normalize_function_call_object(obj)
            if normalized:
                return normalized
            return {"function_call": obj}
        except Exception:
            pass

    # Try code-fenced JSON
    fenced = re.findall(r"```json\s*([\s\S]*?)\s*```", text)
    if fenced:
        blob = fenced[0].strip()
        try:
            obj = json.loads(blob)
            normalized = _normalize_function_call_object(obj)
            if normalized:
                return normalized
            return {"function_call": obj}
        except Exception:
            cand = _extract_first_json_object(blob)
            if cand:
                try:
                    obj = json.loads(cand)
                    normalized = _normalize_function_call_object(obj)
                    if normalized:
                        return normalized
                except Exception:
                    pass

    # Try first balanced JSON object within the text
    cand = _extract_first_json_object(text)
    if cand:
        try:
            obj = json.loads(cand)
            normalized = _normalize_function_call_object(obj)
            if normalized:
                return normalized
        except Exception:
            pass

    # As last resort, try whole text as JSON
    try:
        obj = json.loads(text)
        normalized = _normalize_function_call_object(obj)
        if normalized:
            return normalized
    except Exception:
        pass

    # Give original text back to trigger NL-based fallback
    return message


def parse_natural_language_with_llm(content: str, tool_llm: VLLMModel) -> str:
        """Use LLM to parse natural language content and extract action information"""
        # Create a prompt for the LLM to parse the content
        system_prompt = """You are an expert at parsing natural language responses and converting them into structured actions for a GUI automation agent.

Available actions:
- click: Click on elements by describing what you want to click
- selection: Select an option from a dropdown menu by describing what you want to select
- type: Type text into input fields by describing the field
- press_key: Press specific keys (enter, delete, space, etc.)
- scroll: Scroll the page in different directions (up, down, left, right)
- wait: Wait for a specified number of seconds
- stop: Stop the task and provide final answer
- map_search: Navigate to Google Maps for geographical searches
- content_analyzer: Analyze page content and images (results will be available for next step)

Parse the given content and return a key-value list with the following structure:

action_type: click|selection|type|press_key|scroll|wait|stop|map_search|content_analyzer
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

REMEMBER: Output ONLY valid key-value pairs, nothing else."""

        user_prompt = f"Parse this content and extract the action:\n\n{content.split('assistant')[0]}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Call the LLM
        result = ''
        while result == '':
            response, _, _ = tool_llm.chat(messages=messages, stream=False)
            result = response.content
            print(f"LLM response: {result}")
            if result != '':
                break
        return result