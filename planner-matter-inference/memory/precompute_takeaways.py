import argparse
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import io
import base64

TAKEAWAY_SYSTEM_PROMPT = (
    "You extract actionable heuristics from SUCCESSFUL GUI agent trajectories.\n"
    "Firstly identify keywords of the task (e.g. location search, distance check, all locations, nearest location, etc.) and list them in <keywords>...</keywords>"
    "Then return a concise actionable step-by-step summary for how the task was completed and list them in <steps>...</steps>"
    "Constraints:\n"
    "- Focus on WHAT the agent did to complete the task.\n"
    "- Do not include duplicate steps, remove the duplicate steps if there are any."
    "- No more than 5 steps in total.\n"
    """Strictly follow the format:
    <keywords>keyword1, keyword2, keyword3, ...</keywords>
    <steps>
    step1
    step2
    step3
    ...
    </steps>
    """
)


@dataclass(frozen=True)
class StepPair:
    image_url: str  # e.g. "data:image/png;base64,...."
    action_text: str


def _parse_action_json(message: str) -> Dict[str, Any]:
    """
    Minimal, dependency-free action JSON extraction.
    Returns {"function_call": {"name": ..., "arguments": ...}} or raises ValueError.
    """
    if not isinstance(message, str) or not message:
        raise ValueError("message must be a non-empty string")
    text = re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>', '', message, flags=re.IGNORECASE | re.DOTALL)

    m = re.search(r'Action:\s*(\{.*\})', text, flags=re.DOTALL)
    if m:
        obj = json.loads(m.group(1))
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            return {"function_call": obj}
        if isinstance(obj, dict) and "function_call" in obj:
            return {"function_call": obj["function_call"]}
        raise ValueError("Action: JSON missing required keys")

    fenced = re.findall(r"```json\s*([\s\S]*?)\s*```", text)
    if fenced:
        obj = json.loads(fenced[0].strip())
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            return {"function_call": obj}
        if isinstance(obj, dict) and "function_call" in obj:
            return {"function_call": obj["function_call"]}
        raise ValueError("Fenced JSON missing required keys")

    # Last resort: whole text as JSON
    obj = json.loads(text)
    if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
        return {"function_call": obj}
    if isinstance(obj, dict) and "function_call" in obj:
        return {"function_call": obj["function_call"]}
    raise ValueError("No parseable action JSON found")


def _vllm_chat_completion(
    *,
    server_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
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
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
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


def _iter_success_json_files(memory_root: str) -> List[str]:
    if not os.path.isdir(memory_root):
        raise NotADirectoryError(f"--memory_root is not a directory: {memory_root}")
    paths: List[str] = []
    for root, _dirs, files in os.walk(memory_root):
        if os.path.basename(root) != "success":
            continue
        for fn in files:
            if fn.endswith(".jsonl"):
                paths.append(os.path.join(root, fn))
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No *.jsonl files found under any 'success' folder in: {memory_root}")
    return paths

def resize_image_base64(base64_string: str) -> str:
    """Simple image resize to reduce token count. Accepts raw base64 or data URL."""
    try:
        s = base64_string.strip()
        if s.startswith("data:image"):
            s = s.split(",", 1)[-1] if "," in s else ""
        if not s:
            return base64_string
        image_data = base64.b64decode(s)
        image = Image.open(io.BytesIO(image_data))
        
        # Simple resize
        if image.width > 1024 or image.height > 1024:
            image = image.resize((max(image.width//2, 512), max(image.height//2, 512)), Image.Resampling.LANCZOS)
        
        # Save as JPEG with low quality
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=50)
        output.seek(0)
        
        # Return compressed base64
        return base64.b64encode(output.getvalue()).decode('utf-8')
        
    except Exception:
        return base64_string
    
def _extract_image_url_from_round_messages(messages: List[Dict[str, Any]]) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("Round 'messages' must be a non-empty list")
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image_url":
                continue
            image_url = (item.get("image_url") or {}).get("url")
            image_url = resize_image_base64(image_url)
            if not isinstance(image_url, str) or not image_url:
                raise ValueError("Found image_url item but missing image_url.url")
            return image_url
    raise ValueError("No user image_url found in round messages")


def _extract_action_text(response: Any) -> str:
    if not isinstance(response, str) or not response:
        raise ValueError("Round 'response' must be a non-empty string")
    parsed = _parse_action_json(response)
    if not isinstance(parsed, dict) or "function_call" not in parsed:
        raise ValueError("Could not parse action JSON from round response")
    fc = parsed["function_call"]
    if not isinstance(fc, dict) or "name" not in fc or "arguments" not in fc:
        raise ValueError("Parsed function_call missing required keys")
    name = fc["name"]
    args = fc["arguments"]
    if isinstance(args, str):
        args = json.loads(args)
    if not isinstance(args, dict):
        raise ValueError("function_call.arguments must be a dict (or JSON string of dict)")
    # Keep it compact; we want "strategy", not coordinates/ids.
    reasoning = str(args.get("reasoning", "")).strip()
    desc = str(args.get("description", "")).strip()
    field_desc = str(args.get("field_description", "")).strip()
    text = str(args.get("text", "")).strip()
    key = str(args.get("key", "")).strip()
    direction = str(args.get("direction", "")).strip()
    answer = str(args.get("answer", "")).strip()
    parts = [f"name={name}"]
    if reasoning:
        parts.append(f"reasoning={reasoning}")
    if desc:
        parts.append(f"description={desc}")
    if field_desc:
        parts.append(f"field={field_desc}")
    if text:
        parts.append(f"text={text}")
    if key:
        parts.append(f"key={key}")
    if direction:
        parts.append(f"direction={direction}")
    if answer:
        parts.append(f"answer={answer}")
    return " | ".join(parts)


def _load_step_pairs(trajectory_path: str, max_pairs_per_traj: int) -> Tuple[str, str, List[StepPair]]:
    with open(trajectory_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Trajectory file must be a JSON object: {trajectory_path}")
    task_id = str(obj.get("task_id", "")).strip()
    task_description = str(obj.get("task_description", "")).strip()
    if not task_description:
        raise ValueError(f"Missing task_description in: {trajectory_path}")
    rounds = obj.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError(f"Missing/empty rounds in: {trajectory_path}")

    pairs: List[StepPair] = []
    print(f"trajectory_path: {trajectory_path}")
    print(f"rounds: {len(rounds)}")
    for r in rounds:
        if not isinstance(r, dict):
            raise ValueError(f"Round must be a dict in: {trajectory_path}")
        messages = r.get("messages")
        response = r.get("response")
        image_url = _extract_image_url_from_round_messages(messages)
        try:
            action_text = _extract_action_text(response)
        except Exception as e:
            # continue
            action_text = response
        pairs.append(StepPair(image_url=image_url, action_text=action_text))
        if len(pairs) >= max_pairs_per_traj:
            break

    if not pairs:
        raise ValueError(f"No (image, action) pairs extracted from: {trajectory_path}")
    return task_id, task_description, pairs


def parse_takeaway_text(text: str) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Parse <keywords> and <steps> from model output. Returns (keywords, steps) or (None, None)."""
    if not isinstance(text, str):
        raise ValueError("Model output must be a string")
    s = text.strip()
    keywords = None
    if "<keywords>" in s and "</keywords>" in s:
        try:
            raw = s.split("<keywords>")[1].split("</keywords>")[0].strip()
            keywords = [k.strip() for k in raw.split(",") if k.strip()]
        except Exception:
            pass
    steps = None
    if "<steps>" in s and "</steps>" in s:
        try:
            raw = s.split("<steps>")[1].split("</steps>")[0].strip()
            steps = [line.strip() for line in raw.split("\n") if line.strip()]
        except Exception:
            pass
    return keywords, steps

def _build_messages(task_description: str, pairs: List[StepPair]) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = []
    user_content.append({"type": "text", "text": f"Task: {task_description}"})
    user_content.append({"type": "text", "text": "Trajectory (chronological):"})
    for i, p in enumerate(pairs, start=1):
        user_content.append({"type": "image_url", "image_url": {"url": p.image_url}})
        user_content.append({"type": "text", "text": f"Step {i} action: {p.action_text}"})
    return [
        {"role": "system", "content": TAKEAWAY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory_root", type=str, required=True)
    parser.add_argument("--server_url", type=str, required=True, help="OpenAI-compatible vLLM server URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--model", type=str, required=True, help="Model name served by vLLM, e.g. Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--max_pairs_per_traj", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    if args.max_pairs_per_traj <= 0:
        raise ValueError("--max_pairs_per_traj must be > 0")
    if os.path.exists(args.out_json) and not args.overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite to replace): {args.out_json}")

    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(args.out_json):
        with open(args.out_json, "r", encoding="utf-8") as f:
            out = json.load(f)
    else:
        out = {}
    for i, path in enumerate(_iter_success_json_files(args.memory_root)):
        file_id = os.path.basename(path).split(".")[0]
        if file_id in out and not args.overwrite:
            continue
        _task_id, task_description, pairs = _load_step_pairs(path, max_pairs_per_traj=args.max_pairs_per_traj)
        messages = _build_messages(task_description=task_description, pairs=pairs)
        
        keywords, steps = None, None
        temp_num = 0
        while (keywords is None or steps is None) and temp_num < 3:
            try:
                content = _vllm_chat_completion(
                    server_url=args.server_url,
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except Exception as e:
                content = ''
                temp_num += 1
                continue
            keywords, steps = parse_takeaway_text(content)
            temp_num += 1
        # takeaway = _validate_takeaway_text(content)
        print(f"file_id: {file_id}, keywords: {keywords}, steps: {steps}")
        out[file_id] = {"intent": task_description, "keywords": keywords, "steps": steps, "src": path}

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

