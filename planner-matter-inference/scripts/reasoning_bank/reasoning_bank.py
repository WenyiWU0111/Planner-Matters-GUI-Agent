import os
import json
import re
import base64
from typing import List, Dict, Optional, Tuple
from io import BytesIO

import numpy as np
import faiss
from PIL import Image

from memory.help_functions import CLIPTextSimilarity, CLIPMultimodalSimilarity


REASONING_BANK_DEFAULT = "memory/reasoning_bank.jsonl"
REASONING_INDEX_DEFAULT = "memory_index/reasoning_bank_text"
REASONING_BANK_MM_DEFAULT = "memory/reasoning_bank_mm.jsonl"
REASONING_INDEX_MM_DEFAULT = "memory_index/reasoning_bank_mm"

# Resolve paths
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
# Workspace root is one level above project root
_workspace_root = os.path.dirname(_project_root)
MEDIA_ROOT_DIR = os.path.join(_workspace_root, "media")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _save_image_from_base64(base64_str: str, output_path: str, max_width: int = 768, quality: int = 80) -> None:
    """Save base64 image to disk as JPEG with downscaling."""
    _ensure_dir(output_path)
    # Remove data URL prefix if present
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',', 1)[1]
    
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    
    # Downscale if needed
    if img.width > max_width:
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_width, new_height), Image.LANCZOS)
    
    # Convert to RGB and save as JPEG
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGB')
    img.save(output_path, 'JPEG', quality=quality)


def _load_image_as_base64(image_path: str) -> str:
    """Load image from disk and encode as base64 with data URL prefix."""
    with open(image_path, 'rb') as f:
        img_data = f.read()
    b64 = base64.b64encode(img_data).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def parse_trajectory_rounds(trajectory_obj: Dict) -> Tuple[str, List[Dict]]:
    """
    Parse trajectory JSON to extract task and per-step information.
    
    Returns:
        task_description: str
        steps: List[Dict] with keys: screenshot (base64), action (dict), response (str)
    """
    task = trajectory_obj.get('task_description', '')
    rounds = trajectory_obj.get('rounds', [])
    
    steps = []
    for r in rounds:
        # Extract screenshot from messages
        screenshot = None
        for msg in r.get('messages', []):
            if isinstance(msg.get('content'), list):
                for item in msg['content']:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        screenshot = item['image_url']['url']
                        break
        
        # Extract action from response (JSON block)
        response = r.get('response', '')
        action = _parse_action_from_response(response)
        
        steps.append({
            'screenshot': screenshot,
            'action': action,
            'response': response
        })
    
    return task, steps


def _parse_action_from_response(response: str) -> Optional[Dict]:
    """Extract action JSON from response string."""
    try:
        # Remove markdown code blocks if present
        response = response.replace('```json', '').replace('```', '')
        
        # Find JSON block - look for opening brace followed by "name"
        # Handle whitespace and newlines
        import re
        match = re.search(r'\{\s*"name"\s*:', response)
        if not match:
            match = re.search(r"\{\s*'name'\s*:", response)
        if not match:
            return None
        
        start = match.start()
        
        # Find matching closing brace
        brace_count = 0
        end = start
        for i in range(start, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        
        action_str = response[start:end]
        # Handle single quotes
        action_str = action_str.replace("'", '"')
        return json.loads(action_str)
    except Exception:
        return None


def _read_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def _append_jsonl(path: str, rows: List[Dict]) -> None:
    _ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _concat_text(item: Dict) -> str:
    """Concatenate text fields for text-only items (legacy)."""
    parts = [
        item.get("title", ""),
        item.get("description", ""),
        item.get("content", ""),
        item.get("label", ""),
        item.get("domain", ""),
        item.get("dataset", ""),
    ]
    return "\n".join([p for p in parts if p])


def _concat_text_mm(item: Dict) -> str:
    """Concatenate text fields for multimodal items."""
    parts = [
        item.get("key_takeaway", ""),
        item.get("pre_state_hint", ""),
        item.get("post_state_hint", ""),
        item.get("label", ""),
        item.get("domain", ""),
        item.get("dataset", ""),
    ]
    return "\n".join([p for p in parts if p])


class ReasoningBank:
    """
    Reasoning bank with CLIP embeddings + FAISS IP index.
    Supports both text-only (legacy) and multimodal (text + image) modes.
    """

    def __init__(self,
                 bank_path: str = REASONING_BANK_DEFAULT,
                 index_base_path: str = REASONING_INDEX_DEFAULT,
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 use_multimodal: bool = False) -> None:
        self.bank_path = bank_path
        self.index_path = f"{index_base_path}.faiss"
        self.meta_path = f"{index_base_path}.json"
        self.use_multimodal = use_multimodal
        
        if use_multimodal:
            self.clip = CLIPMultimodalSimilarity(model_name=clip_model_name)
        else:
            self.clip = CLIPTextSimilarity(model_name=clip_model_name)
        
        self.items: List[Dict] = _read_jsonl(self.bank_path)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.embeddings: Optional[np.ndarray] = None
        self._load_or_build_index()

    def _load_or_build_index(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                # meta['count'] can be used for sanity checks
                return
            except Exception:
                pass
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        if not self.items:
            self.index = None
            self.embeddings = None
            return
        
        if self.use_multimodal:
            emb = self._build_multimodal_embeddings(self.items)
        else:
            texts = [_concat_text(it) for it in self.items]
            emb = self.clip.get_text_embeddings(texts)
        
        if emb is None or len(emb) == 0:
            self.index = None
            self.embeddings = None
            return
        
        # normalize for cosine similarity via inner product
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb.astype("float32"))
        self.index = index
        self.embeddings = emb
        _ensure_dir(self.index_path)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"count": len(self.items), "multimodal": self.use_multimodal}, f)
    
    def _build_multimodal_embeddings(self, items: List[Dict]) -> Optional[np.ndarray]:
        """Build multimodal embeddings (text + image) for items."""
        texts = [_concat_text_mm(it) for it in items]
        base64_images = []
        
        for it in items:
            # Load after_image as primary visual evidence
            after_path = it.get('after_image_path')
            if after_path and os.path.exists(after_path):
                try:
                    img_b64 = _load_image_as_base64(after_path)
                    base64_images.append(img_b64)
                except Exception:
                    base64_images.append(None)
            else:
                base64_images.append(None)
        
        # Get multimodal embeddings
        try:
            emb = self.clip.get_multimodal_embeddings(texts, base64_images)
            return emb
        except Exception as e:
            print(f"Failed to build multimodal embeddings: {e}")
            return None

    def add_items(self, new_items: List[Dict], persist: bool = True, update_index: bool = True) -> None:
        if not new_items:
            return
        self.items.extend(new_items)
        if persist:
            _append_jsonl(self.bank_path, new_items)
        if update_index:
            # incremental add
            if self.use_multimodal:
                emb = self._build_multimodal_embeddings(new_items)
            else:
                texts = [_concat_text(it) for it in new_items]
                emb = self.clip.get_text_embeddings(texts)
            
            if emb is not None and len(emb) > 0:
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                if self.index is None:
                    dim = emb.shape[1]
                    self.index = faiss.IndexFlatIP(dim)
                    self.embeddings = emb
                    self.index.add(emb.astype("float32"))
                else:
                    self.index.add(emb.astype("float32"))
                    if self.embeddings is not None:
                        self.embeddings = np.vstack([self.embeddings, emb])
                _ensure_dir(self.index_path)
                faiss.write_index(self.index, self.index_path)
                with open(self.meta_path, "w", encoding="utf-8") as f:
                    json.dump({"count": len(self.items), "multimodal": self.use_multimodal}, f)

    def retrieve(self,
                 query_text: str,
                 top_k: int = 2,
                 domain: Optional[str] = None,
                 label: Optional[str] = None,
                 query_image_base64: Optional[str] = None) -> List[Tuple[int, float]]:
        if self.index is None or not self.items:
            return []
        # optional filter by domain and/or label
        candidate_indices = list(range(len(self.items)))
        if domain:
            candidate_indices = [i for i, it in enumerate(self.items) if it.get("domain", "") == domain]
        if label:
            candidate_indices = [i for i in candidate_indices if self.items[i].get("label", "") == label]
            if not candidate_indices:
                candidate_indices = list(range(len(self.items)))
        
        # embed query
        if self.use_multimodal and query_image_base64:
            # Multimodal query
            try:
                q = self.clip.get_multimodal_embeddings([query_text], [query_image_base64])
            except Exception as e:
                print(f"Multimodal query failed, falling back to text-only: {e}")
                q = self.clip.get_text_embeddings([query_text])
        else:
            # Text-only query
            q = self.clip.get_text_embeddings([query_text])
        
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        # search
        D, I = self.index.search(q.astype("float32"), min(top_k * 5, len(self.items)))
        # re-rank with domain filter if needed
        ranked: List[Tuple[int, float]] = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            if idx not in candidate_indices:
                continue
            ranked.append((int(idx), float(score)))
            if len(ranked) >= top_k:
                break
        if not domain and not label:
            ranked = ranked[:top_k]
        return ranked

    def format_hints(self, indices_scores: List[Tuple[int, float]]) -> str:
        if not indices_scores:
            return ""
        lines = ["Reasoning hints (retrieved):"]
        for rank, (i, _) in enumerate(indices_scores, 1):
            it = self.items[i]
            title = it.get("title", "").strip()
            desc = it.get("description", "").strip()
            if title and desc:
                lines.append(f"{rank}) {title} – {desc}")
            elif title:
                lines.append(f"{rank}) {title}")
            elif desc:
                lines.append(f"{rank}) {desc}")
        return "\n".join(lines)
    
    def format_hints_multimodal(self, indices_scores: List[Tuple[int, float]], 
                                max_images_per_hint: int = 1) -> List[Dict]:
        """
        Format hints with multimodal content (text + images).
        
        Returns a list of message content items suitable for injection into agent messages.
        Each hint includes key_takeaway text and optionally before/after images.
        """
        if not indices_scores:
            return []
        
        content_items = []
        
        # Add header
        content_items.append({
            "type": "text",
            "text": "Before acting on the current page, consider these retrieved cases (state snapshot + key takeaway):"
        })
        
        for rank, (i, score) in enumerate(indices_scores, 1):
            if i < 0 or i >= len(self.items):
                continue
            
            it = self.items[i]
            key_takeaway = it.get("key_takeaway", "").strip()
            pre_hint = it.get("pre_state_hint", "").strip()
            post_hint = it.get("post_state_hint", "").strip()
            
            if not key_takeaway:
                # Fallback to legacy format
                title = it.get("title", "").strip()
                desc = it.get("description", "").strip()
                if title and desc:
                    key_takeaway = f"{title} – {desc}"
                elif title:
                    key_takeaway = title
                elif desc:
                    key_takeaway = desc
            
            if key_takeaway:
                content_items.append({
                    "type": "text",
                    "text": f"\n{rank}) Key takeaway: {key_takeaway}"
                })
            if pre_hint:
                content_items.append({"type": "text", "text": f"Pre-state: {pre_hint}"})
            if post_hint:
                content_items.append({"type": "text", "text": f"Post-state: {post_hint}"})
            
            # Add images if available
            if max_images_per_hint > 0:
                # Resolve paths; prefer explicit fields, fallback to legacy aliases
                state_path = it.get("after_image_path") or it.get("state_image_path")
                before_path = it.get("before_image_path") or it.get("context_prev_image_path")
                
                # Transition annotation
                content_items.append({"type": "text", "text": "Transition (before → after):"})
                
                # Include previous (context) image first
                if max_images_per_hint >= 2 and before_path and os.path.exists(before_path):
                    try:
                        img_b64 = _load_image_as_base64(before_path)
                        content_items.append({"type": "image_url", "image_url": {"url": img_b64}})
                    except Exception:
                        pass
                # Include state (current) image
                if state_path and os.path.exists(state_path):
                    try:
                        img_b64 = _load_image_as_base64(state_path)
                        content_items.append({"type": "image_url", "image_url": {"url": img_b64}})
                    except Exception:
                        pass
        
        return content_items


def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_markdown_items(md_text: str) -> List[Dict]:
    """
    Parse blocks of:
    # Memory Item i
    ## Title ...
    ## Description ...
    ## Content ...
    """
    items: List[Dict] = []
    # split by "# Memory Item" markers
    blocks = re.split(r"\n\s*#\s*Memory Item[^\n]*\n", "\n" + md_text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        def find(section: str) -> str:
            m = re.search(rf"^##\s*{section}\s*(.+)$", block, flags=re.MULTILINE)
            return m.group(1).strip() if m else ""
        title = find("Title")
        desc = find("Description")
        content = find("Content")
        if title or desc or content:
            items.append({"title": title, "description": desc, "content": content})
    return items[:3]


def distill_reasoning_items(tool_llm,
                            prompts_dir: str,
                            is_success: bool,
                            query: str,
                            trajectory_text: str,
                            dataset: str,
                            domain: str,
                            task_id: str,
                            source_path: str,
                            max_items: int = 3) -> List[Dict]:
    """
    Use LLM to convert a trajectory into 1–3 reasoning items.
    """
    prompt_file = "reasoning_bank_success.md" if is_success else "reasoning_bank_failure.md"
    prompt_path = os.path.join(prompts_dir, prompt_file)
    system_prompt = _load_prompt(prompt_path)
    user_prompt = f"Query: {query}\n\nTrajectory:\n{trajectory_text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        resp, _, _ = tool_llm.chat(messages=messages, stream=False)
        text = getattr(resp, "content", "") or str(resp)
        items = _parse_markdown_items(text)
    except Exception:
        items = []
    # enrich metadata
    label = "success" if is_success else "failure"
    out: List[Dict] = []
    for it in items[:max_items]:
        it.update({
            "label": label,
            "dataset": dataset,
            "domain": domain,
            "task_id": str(task_id),
            "source_path": source_path,
        })
        out.append(it)
    return out


def distill_multimodal_reasoning_items(tool_llm,
                                       prompts_dir: str,
                                       trajectory_obj: Dict,
                                       is_success: bool,
                                       dataset: str,
                                       domain: str,
                                       task_id: str,
                                       source_path: str,
                                       max_items: int = 2,
                                       use_visual_stage1: bool = True) -> List[Dict]:
    """
    Multimodal distillation: identify key steps, then extract insights with images.
    
    Stage 1: Key step identification (text-only OR with all screenshots)
    Stage 3: Multimodal extraction per key step with before/after screenshots
    
    Args:
        use_visual_stage1: If True, include all screenshots in Stage 1 for better causality detection.
                          If False, use text-only (faster, cheaper).
    """
    # Parse trajectory
    task, steps = parse_trajectory_rounds(trajectory_obj)
    if not steps:
        return []
    
    # Stage 1: Identify key steps
    if use_visual_stage1:
        print(f"[Stage 1] Using visual mode (with all screenshots)")
        key_indices = _identify_key_steps_visual(
            tool_llm, prompts_dir, task, steps, is_success
        )
    else:
        print(f"[Stage 1] Using text-only mode")
        key_indices = _identify_key_steps_text_only(
            tool_llm, prompts_dir, task, steps, is_success
        )
    
    if not key_indices:
        return []
    
    print(f"[Stage 1] Identified key steps: {key_indices}")
    
    # Stage 3: Extract key takeaways with multimodal evidence
    items = []
    for idx in key_indices[:max_items]:
        if idx < 0 or idx >= len(steps):
            continue
        
        item = _extract_key_takeaway_multimodal(
            tool_llm, prompts_dir, task, steps, idx, is_success
        )
        if not item:
            continue
        
        # Save images
        # Preferred semantics: pre = state at step idx, post = state at step idx+1
        if idx + 1 < len(steps):
            before_img = steps[idx]['screenshot']
            after_img = steps[idx + 1]['screenshot']
        else:
            # Fallback for last step: pre = idx-1 (if any), post = idx
            before_img = steps[idx - 1]['screenshot'] if idx > 0 else None
            after_img = steps[idx]['screenshot']
        
        media_dir = os.path.join(MEDIA_ROOT_DIR, "reasoning_bank", str(task_id))
        os.makedirs(media_dir, exist_ok=True)
        
        before_path = None
        after_path = None
        
        if before_img:
            before_path = os.path.join(media_dir, f"step_{idx-1:02d}.jpg")
            try:
                _save_image_from_base64(before_img, before_path)
            except Exception:
                before_path = None
        
        if after_img:
            after_path = os.path.join(media_dir, f"step_{idx:02d}.jpg")
            try:
                _save_image_from_base64(after_img, after_path)
            except Exception:
                after_path = None
        
        # Enrich with metadata
        item.update({
            "step_index": idx,
            "before_image_path": before_path,
            "after_image_path": after_path,
            "action": steps[idx]['action'],
            "label": "success" if is_success else "failure",
            "dataset": dataset,
            "domain": domain,
            "task_id": str(task_id),
            "source_path": source_path,
        })
        items.append(item)
    
    return items


def _identify_key_steps_text_only(tool_llm, prompts_dir: str, task: str, 
                                   steps: List[Dict], is_success: bool) -> List[int]:
    """Stage 1: Text-only key step identification."""
    # Build compact text trajectory
    traj_lines = []
    for i, step in enumerate(steps):
        action = step['action']
        response = step['response'][:400]  # truncate
        action_str = json.dumps(action) if action else "N/A"
        traj_lines.append(f"Step {i}: Action={action_str} Response={response}")
    
    trajectory_text = "\n".join(traj_lines)
    outcome = "success" if is_success else "failure"
    
    # Load prompt template
    prompt_path = os.path.join(prompts_dir, "reasoning_bank_mm_identify_steps.md")
    template = _load_prompt(prompt_path)
    
    # Fill template
    prompt_text = template.replace("{task}", task).replace("{outcome}", outcome).replace("{trajectory_text}", trajectory_text)
    
    messages = [{"role": "user", "content": prompt_text}]
    
    try:
        resp, _, _ = tool_llm.chat(messages=messages, stream=False)
        text = getattr(resp, "content", "") or str(resp)
        
        # Parse JSON output
        # Try to find JSON array
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            key_steps = json.loads(json_str)
            return [item['step_index'] for item in key_steps if 'step_index' in item]
    except Exception as e:
        print(f"Failed to identify key steps: {e}")
    
    return []


def _identify_key_steps_visual(tool_llm, prompts_dir: str, task: str, 
                                steps: List[Dict], is_success: bool) -> List[int]:
    """Stage 1 (Visual): Multimodal key step identification with all screenshots."""
    outcome = "success" if is_success else "failure"
    
    # Load prompt template
    prompt_path = os.path.join(prompts_dir, "reasoning_bank_mm_identify_steps_visual.md")
    if not os.path.exists(prompt_path):
        print(f"Visual prompt not found, falling back to text-only")
        return _identify_key_steps_text_only(tool_llm, prompts_dir, task, steps, is_success)
    
    template = _load_prompt(prompt_path)
    
    # Build multimodal trajectory
    content_items = []
    
    # Add task and outcome
    header = template.replace("{task}", task).replace("{outcome}", outcome).replace("{trajectory_with_images}", "")
    # Remove the placeholder line
    header = header.split("**Trajectory with screenshots:**")[0] + "**Trajectory with screenshots:**\n"
    content_items.append({"type": "text", "text": header})
    
    # Add each step with screenshot
    for i, step in enumerate(steps):
        action = step['action']
        response = step['response'][:300]  # truncate for brevity
        action_str = json.dumps(action) if action else "N/A"
        
        # Add step text
        step_text = f"\n**Step {i}:**\nAction: {action_str}\nResponse: {response}\n"
        content_items.append({"type": "text", "text": step_text})
        
        # Add screenshot if available
        screenshot = step.get('screenshot')
        if screenshot:
            content_items.append({
                "type": "image_url",
                "image_url": {"url": screenshot}
            })
    
    # Add closing instruction
    closing = f"\nNow analyze the trajectory and identify the 1-2 most critical steps that caused the {outcome}."
    content_items.append({"type": "text", "text": closing})
    
    messages = [{"role": "user", "content": content_items}]
    
    try:
        resp, _, _ = tool_llm.chat(messages=messages, stream=False)
        text = getattr(resp, "content", "") or str(resp)
        
        print(f"[Stage 1 Visual] VLM response: {text[:200]}...")
        
        # Parse JSON output
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            key_steps = json.loads(json_str)
            return [item['step_index'] for item in key_steps if 'step_index' in item]
    except Exception as e:
        print(f"Failed to identify key steps (visual): {e}")
    
    return []


def _extract_key_takeaway_multimodal(tool_llm, prompts_dir: str, task: str,
                                     steps: List[Dict], step_idx: int, 
                                     is_success: bool) -> Optional[Dict]:
    """Stage 3: Multimodal extraction for one key step."""
    if step_idx >= len(steps):
        return None
    
    # Build previous context (1-2 steps before)
    prev_context_lines = []
    for i in range(max(0, step_idx - 2), step_idx):
        action = steps[i]['action']
        action_str = json.dumps(action) if action else "N/A"
        prev_context_lines.append(f"Step {i}: {action_str}")
    prev_context = "\n".join(prev_context_lines) if prev_context_lines else "N/A"
    
    # Current step
    current_step = steps[step_idx]
    action = current_step['action']
    action_str = json.dumps(action) if action else "N/A"
    response = current_step['response'][:400]
    
    # Images
    # Preferred semantics: pre = state at step idx, post = state at step idx+1
    if step_idx + 1 < len(steps):
        before_img = steps[step_idx]['screenshot']
        after_img = steps[step_idx + 1]['screenshot']
    else:
        # Fallback for last step: pre = idx-1 (if any), post = idx
        before_img = steps[step_idx - 1]['screenshot'] if step_idx > 0 else None
        after_img = steps[step_idx]['screenshot']
    
    if not after_img:
        return None
    
    # Load prompt template
    prompt_path = os.path.join(prompts_dir, "reasoning_bank_mm_extract.md")
    template = _load_prompt(prompt_path)
    
    outcome = "success" if is_success else "failure"
    
    # Fill template
    prompt_text = (template
                   .replace("{task}", task)
                   .replace("{outcome}", outcome)
                   .replace("{prev_context}", prev_context)
                   .replace("{step_index}", str(step_idx))
                   .replace("{action}", action_str)
                   .replace("{response}", response))
    
    # Build multimodal message
    content = [{"type": "text", "text": prompt_text}]
    
    if before_img:
        content.append({"type": "image_url", "image_url": {"url": before_img}})
    
    content.append({"type": "image_url", "image_url": {"url": after_img}})
    
    messages = [{"role": "user", "content": content}]
    
    try:
        resp, _, _ = tool_llm.chat(messages=messages, stream=False)
        text = getattr(resp, "content", "") or str(resp)
        
        # Parse JSON output
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            result = json.loads(json_str)
            return result
    except Exception as e:
        print(f"Failed to extract key takeaway for step {step_idx}: {e}")
    
    return None


