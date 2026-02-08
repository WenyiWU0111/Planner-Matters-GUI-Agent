#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import List, Dict
import sys

# Ensure project root is on sys.path so 'agent' and sibling packages are importable
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from agent.llm_config import load_tool_llm
from config.argument_parser import config as load_config
from memory.reasoning_bank import (
    distill_reasoning_items, 
    distill_multimodal_reasoning_items,
    ReasoningBank
)


def load_trajectory(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compact_trajectory(rounds: List[Dict], max_steps: int = 20) -> str:
    """
    Make a short text trajectory from 'rounds': take the assistant 'response' lines
    (function-call JSON + brief reasoning) to keep size small.
    """
    lines = []
    count = 0
    for rd in rounds:
        resp = rd.get("response", "")
        if resp:
            # keep first 400 chars per step
            lines.append(resp.strip()[:400])
            count += 1
            if count >= max_steps:
                break
    return "\n---\n".join(lines)


def merge_trajectories(obj_a: Dict, obj_b: Dict) -> Dict:
    """
    Merge two trajectory objects by concatenating their rounds.
    Prefer a non-empty task_description from obj_b, otherwise obj_a.
    """
    task_a = obj_a.get("task_description", "")
    task_b = obj_b.get("task_description", "")
    merged = {
        "task_description": task_b or task_a,
        "rounds": list(obj_a.get("rounds", [])) + list(obj_b.get("rounds", [])),
    }
    return merged

def infer_label(path: str, obj: Dict) -> bool:
    # Prefer explicit evaluation flag if present
    ev = obj.get("evaluation", {}).get("evaluation", {})
    if "Correctness" in ev:
        return bool(ev["Correctness"])
    # fallback from path
    if "success" in path:
        return True
    if "fail" in path or "failed" in path or "negative" in path:
        return False
    # default success
    return True


def get_task_id_from_path(p: str) -> str:
    base = os.path.basename(p)
    if base.endswith(".jsonl"):
        return base[:-6]
    if base.endswith(".json"):
        return base[:-5]
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True, help="Glob for trajectory jsonl (actually json) files")
    ap.add_argument("--dataset", default="webvoyager")
    ap.add_argument("--domain", default="Amazon")
    ap.add_argument("--bank_jsonl", default="memory/reasoning_bank.jsonl")
    default_prompts_dir = os.path.join(_project_root, "agent", "prompts")
    ap.add_argument("--prompts_dir", default=default_prompts_dir)
    ap.add_argument("--max_items_per_traj", type=int, default=3)
    ap.add_argument("--multimodal", action="store_true", 
                    help="Use multimodal distillation with key step identification and screenshots")
    ap.add_argument("--visual_stage1", action="store_true",
                    help="Include all screenshots in Stage 1 for better causality detection (more aggressive)")
    args = ap.parse_args()

    # Load LLM for distillation using existing config machinery
    # Temporarily sanitize argv to prevent the global config parser from seeing our CLI flags
    _argv_backup = sys.argv
    try:
        sys.argv = [_argv_backup[0]]
        parsed_args = load_config()
    finally:
        sys.argv = _argv_backup
    tool_llm = load_tool_llm(parsed_args, model_name=getattr(parsed_args, 'tool_model_name', 'qwen2.5-vl'))
    prompts_dir = args.prompts_dir
    if not os.path.isabs(prompts_dir):
        prompts_dir = os.path.join(_project_root, prompts_dir)
    
    # Adjust paths for multimodal mode
    bank_path = args.bank_jsonl
    if args.multimodal and bank_path == "memory/reasoning_bank.jsonl":
        bank_path = "memory/reasoning_bank_mm.jsonl"
    
    index_base = "memory_index/reasoning_bank_text"
    if args.multimodal:
        index_base = "memory_index/reasoning_bank_mm"
    
    bank = ReasoningBank(bank_path=bank_path, index_base_path=index_base, use_multimodal=args.multimodal)

    paths = glob.glob(args.input_glob, recursive=True)
    # group by task_id; collect possible {success, positive, negative, failed}
    groups: Dict[str, Dict[str, List[str]]] = {}
    for p in paths:
        tid = get_task_id_from_path(p)
        bucket = groups.setdefault(tid, {"success": [], "positive": [], "negative": [], "failed": []})
        low = p.lower()
        if "success" in low:
            bucket["success"].append(p)
        elif "positive" in low:
            bucket["positive"].append(p)
        elif "negative" in low:
            bucket["negative"].append(p)
        elif "fail" in low or "failed" in low:
            bucket["failed"].append(p)
        else:
            # default route unknown → try success first
            bucket["success"].append(p)

    added = 0
    for task_id, parts in groups.items():
        try:
            # Success case
            if parts["success"]:
                p = parts["success"][0]
                obj = load_trajectory(p)
                
                if args.multimodal:
                    # Multimodal distillation
                    items = distill_multimodal_reasoning_items(
                        tool_llm,
                        prompts_dir=prompts_dir,
                        trajectory_obj=obj,
                        is_success=True,
                        dataset=args.dataset,
                        domain=args.domain,
                        task_id=str(task_id),
                        source_path=p,
                        max_items=args.max_items_per_traj,
                        use_visual_stage1=args.visual_stage1
                    )
                else:
                    # Legacy text-only distillation
                    query = obj.get("task_description", "")
                    rounds = obj.get("rounds", [])
                    traj_txt = compact_trajectory(rounds)
                    items = distill_reasoning_items(
                        tool_llm,
                        prompts_dir=prompts_dir,
                        is_success=True,
                        query=query,
                        trajectory_text=traj_txt,
                        dataset=args.dataset,
                        domain=args.domain,
                        task_id=str(task_id),
                        source_path=p,
                        max_items=args.max_items_per_traj
                    )
                    # portion metadata
                    for it in items:
                        it["portion"] = "full"
                
                if items:
                    bank.add_items(items, persist=True, update_index=True)
                    added += len(items)
                continue

            # Failure case: merge positive then negative if possible
            pos_paths = parts["positive"]
            neg_paths = parts["negative"] or parts["failed"]
            if pos_paths or neg_paths:
                if args.multimodal:
                    # Multimodal: merge positive + negative when both exist, else use whichever is available
                    merged_obj = None
                    source_path = None
                    if pos_paths and neg_paths:
                        ppos = pos_paths[0]
                        pneg = neg_paths[0]
                        objp = load_trajectory(ppos)
                        objn = load_trajectory(pneg)
                        merged_obj = merge_trajectories(objp, objn)
                        source_path = pneg
                    elif neg_paths:
                        pneg = neg_paths[0]
                        merged_obj = load_trajectory(pneg)
                        source_path = pneg
                    else:
                        ppos = pos_paths[0]
                        merged_obj = load_trajectory(ppos)
                        source_path = ppos
                    
                    items_neg = distill_multimodal_reasoning_items(
                        tool_llm,
                        prompts_dir=prompts_dir,
                        trajectory_obj=merged_obj,
                        is_success=False,
                        dataset=args.dataset,
                        domain=args.domain,
                        task_id=str(task_id),
                        source_path=source_path,
                        max_items=args.max_items_per_traj,
                        use_visual_stage1=args.visual_stage1
                    )
                    if items_neg:
                        bank.add_items(items_neg, persist=True, update_index=True)
                        added += len(items_neg)
                else:
                    # Legacy text-only: combine positive + negative rounds if both present
                    combined_rounds: List[Dict] = []
                    query_text = ""
                    source_path = None
                    if pos_paths and neg_paths:
                        ppos = pos_paths[0]
                        pneg = neg_paths[0]
                        objp = load_trajectory(ppos)
                        objn = load_trajectory(pneg)
                        query_text = objn.get("task_description", "") or objp.get("task_description", "")
                        combined_rounds = list(objp.get("rounds", [])) + list(objn.get("rounds", []))
                        source_path = pneg
                    elif neg_paths:
                        pneg = neg_paths[0]
                        objn = load_trajectory(pneg)
                        query_text = objn.get("task_description", "")
                        combined_rounds = list(objn.get("rounds", []))
                        source_path = pneg
                    else:
                        ppos = pos_paths[0]
                        objp = load_trajectory(ppos)
                        query_text = objp.get("task_description", "")
                        combined_rounds = list(objp.get("rounds", []))
                        source_path = ppos
                    
                    combined_txt = compact_trajectory(combined_rounds)
                    items_combined = distill_reasoning_items(tool_llm,
                                                             prompts_dir=prompts_dir,
                                                             is_success=False,
                                                             query=query_text,
                                                             trajectory_text=combined_txt,
                                                             dataset=args.dataset,
                                                             domain=args.domain,
                                                             task_id=str(task_id),
                                                             source_path=source_path,
                                                             max_items=args.max_items_per_traj)
                    for it in items_combined:
                        it["portion"] = "combined"
                    if items_combined:
                        bank.add_items(items_combined, persist=True, update_index=True)
                        added += len(items_combined)
                continue
        except Exception as e:
            print(f"Skip group {task_id}: {e}")
            continue

    print(f"Added {added} reasoning items into {bank_path}")


if __name__ == "__main__":
    main()


