"""Test runner for the GUI Agent (WebVoyager) with planner/memory and fail-reasons."""
import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from browser_env import (
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.helper_functions import RenderHelper, get_action_description
from utils.early_stop import early_stop
from agent.llm_config import load_tool_llm, create_model
from memory.plan_with_memory import (
    generate_plan_with_memory,
    update_plan_with_memory,
    extract_history_context,
)
from memory.experience_memory_planner import ExperienceMemorySimple

from .evaluator import LLMEvaluator


class TestRunner:
    """Handles the main test execution loop"""
    
    def __init__(self, args: argparse.Namespace, agent):
        self.args = args
        self.agent = agent
        self.logger = logging.getLogger("logger")
        # Initialize environment
        self.env = ScriptBrowserEnv(
            headless=True,
            slow_mo=args.slow_mo,
            viewport_size={
                "width": args.viewport_width,
                "height": args.viewport_height,
            },
            save_trace_enabled=args.save_trace_enabled,
            sleep_after_execution=args.sleep_after_execution,
            args=args,  # Pass args to the environment
        )
        # IMPORTANT: do not hardcode a tool model here.
        # This model is used by the environment to judge whether an action changed the page
        # (image comparison) and by the WebVoyager evaluator. It must match whatever server
        # the user actually has running (e.g., qwen3-vl on :8007).
        self.evaluate_model = load_tool_llm(self.args)
        self.evaluator = LLMEvaluator(vllm_client=self.evaluate_model)
        
        # Initialize memory for planning if planner is enabled
        if getattr(self.args, 'use_planner_with_memory', False):
            summary_json_path = getattr(self.args, 'memory_summary_path',
                os.environ.get("DISCRETE_SUMMARY_PATH", "discrete_summary.json"))
            faiss_index_path = getattr(self.args, 'memory_index_path',
                os.environ.get("FAISS_INDEX_PATH", "memory_index/simple_text_350"))
            self.logger.info(f"Initializing memory for planning from {faiss_index_path}")
            self.memory = ExperienceMemorySimple(summary_json_path, faiss_index_path)
            if getattr(self.args, 'checkpoint_planner', None):
                self.planner = create_model(self.args)
            else:
                if 'rl' in getattr(self.args, 'model', '') or 'sft' in getattr(self.args, 'model', '') and not getattr(self.args, 'use_base_planner_model', False):
                    self.planner = self.agent.llm
                else:
                    self.planner = None
        else:
            print("[planner] No planner with memory is enabled")
            self.memory = None
        
    def run(self, config_file_list: list[str]):
        """Run the main test loop"""
        # Process each config file
        for config_file in config_file_list:
            self._process_config_file(config_file)
        # Close environment
        self.env.close()
    
    def _load_fail_reasons(self, fail_reasons_path: str, task_id: str) -> Optional[str]:
        """Load previous fail reasons for a given task_id."""
        if not os.path.exists(fail_reasons_path):
            return None
        try:
            with open(fail_reasons_path, "r", encoding="utf-8") as f:
                fail_reasons = json.load(f)
            return fail_reasons.get(task_id)
        except Exception as e:
            self.logger.warning(f"Failed to load fail reasons: {e}")
            return None

    def _save_fail_reason(self, fail_reasons_path: str, task_id: str, answer_text: str) -> None:
        """Save fail reason for a given task_id."""
        try:
            fail_reasons = {}
            if os.path.exists(fail_reasons_path):
                with open(fail_reasons_path, "r", encoding="utf-8") as f:
                    fail_reasons = json.load(f)
            fail_reasons[task_id] = answer_text
            fail_reasons_dir = os.path.dirname(fail_reasons_path)
            if fail_reasons_dir:
                os.makedirs(fail_reasons_dir, exist_ok=True)
            with open(fail_reasons_path, "w", encoding="utf-8") as f:
                json.dump(fail_reasons, f, indent=2)
            self.logger.info(f"[FailReasons] Saved fail reason for task_id: {task_id}")
        except Exception as e:
            self.logger.error(f"Failed to save fail reason: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_config_file(self, config_file: str) -> None:
        """Process a single config file."""
        render_helper = RenderHelper(config_file, self.args.result_dir)

        with open(config_file, encoding="utf-8") as f:
            _c = json.load(f)
        intent = _c.get("intent", "")
        task_id = _c.get("task_id", _c.get("id", ""))
        if "site" in _c:
            site = _c["site"]
        elif "sites" in _c:
            sites = _c["sites"]
            if not isinstance(sites, list):
                raise TypeError(f"Expected 'sites' to be a list, got {type(sites)}")
            if len(sites) != 1:
                raise ValueError(f"Expected 'sites' to have length 1, got {len(sites)}: {sites}")
            if not isinstance(sites[0], str):
                raise TypeError(f"Expected 'sites[0]' to be a str, got {type(sites[0])}")
            site = sites[0]
        else:
            raise KeyError("Config must contain either 'site' (webvoyager) or 'sites' (mind2web).")

        numbers = re.findall(r"\d+", config_file)
        self.args.task_cnt = int(numbers[0]) if numbers else None
        self.args.hop_cnt = 0
        
        self.logger.info(f"[Config file]: {config_file}")
        self.logger.info(f"[Intent]: {intent}")
        
        self.agent.reset(config_file)
        self.agent.current_step = 0
        trajectory: Trajectory = []
        
        # Environment reset
        obs, info = self.env.reset(
            options={"config_file": config_file}, 
        )
        current_url = info["page"].url
        state_info: StateInfo = {"observation": obs, "info": info, "current_url": current_url}
        trajectory.append(state_info)
        print("CURRENT: ", current_url)
        # if 'about:blank' in current_url or info["is_blocked"]:
        #     self.logger.info(f"[Result] (Cannot navigate to {_c['start_url']}) {config_file}")
        #     return
        terminated = False
        meta_data = {"action_history": [],
                     "action_results": [],  # Track action success/failure for history context
                     "response_history": [],
                     "clean_intent": intent}  # Store original intent before pollution
        
        print("config_file: ", config_file)
        
        # Load previous fail reasons if any
        fail_reasons_path = os.path.join(self.args.result_dir, 'fail_reasons.json')
        previous_fail_reasons = self._load_fail_reasons(fail_reasons_path, task_id)
        
        # Generate plan with memory if enabled
        task_plan = None
        if getattr(self.args, 'use_planner_with_memory', False) and self.memory:
            try:
                self.logger.info("[Planner] Generating plan with memory...")
                planner_server_url = getattr(self.args, 'planner_server_url', 'http://localhost:8000/v1')
                planner_model = getattr(self.args, 'planner_model', 'Qwen/Qwen2.5-VL-7B-Instruct') 
                planner_similar_num = getattr(self.args, 'planner_similar_num', 10)
                planner_temperature = getattr(self.args, 'planner_temperature', 0.7)
                planner_max_tokens = getattr(self.args, 'planner_max_tokens', 200)
                planner_api_key = os.environ.get('OPEN_ROUTER_API_KEY', 'EMPTY')
                
                # Get screenshot if available
                screenshot = obs.get('image') if isinstance(obs, dict) and 'image' in obs else None
                
                plan_text, memory_steps_text, file_id_list = generate_plan_with_memory(
                    query=intent,
                    memory=self.memory,
                    server_url=planner_server_url,
                    model=planner_model,
                    planner_model=self.planner,
                    similar_num=planner_similar_num,
                    use_continuous_memory=getattr(self.args, 'use_continuous_memory', False),
                    temperature=planner_temperature,
                    max_tokens=planner_max_tokens,
                    screenshot=screenshot,
                    previous_fail_reasons=previous_fail_reasons,
                    api_key=planner_api_key,
                )
                
                task_plan = {
                    'plan': plan_text,
                    'intent': intent,
                    'memory_steps_text': memory_steps_text,
                    'file_id_list': file_id_list,
                }
                meta_data['task_plan'] = task_plan
                meta_data['intent'] = task_plan['intent']
                meta_data['memory_steps_text'] = memory_steps_text
                meta_data['file_id_list'] = file_id_list
            except Exception as e:
                self.logger.error(f"[Planner] Failed to generate plan: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("[planner] No planner with memory is enabled")
        
        # Process single task
        current_intent = intent + "Once you find the result, please directly yield a stop action, and give a brief explanation in your answer!"
            
        # Process current task
        while True:
            current_url = current_url.lower()

            early_stop_flag, stop_info = early_stop(
                trajectory, self.args.max_steps, {
                    "parsing_failure": self.args.parsing_failure_th,
                    "repeating_action": self.args.repeating_action_failure_th,
                }
            )

            if early_stop_flag:
                action = create_stop_action(f"Early stop: {stop_info}")
            else:
                def gen_action(intent, meta):
                    # Pass the plan to the agent via meta_data if available
                    if task_plan and 'task_plan' not in meta:
                        meta['task_plan'] = task_plan
                        meta['intent'] = task_plan.get('intent', intent)
                    action, meta =  self.agent.next_action_custom(
                        trajectory,
                        intent,
                        meta_data=meta,
                    )
                    return action, meta
                
                action, meta_data = gen_action(current_intent, meta_data)
                    
            trajectory.append(action)
            
            action_str = get_action_description(action)
            
            try:
                render_helper.render(
                    action, state_info, meta_data, self.args.render_screenshot
                )
            except Exception as e:
                self.logger.error(f"Error rendering screenshot: {e}")
                pass
            meta_data["action_history"].append(action_str)
            meta_data["page"] = self.env.page

            if isinstance(action, list):
                last_action_type = action[-1]["action_type"]
            else:
                last_action_type = action["action_type"]
            if last_action_type in [ActionTypes.STOP, 'finished']:
                self.logger.info(f"[Task] Completed")
                break
            try:
                done = False
                terminated = False
                max_retries = 3
                for i in range(max_retries):
                    obs, reasoning, terminated, done, info, current_url = self.env.step(action, observation=obs, old_info=info, tool_llm=self.evaluate_model)
                    if done:
                        break
                    time.sleep(0.1)
                if not done:
                    meta_data['error_message'] = reasoning
                
                # Track action results for history context (if flag enabled)
                if getattr(self.args, 'use_history_context', False):
                    meta_data["action_results"].append({
                        "action": action_str,
                        "success": done,
                        "reasoning": reasoning if reasoning else "",
                        "step": len(meta_data["action_history"])
                    })
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.logger.error(f"Error in step: {e}")
                terminated = False
                done = False
                reasoning = ""
                if len(trajectory) >= 2:
                    last_state = trajectory[-2]
                    if isinstance(last_state, dict):
                        if "observation" in last_state:
                            obs = last_state["observation"]
                        if "info" in last_state:
                            info = last_state["info"]
                            if "page" in info and hasattr(info.get("page"), "url"):
                                current_url = info["page"].url
            # observation, 0.0, done, truncated, info
            print("CURRENT: ", current_url)

            state_info = {"observation": obs, "info": info, 'step_done': done}
            meet_login = self.agent.check_login(state_info)
            if meet_login:
                self.logger.info(f"[Task] Met login, reset to the starting page.")
                obs, info = self.env.reset(
                    options={"config_file": config_file}, 
                )
                current_url = info["page"].url
                state_info = {"observation": obs, "info": info, 'step_done': True}
            trajectory.append(state_info)
            
            if getattr(self.args, 'use_planner_with_memory', False) and self.memory and meta_data.get('task_plan'):
                try:
                    planner_server_url = getattr(self.args, 'planner_server_url', 'http://localhost:8000/v1')
                    planner_model = getattr(self.args, 'planner_model', 'Qwen/Qwen2.5-VL-7B-Instruct')
                    planner_temperature = getattr(self.args, 'planner_temperature', 0.7)
                    planner_max_tokens = getattr(self.args, 'planner_max_tokens', 200)
                    planner_api_key = os.environ.get('OPEN_ROUTER_API_KEY', 'EMPTY')
                    
                    # Extract history context if flag enabled
                    history_context = None
                    if getattr(self.args, 'use_history_context', False) and meta_data.get('action_results'):
                        history_context = extract_history_context(
                            action_results=meta_data['action_results'],
                            max_recent=10
                        )
                    
                    # Only pass tool_llm and memory if --use_adaptive_memory flag is enabled
                    use_adaptive = getattr(self.args, 'use_adaptive_memory', False)
                    adaptive_tool_llm = self.evaluate_model if use_adaptive else None
                    adaptive_memory = self.memory if use_adaptive else None
                    
                    updated_plan, updated_memory_steps_text, updated_file_id_list = update_plan_with_memory(
                        plan=meta_data['task_plan'].get('plan', ''),
                        query=meta_data['task_plan'].get('intent', intent),
                        memory_steps_text=meta_data.get('memory_steps_text', ''),
                        file_id_list=meta_data.get('file_id_list', []),
                        action_history=meta_data['action_history'],
                        trajectory=trajectory,
                        tool_llm=adaptive_tool_llm,
                        history_context=history_context,
                        memory=adaptive_memory,
                        server_url=planner_server_url,
                        model=planner_model,
                        planner_model=self.planner,
                        use_continuous_memory=getattr(self.args, 'use_continuous_memory', False),
                        temperature=planner_temperature,
                        max_tokens=planner_max_tokens,
                        previous_fail_reasons=previous_fail_reasons,
                        api_key=planner_api_key,
                    )
                    meta_data['task_plan'].update({'plan': updated_plan})
                    meta_data['memory_steps_text'] = updated_memory_steps_text
                    meta_data['task_plan']['memory_steps_text'] = updated_memory_steps_text
                    meta_data['file_id_list'] = updated_file_id_list
                except Exception as e:
                    self.logger.error(f"[Planner] Failed to update plan: {e}")
                    import traceback
                    traceback.print_exc()

            if terminated:
                # add a action place holder
                trajectory.append(create_stop_action(""))
                self.logger.info(f"[Task] Terminated")
                break
                
        try:
            score, answer_text, ori_answer = self.evaluator(config_file, self.args.result_dir)
        except Exception as e:
            self.logger.error(f"Error in evaluator: {e}")
            score, answer_text, ori_answer = 0.0, "Error in evaluator", "Error in evaluator"
        
        last_action = trajectory[-1]
        pred = last_action.get("answer", "")
        reasoning = last_action.get("reasoning", "")
        self.logger.info(f"[Result] Predicted answer: {pred}\nReasoning: {reasoning}")
        
        result = "PASS" if score==1 else "FAIL"
        self.logger.info(f"[Result] ({result}) {config_file}")
        self.logger.info(f"Evaluator Response: {answer_text}")
        
        # Save fail reasons if task failed
        if score != 1:
            fail_reasons_path = os.path.join(self.args.result_dir, 'fail_reasons.json')
            self._save_fail_reason(fail_reasons_path, task_id, answer_text)
        
        # Close render helper with evaluation results
        render_helper.close(score=score, answer_text=answer_text, ori_answer=ori_answer)
        