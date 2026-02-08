"""Test runner for the GUI Agent (WebVoyager evaluation)."""
import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

from agent.llm_config import load_tool_llm
from browser_env import (
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.helper_functions import RenderHelper, get_action_description
from scripts.reasoning_bank.reasoning_bank import ReasoningBank, distill_multimodal_reasoning_items
from utils.early_stop import early_stop

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
        
    def run(self, config_file_list: list[str]):
        """Run the main test loop"""
        # Process each config file
        for config_file in config_file_list:
            self._process_config_file(config_file)
        # Close environment
        self.env.close()
    
    def _distill_workflow_memory(
        self, 
        config_file: str, 
        trajectory: Trajectory, 
        intent: str, 
        site: str, 
        task_id: str,
        score: float,
        meta_data: dict
    ):
        """Distill workflow memory from current successful task"""
        workflow_prompt_path = Path(f'workflow_memory/{site}.txt')
        workflow_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"[WorkflowMemory] Starting distillation for domain: {site}")
        
        # # Only distill from successful tasks
        # if score < 1.0:
        #     self.logger.info(f"[WorkflowMemory] Task {task_id} failed (score={score}), skipping workflow distillation")
        #     return
        
        # Extract screenshots from trajectory
        screenshots = []
        for obj in trajectory:
            if isinstance(obj, dict) and 'observation' in obj:
                obs = obj['observation']
                if isinstance(obs, dict) and 'image' in obs:
                    screenshots.append(obs['image'])
        
        if not screenshots:
            self.logger.warning(f"[WorkflowMemory] No screenshots found in trajectory for task {task_id}")
            return
        
        # Get response_history from meta_data
        response_history = meta_data.get('response_history', [])
    
        instruction = ""
        one_shot = ""

        instruction_path = "agent/prompts/awm_instruction.txt"
        one_shot_path = "agent/prompts/awm_one_shot.txt"
        
        with open(instruction_path, "r", encoding="utf-8") as f:
            instruction = f.read()
        with open(one_shot_path, "r", encoding="utf-8") as f:
            one_shot = f.read()
        
        # Format the task example with screenshots and response history
        response_text = "\n".join([str(r)[:500] for r in response_history]) if response_history else "No response history"
        
        # Build prompt with screenshots
        example_text = f"Query: {intent}\nResponse History:\n{response_text}"
        
        # Prepare messages with screenshots
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": [
                {"type": "text", "text": one_shot + "\n\n" + example_text}
            ]}
        ]
        
        # Add screenshots (last 5)
        image_contents = []
        for img in screenshots[-5:]:
            if isinstance(img, str):
                # If it's already base64, use it directly
                url = img if isinstance(img, str) and img.startswith("data:") else f"data:image/png;base64,{img}"
                image_contents.append({"type": "image_url", "image_url": {"url": url}})
            else:
                # If it's a PIL Image or numpy array, convert to base64
                import base64
                from io import BytesIO
                from PIL import Image
                if hasattr(img, 'save'):
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    })
        
        if image_contents:
            messages.append({'role': 'user', 'content': image_contents})
        
        # Generate workflow using LLM
        try:
            response, _, _ = self.evaluate_model.chat(
                messages=messages,
                temperature=1.0,
                max_tokens=2048
            )
            workflows = response.content.strip()
            # Save workflow
            with open(workflow_prompt_path, "w", encoding="utf-8") as f:
                f.write(workflows)
            
            self.logger.info(f"[WorkflowMemory] Workflow memory updated: {workflow_prompt_path}")
            self.logger.info(f"[WorkflowMemory] Workflow memory: {workflows}")
        except Exception as e:
            self.logger.error(f"[WorkflowMemory] Failed to generate workflow: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_config_file(self, config_file: str):
        """Process a single config file"""

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
                     "response_history": [],
                     "clean_intent": intent}  # Store original intent before pollution
        
        print("config_file: ", config_file)
        
        # Start conversation for this task if training data collection is enabled
        if hasattr(self.agent, 'training_collector') and self.agent.training_collector:
            collector = self.agent.training_collector
            if collector and collector.enabled:
                # Create conversation ID from task info
                conversation_id = f"{site}_{config_file.split('/')[-1].split('.')[0]}".replace(' ', '_')
                collector.start_conversation(
                    conversation_id=conversation_id,
                    task_description=intent
                )
                self.logger.info(f"Started conversation collection for task: {conversation_id}")
        
        intent_list = [intent]

        # Process the task (single intent)
        for sub_query_idx, current_intent in enumerate(intent_list):
            current_intent += " Once you find the result, please directly yield a stop action, and give a brief explanation in your answer!"
            enhanced_intent = current_intent

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
                        action, meta =  self.agent.next_action_custom(
                            trajectory,
                            intent,
                            meta_data=meta,
                        )
                        return action, meta
                    if 'error_message' in meta_data:
                        enhanced_intent += f" The error message from the previous action is: {meta_data['error_message']}, please try another action."
                    action, meta_data = gen_action(enhanced_intent, meta_data)
                        
                trajectory.append(action)
                
                action_str = get_action_description(action)
                # Ensure augmented_intent and verifier_feedback are in meta_data for rendering
                if 'augmented_intent' not in meta_data:
                    meta_data['augmented_intent'] = meta_data.get('task_plan', {}).get('augmented_intent', None)
                if 'verifier_feedback' not in meta_data:
                    meta_data['verifier_feedback'] = None
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
                    self.logger.info("Task completed")
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
                    self.logger.info("Met login, reset to the starting page.")
                    obs, info = self.env.reset(
                        options={"config_file": config_file}, 
                    )
                    current_url = info["page"].url
                    state_info = {"observation": obs, "info": info, 'step_done': True}
                trajectory.append(state_info)

                if terminated:
                    trajectory.append(create_stop_action(""))
                    self.logger.info("Task terminated")
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
        
        # self.metrics_dict[config_file] = {
        #         "config": config_file,
        #         "success": score,
        #     }
        # self.trajSuccess[config_file] = score

        result = "PASS" if score==1 else "FAIL"
        self.logger.info(f"[Result] ({result}) {config_file}")
        self.logger.info(f"Evaluator Response: {answer_text}")
        
        # Close render helper with evaluation results
        render_helper.close(score=score, answer_text=answer_text, ori_answer=ori_answer)
        
        # End-of-episode distillation into Reasoning Bank (optional)
        try:
            if getattr(self.args, 'use_reasoning_bank', False):
                is_success = bool(score == 1.0)
                
                # Get trajectory from training collector if available
                trajectory_obj = None
                if hasattr(self.agent, 'training_collector') and self.agent.training_collector:
                    collector = self.agent.training_collector
                    if collector and collector.enabled and hasattr(collector, 'conversation_history'):
                        # Build trajectory object from collector data
                        trajectory_obj = {
                            'task_description': intent,
                            'rounds': collector.conversation_history
                        }
                
                # Load bank and distill
                bank = ReasoningBank(
                    bank_path=getattr(self.args, 'reasoning_bank_path', 'memory/reasoning_bank.jsonl'),
                    index_base_path=getattr(self.args, 'reasoning_index_base', 'memory_index/reasoning_bank_text'),
                    use_multimodal=getattr(self.args, 'reasoning_bank_multimodal', False)
                )
                prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent", "prompts")
                if not os.path.exists(prompts_dir):
                    # fallback relative to project root
                    prompts_dir = "agent/prompts"
                
                items = []
                # Use multimodal distillation if trajectory object is available
                if trajectory_obj and getattr(self.args, 'reasoning_bank_multimodal', False):
                    self.logger.info("[ReasoningBank] Using multimodal distillation")
                    items = distill_multimodal_reasoning_items(
                        tool_llm=self.evaluate_model,
                        prompts_dir=prompts_dir,
                        trajectory_obj=trajectory_obj,
                        is_success=is_success,
                        dataset="webvoyager",
                        domain=site,
                        task_id=str(task_id),
                        source_path=config_file,
                        max_items=2,
                        use_visual_stage1=True
                    )
                else:
                    # Fallback to text-only distillation
                    self.logger.info("[ReasoningBank] Using text-only distillation (no trajectory object)")
                    response_history = meta_data.get('response_history', [])
                    if isinstance(response_history, list) and response_history:
                        parts = [str(resp)[:400] for resp in response_history[:30] if resp]
                        trajectory_text = "\n---\n".join(parts)
                    else:
                        trajectory_text = ""
                    
                    from memory.reasoning_bank import distill_reasoning_items
                    items = distill_reasoning_items(
                        tool_llm=self.evaluate_model,
                        prompts_dir=prompts_dir,
                        is_success=is_success,
                        query=intent,
                        trajectory_text=trajectory_text,
                        dataset="webvoyager",
                        domain=site,
                        task_id=str(task_id),
                        source_path=config_file,
                        max_items=3
                    )
                
                if items:
                    # Log distilled items summary before writing
                    try:
                        if 'key_takeaway' in items[0]:
                            takeaways = [it.get("key_takeaway", "")[:80] for it in items]
                            self.logger.info(f"[ReasoningBank] distilled {len(items)} multimodal items "
                                             f"(label={'success' if is_success else 'failure'}) "
                                             f"for task_id={task_id}: {takeaways}")
                        else:
                            titles = [it.get("title", "") for it in items]
                            self.logger.info(f"[ReasoningBank] distilled {len(items)} text items "
                                             f"(label={'success' if is_success else 'failure'}) "
                                             f"for task_id={task_id}: {titles}")
                    except Exception:
                        pass
                    bank.add_items(items, persist=True, update_index=True)
                    # Log persistence and index update details
                    try:
                        self.logger.info(f"[ReasoningBank] bank updated: path={bank.bank_path}, "
                                         f"index={bank.index_path}, total_items={len(bank.items)}")
                    except Exception:
                        pass
                else:
                    self.logger.info("[ReasoningBank] no items distilled for this episode")
        except Exception as e:
            self.logger.error(f"Reasoning bank distillation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Distill workflow memory if enabled
        if getattr(self.args, "use_awm", False):
            try:
                self._distill_workflow_memory(
                    config_file=config_file,
                    trajectory=trajectory,
                    intent=intent,
                    site=site,
                    task_id=task_id,
                    score=score,
                    meta_data=meta_data
                )
            except Exception as e:
                self.logger.error(f"Workflow memory distillation failed: {e}")
                import traceback
                traceback.print_exc()
            
        # End conversation for this task if training data collection is enabled
        if hasattr(self.agent, 'training_collector') and self.agent.training_collector:
            collector = self.agent.training_collector
            if collector and collector.enabled and collector.current_conversation_id:
                # Create conversation summary
                conversation_summary = {
                    "task_id": config_file.split('/')[-1].split('.')[0],
                    "site": site,
                    "sub_domain": '',
                    "success": score,
                    "final_url": current_url,
                    "task_completed": True,
                    "task_description": intent
                }
                
                # End the conversation
                if self.args.save_examples_memory:
                    saved_file = collector.end_conversation(conversation_summary, score)
                    if saved_file:
                        self.logger.info(f"Conversation saved: {saved_file}")
                    else:
                        self.logger.info("Conversation not saved")
                else:
                    self.logger.info("not save_examples_memory")
                    