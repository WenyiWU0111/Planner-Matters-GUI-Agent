"""Test runner for the GUI Agent"""
import argparse
import json
import logging
import os
import re
from typing import Optional

from browser_env import (
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from .evaluator import evaluator_router
from utils.early_stop import early_stop
from agent.llm_config import create_model
from memory.plan_with_memory import generate_plan_with_memory, update_plan_with_memory
from memory.experience_memory_planner import ExperienceMemorySimple


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
        
        # Initialize memory for planning if planner is enabled
        if getattr(self.args, 'use_planner_with_memory', False):
            summary_json_path = getattr(self.args, 'memory_summary_path',
                os.environ.get("DISCRETE_SUMMARY_PATH", "discrete_summary.json"))
            faiss_index_path = getattr(self.args, 'memory_index_path',
                os.environ.get("FAISS_INDEX_PATH", "memory_index/simple_text_350"))
            self.logger.info(f"Initializing memory for planning from {faiss_index_path}")
            self.memory = ExperienceMemorySimple(summary_json_path, faiss_index_path)
            if getattr(self.args, 'checkpoint_planner', None):
                self.planner = create_model(checkpoint_path=self.args.checkpoint_planner)
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
        """Process a single config file"""
        parts = config_file.replace("MMInA/", "").split("/")
        sub_domain = parts[1] if len(parts) > 1 else ""

        render_helper = RenderHelper(config_file, self.args.result_dir)

        with open(config_file, encoding="utf-8") as f:
            _c = json.load(f)
        intent = _c.get("intent", "")
        if "library.kiwix.org" in intent and "wikipedia" in intent.lower():
            intent = intent.replace(
                "https://library.kiwix.org/iewer#wikipedia_en_all_maxi_2024-01/A/User%3AThe_other_Kiwix_guy/Landing",
                "https://www.wikipedia.org/",
            )
        task_id = _c.get("task_id", _c.get("id", ""))
        site = _c["sites"][0] if _c.get("sites") else ""

        numbers = re.findall(r'\d+', config_file)
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
        
        meta_data = {"action_history": [],
                     "response_history": []}
        
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
                    temperature=planner_temperature,
                    max_tokens=planner_max_tokens,
                    screenshot=screenshot or "",
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
            except Exception as e:
                self.logger.error(f"[Planner] Failed to generate plan: {e}")
                import traceback
                traceback.print_exc()
        else:
            self.logger.info("[planner] No planner with memory is enabled")
        
        # Start conversation for this task if training data collection is enabled
        if hasattr(self.agent, 'training_collector') and self.agent.training_collector:
            from utils.training_data_collector import get_collector
            collector = get_collector()
            if collector and collector.enabled:
                # Create conversation ID from task info
                conversation_id = f"{sub_domain}_{config_file.split('/')[-1].split('.')[0]}"
                collector.start_conversation(
                    conversation_id=conversation_id,
                    task_description=intent
                )
                self.logger.info(f"Started conversation collection for task: {conversation_id}")
        
        intent_list = [intent]

        # Process the task (single intent)
        for sub_query_idx, current_intent in enumerate(intent_list):
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
                    
                    action, meta_data = gen_action(enhanced_intent, meta_data)
                    
                if isinstance(action, list):
                    trajectory.extend(action)
                else:
                    trajectory.append(action)
                
                action_str = get_action_description(action)
                render_helper.render(
                    action, state_info, meta_data, self.args.render_screenshot
                )
                meta_data["action_history"].append(action_str)
                meta_data["page"] = self.env.page
                

                if isinstance(action, list):
                    last_action_type = action[-1]["action_type"]
                else:
                    last_action_type = action["action_type"]
                if last_action_type in [ActionTypes.STOP, 'finished']:
                    self.logger.info("Task completed")
                    break
                
                obs, _, terminated, _, info, current_url = self.env.step(action, observation=obs)
                # observation, 0.0, done, truncated, info
                print("CURRENT: ", current_url)

                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)
                
                if getattr(self.args, 'use_planner_with_memory', False) and self.memory and meta_data.get('task_plan'):
                    try:
                        planner_server_url = getattr(self.args, 'planner_server_url', 'http://localhost:8000/v1')
                        planner_model = getattr(self.args, 'planner_model', 'Qwen/Qwen2.5-VL-7B-Instruct')
                        planner_temperature = getattr(self.args, 'planner_temperature', 0.7)
                        planner_max_tokens = getattr(self.args, 'planner_max_tokens', 200)
                        planner_api_key = os.environ.get('OPEN_ROUTER_API_KEY', 'EMPTY')
                        
                        updated_plan, updated_memory_steps_text = update_plan_with_memory(
                            plan=meta_data['task_plan'].get('plan', ''),
                            query=meta_data['task_plan'].get('intent', intent),
                            memory_steps_text=meta_data.get('memory_steps_text', ''),
                            action_history=meta_data['action_history'],
                            trajectory=trajectory,
                            tool_llm=self.agent.tool_llm,
                            server_url=planner_server_url,
                            model=planner_model,
                            planner_model=self.planner,
                            temperature=planner_temperature,
                            max_tokens=planner_max_tokens,
                            previous_fail_reasons=previous_fail_reasons,
                            api_key=planner_api_key,
                        )
                        meta_data['task_plan'].update({'plan': updated_plan})
                        meta_data['memory_steps_text'] = updated_memory_steps_text
                    except Exception as e:
                        self.logger.error(f"[Planner] Failed to update plan: {e}")
                        import traceback
                        traceback.print_exc()

                if terminated:
                    trajectory.append(create_stop_action(""))
                    self.logger.info("Task terminated")
                    break
                
        # evaluate the scores
        evaluator = evaluator_router(config_file, self.agent.tool_llm)
        score, answer_text = evaluator(
            trajectory=trajectory,
            config_file=config_file,
            page=self.env.page,
            client=self.env.get_page_client(self.env.page),
        )
        
        last_action = trajectory[-1]
        pred = last_action.get("answer", "")
        reasoning = last_action.get("reasoning", "")
        score = 0.0 if 'Early stop' in pred else score
        self.logger.info(f"[Result] Predicted answer: {pred}\nReasoning: {reasoning}")
        result = "PASS" if score == 1 else "FAIL"
        self.logger.info(f"[Result] ({result}) {config_file}")
        # Save fail reasons if task failed
        if score != 1:
            fail_reasons_path = os.path.join(self.args.result_dir, 'fail_reasons.json')
            self._save_fail_reason(fail_reasons_path, task_id, answer_text)
        self.agent.experience_memory = None
        self.agent.experience_texts, self.agent.experience_images = None, None
        
        # Log only essential agent results (no token/timing details)
        self.logger.info(f"[Result] {config_file} - Success: {score}")

        render_helper.close(score=score, answer_text=answer_text)

        # End conversation for this task if training data collection is enabled
        if hasattr(self.agent, 'training_collector') and self.agent.training_collector:
            from utils.training_data_collector import get_collector
            collector = get_collector()
            if collector and collector.enabled and collector.current_conversation_id:
                # Create conversation summary
                conversation_summary = {
                    "task_id": config_file.split('/')[-1].split('.')[0],
                    "site": site,
                    "sub_domain": sub_domain,
                    "success": score,
                    "final_url": current_url,
                    "task_completed": True,
                    "task_description": intent
                }
                
                # End the conversation
                if self.args.collect_training_data:
                    saved_file = collector.end_conversation(conversation_summary, score)
                    if saved_file:
                        self.logger.info(f"Conversation saved: {saved_file}")
    