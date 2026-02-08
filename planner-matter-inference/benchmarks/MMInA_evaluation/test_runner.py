"""Test runner for the GUI Agent"""
import argparse
import json
import logging
import re

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
from agent.llm_config import load_tool_llm


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
        
    def run(self, config_file_list: list[str]):
        """Run the main test loop"""
        
        # Process each config file
        for config_file in config_file_list:
            self._process_config_file(config_file)
        # Close environment
        self.env.close()
            
    def _process_config_file(self, config_file: str):
        """Process a single config file"""
        parts = config_file.replace("MMInA/", "").split("/")
        sub_domain = parts[1] if len(parts) > 1 else ""

        render_helper = RenderHelper(config_file, self.args.result_dir)

        with open(config_file, encoding="utf-8") as f:
            _c = json.load(f)
        intent = _c.get("intent", "")
        # Normalize legacy Kiwix Wikipedia URL to standard Wikipedia
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

                if terminated:
                    trajectory.append(create_stop_action(""))
                    self.logger.info("Task terminated")
                    break
                
        # Evaluate the scores
        evaluate_model = load_tool_llm(self.args)
        evaluator = evaluator_router(config_file, evaluate_model)
        score, answer_text = evaluator(
            trajectory=trajectory,
            config_file=config_file,
            page=self.env.page,
            client=self.env.get_page_client(self.env.page),
        )

        last_action = trajectory[-1]
        pred = last_action.get("answer", "")
        reasoning = last_action.get("reasoning", "")
        score = 0.0 if "Early stop" in (pred or "") else score
        self.logger.info(f"[Result] Predicted answer: {pred}\nReasoning: {reasoning}")
        result = "PASS" if score == 1 else "FAIL"
        self.logger.info(f"[Result] ({result}) {config_file}")
        
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
                
                # # End the conversation
                # if score == 1:
                if self.args.collect_training_data:
                    saved_file = collector.end_conversation(conversation_summary, score)
                    if saved_file:
                        self.logger.info(f"Conversation saved: {saved_file}")