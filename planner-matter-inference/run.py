"""Main script to run the GUI Agent."""
import glob
import os

from agent.agent_new import construct_agent
from agent.llm_config import load_grounding_model_vllm
from benchmarks.MMInA_evaluation.test_runner_new import TestRunner as MMInATestRunner
from benchmarks.webvoyager_evaluation.test_runner_new import TestRunner as WebVoyagerTestRunner
from config.argument_parser import config
from utils.help_functions import (
    create_test_file_list_mmina,
    create_test_file_list_webvoyager,
    get_unfinished,
    prepare,
    set_global_variables,
)
from utils.logging_setup import setup_logging

def main():
    """Main execution function"""
    args = config()
    args.sleep_after_execution = 2.5
    
    # Setup logging
    datetime, LOG_FILE_NAME, logger = setup_logging(args)
    set_global_variables(datetime, LOG_FILE_NAME, logger)
    
    # Prepare environment
    prepare(args)
    logger.info(f"Observation context length: {args.max_obs_length}")
    logger.info(f"Use discrete memory: {getattr(args, 'use_discrete_memory', False)}")
    logger.info(f"Use graph memory: {getattr(args, 'use_graph_memory', False)}")
    
    # Load model and agent
    # model, tokenizer = load_model(args)
    # args.loaded_model = model
    # args.loaded_tokenizer = tokenizer
    
    # Load grounding model using vLLM
    grounding_model = load_grounding_model_vllm(args)
    args.grounding_model = grounding_model
    
    agent = construct_agent(args)
    
    # Run evaluation based on type
    if args.evaluation_type == "mmina":
        # Default MMInA evaluation
        test_file_list = create_test_file_list_mmina(args.domain)
        if not args.debug:
            test_file_list = get_unfinished(test_file_list, args.result_dir, 'mmina')[:120]
        logger.info(f"Total {len(test_file_list)} tasks to process")
        run_tests_mmina(args, agent, test_file_list)
    elif args.evaluation_type == "webvoyager":
        if args.data_path is not None:
            test_file_list = glob.glob(os.path.join(args.data_path, "**", "*.json"), recursive=True)
        else:
            test_file_list = create_test_file_list_webvoyager(args.domain, args.test_start_idx, args.test_end_idx)
        if not args.debug:
            test_file_list = get_unfinished(test_file_list, args.result_dir, 'webvoyager_evaluation')
        test_file_list = test_file_list[:100]
        logger.info(f"Total {len(test_file_list)} tasks to process")
        run_tests_webvoyager(args, agent, test_file_list)
    elif args.evaluation_type == "mind2web":
        test_file_list = glob.glob(os.path.join(f"data/benchmarks/mind2web/{args.domain}", "**", "*.json"), recursive=True)
        # Sort by task_id to ensure consistent ordering (Info_1, Info_2, ..., Info_100)
        def extract_task_num(path):
            import re
            basename = os.path.basename(path)
            match = re.search(r'_(\d+)\.json$', basename)
            return int(match.group(1)) if match else 0
        test_file_list = sorted(test_file_list, key=extract_task_num)
        # Apply start/end index filtering
        test_file_list = test_file_list[args.test_start_idx:args.test_end_idx]
        if not args.debug:
            test_file_list = get_unfinished(test_file_list, args.result_dir, 'data/benchmarks/mind2web')
        logger.info(f"Total {len(test_file_list)} tasks to process (idx {args.test_start_idx} to {args.test_end_idx})")
        # Use the WebVoyager runner for Mind2Web tasks (shared pipeline).
        # Mind2Web configs are supported by the WebVoyager runner via `sites` -> `site` mapping.
        run_tests_webvoyager(args, agent, test_file_list)
    elif args.evaluation_type == "mind2web_executable":
        # Run on curated executable subset (tasks verified as accessible)
        mind2web_data_root = os.environ.get("MIND2WEB_DATA_ROOT", "data/benchmarks/mind2web_executable")
        test_file_list = glob.glob(os.path.join(mind2web_data_root, args.domain, "**", "*.json"), recursive=True)
        def extract_task_num(path):
            import re
            basename = os.path.basename(path)
            match = re.search(r'_(\d+)\.json$', basename)
            return int(match.group(1)) if match else 0
        test_file_list = sorted(test_file_list, key=extract_task_num)
        if not args.debug:
            test_file_list = get_unfinished(test_file_list, args.result_dir, mind2web_data_root)
        test_file_list = [item for item in test_file_list if not 'Shopping' in item]
        logger.info(f"[mind2web_executable] Total {len(test_file_list)} executable tasks to process")
        run_tests_webvoyager(args, agent, test_file_list)
    elif args.evaluation_type == "expand_memory":
        test_file_list = create_test_file_list_expand_memory(args.domain, args.test_start_idx, args.test_end_idx)
        if not args.debug:
            test_file_list = get_unfinished(test_file_list, args.result_dir)
        # print(test_file_list)
        logger.info(f"Total {len(test_file_list)} tasks to process")
        run_tests_webvoyager(args, agent, test_file_list)
    logger.info(f"Test finished. Log file: {LOG_FILE_NAME}")


def run_tests_mmina(args, agent, config_file_list):
    """Run the main test loop"""
    
    test_runner = MMInATestRunner(args, agent)
    test_runner.run(config_file_list)

def run_tests_webvoyager(args, agent, config_file_list):
    """Run the main test loop."""
    test_runner = WebVoyagerTestRunner(args, agent)
    test_runner.run(config_file_list)


if __name__ == "__main__":
    main()
