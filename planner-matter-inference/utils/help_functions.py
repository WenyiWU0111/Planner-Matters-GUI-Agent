"""Helper functions for the GUI Agent"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List
import glob
import random

# Global variables for utility functions
datetime = None
LOG_FILE_NAME = None
logger = None

# Domain task numbers
MMINA_DICT = {
    'full': 1000,
    'shopping': 200,
    'wikipedia': 308,
    'normal': 176,
    'multi567': 180,
    'compare': 100,
    'multipro': 86
}


def set_global_variables(dt, log_file, log):
    """Set global variables for utility functions"""
    global datetime, LOG_FILE_NAME, logger
    datetime = dt
    LOG_FILE_NAME = log_file
    logger = log


def prepare(args):
    """Prepare the environment and settings"""
    # Set default values
    args.render = True
    args.render_screenshot = True
    args.save_trace_enabled = True
    args.current_viewport_only = False
    
    # Create result directory if it doesn't exist
    if args.result_dir:
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    
    # Dump configuration
    dump_config(args)


def is_domain_type(domain: str, domain_type: str) -> bool:
    """
    Check if a domain is of a specific type
    
    Args:
        domain: Domain name
        domain_type: Type to check for
        
    Returns:
        True if domain is of the specified type
    """
    domain_type_mapping = {
        'singlehop': ['shopping', 'wikipedia'],
        '2hop': ['normal', 'compare'],
        'multihop': ['multi567', 'multipro']
    }
    
    return domain in domain_type_mapping.get(domain_type, [])


def save_scores_to_json(scores: Dict[str, Any], result_dir: str):
    """
    Save scores to JSON file
    
    Args:
        scores: Scores dictionary
        result_dir: Result directory path
    """
    scores_file = Path(result_dir) / "scores.json"
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=4)
    
    logging.getLogger("logger").info(f"Scores saved to {scores_file}")


def dump_config(args):
    """
    Dump configuration to file
    
    Args:
        args: Arguments object
    """
    if args.result_dir:
        config_file = Path(args.result_dir) / "config.json"
        config_data = vars(args)
        
        # Remove non-serializable objects
        for key in ['loaded_model', 'loaded_tokenizer', 'grounding_model', 'grounding_tokenizer', 'lm_config']:
            if key in config_data:
                config_data[key] = str(type(config_data[key]))
        
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=4, default=str)
        
        logging.getLogger("logger").info(f"Configuration saved to {config_file}")


def get_unfinished(config_files: List[str], result_dir: str, config_dir: str=None) -> List[str]:
    """
    Get list of unfinished config files by checking existing result files.
    
    Args:
        config_files: List of all config file paths
        result_dir: Directory containing result files
        
    Returns:
        List of config files that haven't been processed yet
    """
    result_files = glob.glob(f"{result_dir}/*.html")
    

    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = json.load(f)
            # Handle both formats: MMInA uses 'task_id', WebVoyager uses 'id'
            task_id = str(config.get('task_id') or config.get('id', ''))
            if task_id not in task_ids:
                unfinished_configs.append(config_file)
    
    return unfinished_configs


def create_test_file_list_mmina(domain: str, start_idx: int = 0, end_idx: int = None) -> list:
    """
    Create a list of test files for a domain
    
    Args:
        domain: Domain name
        start_idx: Starting index
        end_idx: Ending index
        
    Returns:
        List of test file paths
    """
    if domain not in MMINA_DICT:
        raise ValueError(f"Invalid domain: {domain}")
    
    task_num = MMINA_DICT[domain]
    if end_idx is None:
        end_idx = task_num
    data_root = os.environ.get("MMINA_DATA_ROOT", "data/benchmarks/mmina")
    test_file_list = [f"{data_root}/{domain}/{i+1}.json" for i in range(start_idx, min(end_idx, task_num))]
    return test_file_list

def create_test_file_list_webvoyager(domain: str, start_idx: int = 0, end_idx: int = None) -> list:
    """
    Create a list of test files for a domain
    
    Args:
        domain: Domain name
        start_idx: Starting index
        end_idx: Ending index
        
    Returns:
        List of test file paths
    """
    data_root = os.environ.get("WEBVOYAGER_DATA_ROOT", "data/benchmarks/webvoyager")
    all_items = os.listdir(data_root) 
    all_domains = [item for item in all_items if os.path.isdir(os.path.join(data_root, item))]
    if domain not in all_domains and domain != 'all':
        raise ValueError(f"Invalid domain: {domain}, available domains: {all_domains}")
    
    if domain == 'all':
        all_test_files = []
        all_domains = ['Amazon', 'Allrecipes', 'Google_Map', 'Coursera']
        for domain in all_domains:
            all_test_files.extend(glob.glob(f"{data_root}/{domain}/*.json"))
        all_test_files = [item for item in all_test_files if '--' in item]
    else:
        all_test_files = glob.glob(f"{data_root}/{domain}/*.json")
    return all_test_files
