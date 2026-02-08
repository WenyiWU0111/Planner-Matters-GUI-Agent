"""Logging setup for the GUI Agent."""
import argparse
import logging
from pathlib import Path


def setup_logging(args: argparse.Namespace):
    """Setup logging configuration and return logging components"""
    # Write logs into the result directory for this run
    result_dir = getattr(args, "result_dir", "")
    if result_dir:
        log_folder = Path(result_dir)
    else:
        log_folder = Path("results") / args.evaluation_type / args.model / args.domain / args.datetime
    log_folder.mkdir(parents=True, exist_ok=True)
    datetime = args.datetime
    LOG_FILE_NAME = str(log_folder / f"log_{args.model}_{datetime}.log")
    
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_FILE_NAME)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # Set the log format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    return datetime, LOG_FILE_NAME, logger 