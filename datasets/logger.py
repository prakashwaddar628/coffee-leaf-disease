"""
Logging configuration for Dataset Infrastructure.
Sets up professional logging to console (via rich) and files.
"""

import logging
from pathlib import Path
from rich.logging import RichHandler
try:
    from datasets.config import LOGS_DIR
except ImportError:
    # Fallback if run directly from datasets folder
    from config import LOGS_DIR

def setup_logger(name: str, log_file: str) -> logging.Logger:
    """
    Sets up a logger with a file handler and a rich console handler.

    Args:
        name (str): The name of the logger.
        log_file (str): The filename for the log output.

    Returns:
        logging.Logger: Configured logger instance.
    """
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / log_file

        logger = logging.getLogger(name)
        
        # Avoid adding multiple handlers if logger is already configured
        if logger.hasHandlers():
            return logger
            
        logger.setLevel(logging.INFO)

        # File Handler
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)

        # Console Handler using Rich
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False
        )
        console_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger
    except Exception as e:
        print(f"Failed to setup logger {name}: {e}")
        # Fallback to a basic logger
        fallback_logger = logging.getLogger(name)
        fallback_logger.setLevel(logging.INFO)
        return fallback_logger

# Pre-configured loggers for different modules
download_logger = setup_logger("downloader", "download.log")
dataset_logger = setup_logger("dataset", "dataset.log")