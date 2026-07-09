"""
Global Configuration module for Coffee Leaf Disease Research.
Loads settings from dataset.yaml and defines global paths.
"""

import yaml
from pathlib import Path
from typing import Any, Dict

# Define global paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

def load_config(config_path: Path = PROJECT_ROOT / "datasets" / "dataset.yaml") -> Dict[str, Any]:
    """
    Loads the dataset configuration from a YAML file.

    Args:
        config_path (Path): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Dictionary containing dataset configurations.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config if config else {}
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        return {}

# Global config instance
CONFIG = load_config()