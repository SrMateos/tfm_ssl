from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config/defaults.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"Config file {config_path} not found, using defaults")
        return get_default_config()

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}, using defaults")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Return default configuration"""
    return {
        "debug": False,
        "model": {
            "architecture": "SwinUNETR",
            "feature_size": 48,
            "in_channels": 1,
            "out_channels": 1
        },
        "training": {
            "epochs": 200,
            "batch_size": 4,
            "learning_rate": 1e-4
        },
        "data": {
            "patch_size": [64, 64, 64],
            "train_val_split": 0.8,
            "task1": True
        },
        "inference": {
            "sw_batch_size": 1,
            "overlap": 0.5,
            "mode": "gaussian"
        }
    }

def save_config(config: Dict[str, Any], config_path: str = "config/current.yaml"):
    """Save configuration to YAML file"""
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
