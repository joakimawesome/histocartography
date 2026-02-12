import os
import sys
import random
import json
import subprocess
import numpy as np
import torch
from typing import Dict, Any

def set_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def capture_environment() -> Dict[str, str]:
    """
    Captures the current environment information including git commit hash 
    and pip installed packages.

    Returns:
        Dict[str, str]: Dictionary containing 'git_commit' and 'pip_freeze'.
    """
    env_info = {}
    
    # Capture Git Commit Hash
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        env_info['git_commit'] = commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        env_info['git_commit'] = "unknown (not a git repo or git not installed)"

    # Capture Pip Freeze
    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze'], 
            stderr=subprocess.DEVNULL
        ).decode('ascii')
        env_info['pip_freeze'] = pip_freeze
    except (subprocess.CalledProcessError, FileNotFoundError):
        env_info['pip_freeze'] = "unknown (pip not found)"
        
    return env_info

def save_metadata(output_dir: str, config: Dict[str, Any], env_info: Dict[str, str] = None) -> None:
    """
    Saves configuration and environment metadata to a JSON file.

    Args:
        output_dir (str): Directory to save the metadata.json file.
        config (Dict[str, Any]): The configuration dictionary used for the run.
        env_info (Dict[str, str], optional): Environment info from capture_environment. 
                                             If None, it will be captured.
    """
    if env_info is None:
        env_info = capture_environment()
        
    metadata = {
        "config": config,
        "environment": env_info
    }
    
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Metadata saved to {metadata_path}")
