from pathlib import Path
import torch

def get_default_device() -> torch.device:
    """Get the default Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

def project_root_dir() -> Path:
    current_dir = Path(__file__).parent.parent.parent.parent
    return current_dir