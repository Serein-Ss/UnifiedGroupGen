# Upstream device helpers extracted without training CLI/environment side effects.
from functools import lru_cache
from pathlib import Path
import torch
MODELS_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_ATOMIC_NUM = 100
@lru_cache
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@lru_cache
def get_pyg_device() -> torch.device:
    """
    Some operations of pyg don't work on MPS, so fall back to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

