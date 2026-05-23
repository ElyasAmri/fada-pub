"""Comprehensive seed setting for reproducible experiments."""
import os
import random

import numpy as np


def set_all_seeds(seed: int = 42):
    """Set seeds for all RNGs to ensure reproducibility.

    Sets seeds for: random, numpy, os hash seed, and optionally
    PyTorch (CPU + CUDA) with deterministic backends.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
