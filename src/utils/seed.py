#!/usr/bin/env python3
"""Reproducibility helpers."""
from __future__ import annotations
import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all relevant RNG seeds for reproducibility.

    Note: fully deterministic CUDA ops would also require torch.use_deterministic_algorithms(True),
    which can break certain ops. We do not enable it by default; pass deterministic=True
    if you need bit-exact reproducibility (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
