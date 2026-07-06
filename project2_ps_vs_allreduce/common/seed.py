"""
Random seed utilities.

This module provides a unified way to control randomness for:
1. Python random;
2. NumPy;
3. PyTorch CPU;
4. PyTorch CUDA;
5. PyTorch DataLoader workers.

Reproducibility is important because Single, PS, and DDP results
will be compared later under the same dataset, model, learning rate,
batch size, and random seed.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seed for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed used by Python, NumPy, and PyTorch.
    deterministic : bool, default=False
        If True, enable deterministic CuDNN behavior.
        This improves reproducibility but may reduce training speed.

    Notes
    -----
    In this project, deterministic=False is recommended for normal runs.
    For debugging or result verification, deterministic=True can be used.
    """

    if seed < 0:
        raise ValueError(f"seed must be non-negative, but got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """
    Set seed for each PyTorch DataLoader worker.

    Parameters
    ----------
    worker_id : int
        Worker id automatically passed by PyTorch DataLoader.

    Notes
    -----
    PyTorch DataLoader may launch multiple subprocesses for data loading.
    This function makes each DataLoader worker use a reproducible seed.
    """

    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)
    random.seed(worker_seed)