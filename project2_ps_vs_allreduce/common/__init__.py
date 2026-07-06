"""
Common utilities for PS vs AllReduce project.

This package contains shared modules used by:
1. single-process baseline;
2. multiprocessing Parameter Server;
3. PyTorch DDP / AllReduce training.
"""

from .seed import set_seed, seed_worker

__all__ = [
    "set_seed",
    "seed_worker",
]