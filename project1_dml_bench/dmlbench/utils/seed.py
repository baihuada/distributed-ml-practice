"""
seed.py

功能：
1. 固定 Python、NumPy、PyTorch 的随机种子；
2. 尽量保证同一环境下实验可复现；
3. 为后续 Centralized SGD、Sync-SGD、Local SGD、Async-SGD 提供统一随机控制。

注意：
PyTorch 官方说明中，完全可复现不能在所有平台、所有版本、CPU/GPU 间绝对保证。
本文件的目标是让同一机器、同一环境下的实验尽可能稳定。
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    固定随机种子。

    参数：
        seed:
            随机种子，默认 42。
        deterministic:
            是否开启 PyTorch 确定性设置。
            True 表示尽量使用确定性算法，结果更稳定；
            False 表示允许非确定性算法，可能速度更快。

    返回：
        None
    """

    # 1. 固定 Python 哈希种子。
    # 注意：严格来说，PYTHONHASHSEED 最好在 Python 程序启动前设置。
    # 这里设置主要用于提醒和部分场景下的稳定性控制。
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. 固定 Python 内置 random 模块。
    random.seed(seed)

    # 3. 固定 NumPy 随机数。
    np.random.seed(seed)

    # 4. 固定 PyTorch CPU 随机数。
    torch.manual_seed(seed)

    # 5. 固定 PyTorch GPU 随机数。
    # 如果当前机器没有 GPU，这两行不会造成问题。
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # 6. 尽量使用确定性算法。
        # 这样有利于实验复现，但可能降低训练速度。
        torch.backends.cudnn.deterministic = True

        # 7. 关闭 cudnn benchmark。
        # benchmark 会自动寻找最快算法，但可能导致不同运行之间结果略有差异。
        torch.backends.cudnn.benchmark = False

        # 8. 尽量启用确定性算法。
        # warn_only=True 表示遇到不支持确定性的操作时只警告，不直接报错。
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # 兼容较旧版本 PyTorch。
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """
    DataLoader 多进程 worker 的随机种子设置函数。

    参数：
        worker_id:
            DataLoader 自动传入的 worker 编号。

    作用：
        当 DataLoader 使用 num_workers > 0 时，
        每个 worker 进程都需要单独设置 NumPy 和 random 的随机种子。
    """

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_generator(seed: int = 42) -> torch.Generator:
    """
    构造 PyTorch 随机数生成器。

    参数：
        seed:
            随机种子。

    返回：
        torch.Generator 对象。

    作用：
        传给 DataLoader 的 generator 参数，
        使 shuffle=True 时的数据打乱顺序尽量可复现。
    """

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator