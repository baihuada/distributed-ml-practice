"""
CSV 日志模块。

本文件负责：
1. 缓存每个 epoch 的训练日志；
2. 保存为 CSV 文件；
3. 统一 Single、PS、DDP 的日志字段；
4. 为后续实验汇总和画图提供标准化输入。

注意：
Single baseline 的 comm_mb 可以设置为 0.0。
PS 和 DDP 阶段会根据通信量估算公式填写 comm_mb。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


DEFAULT_LOG_FIELDS = [
    "system",
    "dataset",
    "model",
    "num_workers",
    "epoch",
    "train_loss",
    "train_acc",
    "test_loss",
    "test_acc",
    "epoch_time",
    "elapsed_time",
    "samples_per_sec",
    "num_train_samples",
    "num_test_samples",
    "model_params",
    "model_size_mb",
    "comm_mb",
    "seed",
    "lr",
    "batch_size",
]


class CSVLogger:
    """
    CSV 实验日志记录器。

    用法示例：
    logger = CSVLogger(log_path="results/raw/single_mnist_mlp_seed42.csv")
    logger.log({"epoch": 1, "train_loss": 0.5, "train_acc": 0.9})
    logger.save()
    """

    def __init__(
        self,
        log_path: str | Path,
        fieldnames: Optional[Iterable[str]] = None,
        auto_save: bool = False,
    ) -> None:
        """
        初始化日志记录器。

        参数
        ----
        log_path : str | Path
            CSV 文件保存路径。
        fieldnames : Optional[Iterable[str]]
            日志字段名。如果为 None，则使用 DEFAULT_LOG_FIELDS。
        auto_save : bool
            如果为 True，每次 log 后立即写入 CSV。
            第一版建议 False，训练结束后统一 save 即可。
        """

        self.log_path = Path(log_path)
        self.fieldnames = list(fieldnames) if fieldnames is not None else list(DEFAULT_LOG_FIELDS)
        self.auto_save = auto_save
        self.records: List[Dict[str, Any]] = []

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        """
        添加一条 epoch 日志。

        参数
        ----
        record : Dict[str, Any]
            当前 epoch 的日志字典。
        """

        normalized_record = self._normalize_record(record)
        self.records.append(normalized_record)

        if self.auto_save:
            self.save()

    def save(self) -> None:
        """
        将所有日志保存为 CSV 文件。
        """

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        with self.log_path.open(mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

    def to_dataframe(self) -> pd.DataFrame:
        """
        将当前日志转换为 pandas DataFrame。

        返回
        ----
        pd.DataFrame
            日志表格。
        """

        return pd.DataFrame(self.records, columns=self.fieldnames)

    def latest(self) -> Dict[str, Any]:
        """
        返回最近一条日志。

        返回
        ----
        Dict[str, Any]
            最近一个 epoch 的日志。
        """

        if not self.records:
            return {}

        return self.records[-1]

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        将输入日志补齐为统一字段。

        参数
        ----
        record : Dict[str, Any]
            原始日志。

        返回
        ----
        Dict[str, Any]
            字段补齐后的日志。
        """

        normalized = {}

        for field in self.fieldnames:
            normalized[field] = record.get(field, "")

        return normalized


def build_log_filename(
    system: str,
    dataset: str,
    model: str,
    seed: int,
    num_workers: int = 1,
    suffix: str = "",
) -> str:
    """
    构造标准日志文件名。

    参数
    ----
    system : str
        系统名称，例如 single、ps、ddp。
    dataset : str
        数据集名称。
    model : str
        模型名称。
    seed : int
        随机种子。
    num_workers : int
        worker 数量。Single baseline 默认 1。
    suffix : str
        可选后缀，用于区分额外实验。

    返回
    ----
    str
        CSV 文件名。
    """

    parts = [
        system.lower(),
        dataset.lower(),
        model.lower(),
        f"workers{num_workers}",
        f"seed{seed}",
    ]

    if suffix:
        parts.append(suffix)

    return "_".join(parts) + ".csv"


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    """
    保存实验配置为 JSON 文件。

    参数
    ----
    config : Dict[str, Any]
        实验配置字典。
    path : str | Path
        保存路径。
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(mode="w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_csv_log(path: str | Path) -> pd.DataFrame:
    """
    读取 CSV 日志文件。

    参数
    ----
    path : str | Path
        CSV 文件路径。

    返回
    ----
    pd.DataFrame
        读取后的日志表格。
    """

    return pd.read_csv(path)


def print_epoch_log(record: Dict[str, Any]) -> None:
    """
    按统一格式打印单个 epoch 的日志。

    参数
    ----
    record : Dict[str, Any]
        当前 epoch 的日志字典。
    """

    epoch = record.get("epoch", "")
    train_loss = float(record.get("train_loss", 0.0))
    train_acc = float(record.get("train_acc", 0.0))
    test_loss = float(record.get("test_loss", 0.0))
    test_acc = float(record.get("test_acc", 0.0))
    epoch_time = float(record.get("epoch_time", 0.0))
    samples_per_sec = float(record.get("samples_per_sec", 0.0))
    comm_mb = float(record.get("comm_mb", 0.0))

    print(
        f"Epoch {epoch} | "
        f"train_loss={train_loss:.4f}, train_acc={train_acc * 100:.2f}% | "
        f"test_loss={test_loss:.4f}, test_acc={test_acc * 100:.2f}% | "
        f"epoch_time={epoch_time:.2f}s | "
        f"samples/s={samples_per_sec:.2f} | "
        f"comm_mb={comm_mb:.4f}"
    )