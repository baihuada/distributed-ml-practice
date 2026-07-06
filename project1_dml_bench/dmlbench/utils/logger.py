"""
logger.py

功能：
1. 保存训练过程中的实验指标；
2. 将每个 epoch 的结果写入 CSV 文件；
3. 为后续算法对比和画图提供统一日志格式。
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd


class ExperimentLogger:
    """
    实验日志记录器。

    使用方式：
        logger = ExperimentLogger(save_path="results/raw/centralized_seed42.csv")
        logger.log({"epoch": 1, "train_loss": 0.5, "test_acc": 90.1})
        logger.save()

    说明：
        内部用 list[dict] 存储每一轮结果；
        save() 时统一保存为 CSV。
    """

    def __init__(self, save_path: str, auto_save: bool = True) -> None:
        """
        参数：
            save_path:
                CSV 保存路径。
            auto_save:
                每次 log 后是否自动保存。
                True 更安全，训练中断时也能保留已有结果；
                False 稍快，但需要训练结束后手动 save。
        """

        self.save_path = Path(save_path)
        self.auto_save = auto_save
        self.records: List[Dict[str, Any]] = []

        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        """
        记录一条实验结果。

        参数：
            record:
                一个字典，例如：
                {
                    "epoch": 1,
                    "train_loss": 0.5,
                    "train_acc": 85.0,
                    "test_loss": 0.4,
                    "test_acc": 88.0
                }
        """

        self.records.append(record)

        if self.auto_save:
            self.save()

    def save(self) -> None:
        """
        将当前所有记录保存到 CSV 文件。
        """

        df = pd.DataFrame(self.records)
        df.to_csv(self.save_path, index=False, encoding="utf-8-sig")

    def to_dataframe(self) -> pd.DataFrame:
        """
        将日志转成 pandas DataFrame。

        返回：
            pd.DataFrame
        """

        return pd.DataFrame(self.records)

    def latest(self) -> Optional[Dict[str, Any]]:
        """
        返回最新一条日志记录。
        如果当前没有记录，则返回 None。
        """

        if not self.records:
            return None
        return self.records[-1]


def summarize_final_result(csv_path: str) -> Dict[str, Any]:
    """
    读取一个实验 CSV，返回最后一轮结果。

    参数：
        csv_path:
            实验日志 CSV 路径。

    返回：
        最后一行记录，字典形式。
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(path)
    if len(df) == 0:
        raise ValueError(f"CSV file is empty: {csv_path}")

    return df.iloc[-1].to_dict()


if __name__ == "__main__":
    # 简单测试 logger 是否正常。
    logger = ExperimentLogger("results/raw/test_logger.csv")

    logger.log({
        "epoch": 1,
        "train_loss": 1.23,
        "train_acc": 50.0,
        "test_loss": 1.10,
        "test_acc": 55.0,
    })

    logger.log({
        "epoch": 2,
        "train_loss": 0.80,
        "train_acc": 70.0,
        "test_loss": 0.75,
        "test_acc": 72.0,
    })

    print(logger.to_dataframe())
    print("saved to results/raw/test_logger.csv")