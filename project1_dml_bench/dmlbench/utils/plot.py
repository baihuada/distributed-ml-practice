"""
plot.py

功能：
1. 读取实验 CSV 日志；
2. 绘制 loss / accuracy 曲线；
3. 绘制多个算法的对比曲线；
4. 绘制 worker 数据类别分布；
5. 为后续 DML-Bench 实验报告提供图表。
"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_parent_dir(save_path: Union[str, Path]) -> None:
    """
    确保图片保存路径的父目录存在。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)


def plot_metric_curve(
    csv_path: Union[str, Path],
    x_key: str,
    y_key: str,
    save_path: Union[str, Path],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    从单个 CSV 文件中绘制单条指标曲线。
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if x_key not in df.columns:
        raise KeyError(f"x_key '{x_key}' not found in CSV columns: {df.columns.tolist()}")
    if y_key not in df.columns:
        raise KeyError(f"y_key '{y_key}' not found in CSV columns: {df.columns.tolist()}")

    ensure_parent_dir(save_path)

    plt.figure(figsize=(7, 5))
    plt.plot(df[x_key], df[y_key], marker="o", linewidth=1.8)
    plt.xlabel(xlabel or x_key)
    plt.ylabel(ylabel or y_key)
    plt.title(title or f"{y_key} vs {x_key}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_loss_and_accuracy(
    csv_path: Union[str, Path],
    output_dir: Union[str, Path] = "results/figures",
    prefix: Optional[str] = None,
) -> Dict[str, str]:
    """
    为一个实验日志同时绘制：
    1. train_loss / test_loss 曲线；
    2. train_acc / test_acc 曲线。
    """

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["epoch", "train_loss", "test_loss", "train_acc", "test_acc"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in {csv_path}")

    if prefix is None:
        prefix = csv_path.stem

    loss_path = output_dir / f"{prefix}_loss.png"
    acc_path = output_dir / f"{prefix}_accuracy.png"

    # 绘制 loss 曲线
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", linewidth=1.8, label="Train loss")
    plt.plot(df["epoch"], df["test_loss"], marker="s", linewidth=1.8, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Test Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # 绘制 accuracy 曲线
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_acc"], marker="o", linewidth=1.8, label="Train accuracy")
    plt.plot(df["epoch"], df["test_acc"], marker="s", linewidth=1.8, label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train/Test Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    plt.close()

    return {
        "loss": str(loss_path),
        "accuracy": str(acc_path),
    }


def compare_metric_curves(
    csv_paths: Sequence[Union[str, Path]],
    labels: Sequence[str],
    x_key: str,
    y_key: str,
    save_path: Union[str, Path],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    对比多个实验的同一个指标。

    例如：
        Centralized SGD vs Sync-SGD 的 test_acc 曲线。
    """

    if len(csv_paths) != len(labels):
        raise ValueError("csv_paths and labels must have the same length.")

    ensure_parent_dir(save_path)

    plt.figure(figsize=(7, 5))

    for csv_path, label in zip(csv_paths, labels):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if x_key not in df.columns:
            raise KeyError(f"x_key '{x_key}' not found in {csv_path}")
        if y_key not in df.columns:
            raise KeyError(f"y_key '{y_key}' not found in {csv_path}")

        plt.plot(df[x_key], df[y_key], marker="o", linewidth=1.8, label=label)

    plt.xlabel(xlabel or x_key)
    plt.ylabel(ylabel or y_key)
    plt.title(title or f"{y_key} comparison")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_label_distribution(
    summary_df: pd.DataFrame,
    save_path: Union[str, Path],
    title: str = "Label distribution across workers",
) -> None:
    """
    绘制 worker 数据类别分布图。

    summary_df 需要包含：
        worker_id, label, count
    """

    required_cols = ["worker_id", "label", "count"]
    for col in required_cols:
        if col not in summary_df.columns:
            raise KeyError(f"summary_df must contain column '{col}'")

    ensure_parent_dir(save_path)

    pivot = summary_df.pivot(index="label", columns="worker_id", values="count").fillna(0)

    labels = pivot.index.to_numpy()
    worker_ids = pivot.columns.to_list()

    x = np.arange(len(labels))
    width = 0.8 / max(len(worker_ids), 1)

    plt.figure(figsize=(9, 5))

    for idx, worker_id in enumerate(worker_ids):
        offset = (idx - (len(worker_ids) - 1) / 2) * width
        plt.bar(x + offset, pivot[worker_id].to_numpy(), width=width, label=f"Worker {worker_id}")

    plt.xlabel("Label")
    plt.ylabel("Number of samples")
    plt.title(title)
    plt.xticks(x, labels)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot DML-Bench curves from CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to experiment CSV file.")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    saved_paths = plot_loss_and_accuracy(
        csv_path=args.csv,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )

    print("Figures saved:")
    for name, path in saved_paths.items():
        print(f"{name}: {path}")