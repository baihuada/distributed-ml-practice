"""
实验图表绘制脚本。

本文件负责：
1. 读取 results/raw/*.csv；
2. 读取 results/tables/summary.csv；
3. 绘制 accuracy 曲线；
4. 绘制 epoch time 对比图；
5. 绘制 samples/s 对比图；
6. 绘制通信量对比图；
7. 绘制 worker 扩展性图。

运行示例：
python -m utils.plot --raw-dir results/raw --summary results/tables/summary.csv --fig-dir results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """

    parser = argparse.ArgumentParser(
        description="Plot experiment figures for PS vs AllReduce"
    )

    parser.add_argument(
        "--raw-dir",
        type=str,
        default="results/raw",
        help="原始 CSV 日志目录",
    )

    parser.add_argument(
        "--summary",
        type=str,
        default="results/tables/summary.csv",
        help="汇总表路径",
    )

    parser.add_argument(
        "--fig-dir",
        type=str,
        default="results/figures",
        help="图像输出目录",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="图片 dpi",
    )

    return parser.parse_args()


def system_label_from_df(df: pd.DataFrame) -> str:
    """
    根据单个实验日志构造系统展示标签。
    """

    system = str(df["system"].iloc[0]).lower()
    workers = int(df["num_workers"].iloc[0])

    if system == "single":
        return "Single"

    if system == "ps":
        return f"PS-{workers}"

    if system == "ddp":
        return f"DDP-{workers}"

    return f"{system}-{workers}"


def system_sort_key(label: str) -> tuple:
    """
    对系统标签排序。

    排序顺序：
    Single -> PS-2 -> PS-4 -> DDP-2 -> DDP-4
    """

    label_lower = label.lower()

    if label_lower == "single":
        return (0, 1)

    if label_lower.startswith("ps-"):
        workers = int(label_lower.split("-")[1])
        return (1, workers)

    if label_lower.startswith("ddp-"):
        workers = int(label_lower.split("-")[1])
        return (2, workers)

    return (99, 99)


def load_raw_logs(raw_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """
    读取所有原始 CSV 日志。

    参数
    ----
    raw_dir : str | Path
        原始 CSV 目录。

    返回
    ----
    Dict[str, pd.DataFrame]
        system_label 到日志 DataFrame 的映射。
    """

    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"原始日志目录不存在: {raw_dir}")

    csv_files = sorted(raw_dir.glob("*.csv"))

    logs: Dict[str, pd.DataFrame] = {}

    for csv_path in csv_files:
        if csv_path.name.lower() == "summary.csv":
            continue

        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"[警告] 跳过空文件: {csv_path}")
            continue

        required = ["system", "num_workers", "epoch", "test_acc"]
        missing = [col for col in required if col not in df.columns]

        if missing:
            print(f"[警告] 文件 {csv_path} 缺少字段 {missing}，跳过")
            continue

        df = df.sort_values("epoch").reset_index(drop=True)

        label = system_label_from_df(df)
        logs[label] = df

    if not logs:
        raise RuntimeError(f"没有在 {raw_dir} 找到有效实验日志")

    logs = dict(sorted(logs.items(), key=lambda item: system_sort_key(item[0])))

    return logs


def load_summary(summary_path: str | Path) -> pd.DataFrame:
    """
    读取 summary.csv。
    """

    summary_path = Path(summary_path)

    if not summary_path.exists():
        raise FileNotFoundError(
            f"汇总表不存在: {summary_path}。请先运行 python -m utils.summarize"
        )

    df = pd.read_csv(summary_path)

    if df.empty:
        raise RuntimeError(f"汇总表为空: {summary_path}")

    if "system_label" not in df.columns:
        raise ValueError("summary.csv 中缺少 system_label 字段")

    df = df.sort_values(
        by="system_label",
        key=lambda col: col.map(system_sort_key),
    ).reset_index(drop=True)

    return df


def setup_matplotlib() -> None:
    """
    设置 matplotlib 基础参数。

    不手动指定颜色，使用 matplotlib 默认颜色循环。
    """

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3


def save_figure(fig_path: Path, dpi: int) -> None:
    """
    保存当前图像。
    """

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_system_accuracy(
    logs: Dict[str, pd.DataFrame],
    fig_dir: str | Path,
    dpi: int,
) -> None:
    """
    绘制 test accuracy 曲线。
    """

    fig_dir = Path(fig_dir)

    plt.figure()

    for label, df in logs.items():
        plt.plot(
            df["epoch"],
            df["test_acc"] * 100,
            marker="o",
            label=label,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("System Accuracy Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure(fig_dir / "system_accuracy.png", dpi=dpi)


def plot_epoch_time_compare(
    summary_df: pd.DataFrame,
    fig_dir: str | Path,
    dpi: int,
) -> None:
    """
    绘制平均 epoch time 对比图。
    """

    fig_dir = Path(fig_dir)

    plt.figure()

    plt.bar(
        summary_df["system_label"],
        summary_df["mean_epoch_time"],
    )

    plt.xlabel("System")
    plt.ylabel("Mean Epoch Time (s)")
    plt.title("Epoch Time Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.3)

    save_figure(fig_dir / "epoch_time_compare.png", dpi=dpi)


def plot_samples_per_sec_compare(
    summary_df: pd.DataFrame,
    fig_dir: str | Path,
    dpi: int,
) -> None:
    """
    绘制平均 samples/s 对比图。
    """

    fig_dir = Path(fig_dir)

    plt.figure()

    plt.bar(
        summary_df["system_label"],
        summary_df["mean_samples_per_sec"],
    )

    plt.xlabel("System")
    plt.ylabel("Mean Samples/s")
    plt.title("Throughput Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.3)

    save_figure(fig_dir / "samples_per_sec_compare.png", dpi=dpi)


def plot_comm_mb_compare(
    summary_df: pd.DataFrame,
    fig_dir: str | Path,
    dpi: int,
) -> None:
    """
    绘制通信量对比图。
    """

    fig_dir = Path(fig_dir)

    plt.figure()

    plt.bar(
        summary_df["system_label"],
        summary_df["final_comm_mb"],
    )

    plt.xlabel("System")
    plt.ylabel("Estimated Communication per Epoch (MB)")
    plt.title("Communication Cost Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.3)

    save_figure(fig_dir / "comm_mb_compare.png", dpi=dpi)


def plot_scalability_workers(
    summary_df: pd.DataFrame,
    fig_dir: str | Path,
    dpi: int,
) -> None:
    """
    绘制 worker 数扩展性图。

    横轴：num_workers
    纵轴：mean_samples_per_sec
    曲线：PS 和 DDP
    """

    fig_dir = Path(fig_dir)

    filtered_df = summary_df[
        summary_df["system"].str.lower().isin(["ps", "ddp"])
    ].copy()

    if filtered_df.empty:
        print("[警告] 没有 PS/DDP 结果，跳过 scalability_workers.png")
        return

    plt.figure()

    for system_name in ["ps", "ddp"]:
        sub_df = filtered_df[
            filtered_df["system"].str.lower() == system_name
        ].copy()

        if sub_df.empty:
            continue

        sub_df = sub_df.sort_values("num_workers")

        plt.plot(
            sub_df["num_workers"],
            sub_df["mean_samples_per_sec"],
            marker="o",
            label=system_name.upper(),
        )

    plt.xlabel("Number of Workers")
    plt.ylabel("Mean Samples/s")
    plt.title("Scalability with Number of Workers")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure(fig_dir / "scalability_workers.png", dpi=dpi)


def plot_train_loss_curves(
    logs: Dict[str, pd.DataFrame],
    fig_dir: str | Path,
    dpi: int,
) -> None:
    """
    绘制训练 loss 曲线。
    """

    fig_dir = Path(fig_dir)

    plt.figure()

    for label, df in logs.items():
        if "train_loss" not in df.columns:
            continue

        plt.plot(
            df["epoch"],
            df["train_loss"],
            marker="o",
            label=label,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure(fig_dir / "train_loss_curves.png", dpi=dpi)


def plot_test_loss_curves(
    logs: Dict[str, pd.DataFrame],
    fig_dir: str | Path,
    dpi: int,
) -> None:
    """
    绘制测试 loss 曲线。
    """

    fig_dir = Path(fig_dir)

    plt.figure()

    for label, df in logs.items():
        if "test_loss" not in df.columns:
            continue

        plt.plot(
            df["epoch"],
            df["test_loss"],
            marker="o",
            label=label,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.title("Test Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure(fig_dir / "test_loss_curves.png", dpi=dpi)


def plot_all(
    raw_dir: str | Path,
    summary_path: str | Path,
    fig_dir: str | Path,
    dpi: int = 200,
) -> None:
    """
    绘制全部图表。
    """

    setup_matplotlib()

    logs = load_raw_logs(raw_dir)
    summary_df = load_summary(summary_path)

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_system_accuracy(logs, fig_dir, dpi)
    plot_epoch_time_compare(summary_df, fig_dir, dpi)
    plot_samples_per_sec_compare(summary_df, fig_dir, dpi)
    plot_comm_mb_compare(summary_df, fig_dir, dpi)
    plot_scalability_workers(summary_df, fig_dir, dpi)

    # 额外生成 loss 曲线，便于检查训练是否正常
    plot_train_loss_curves(logs, fig_dir, dpi)
    plot_test_loss_curves(logs, fig_dir, dpi)

    print("=" * 80)
    print("图表绘制完成")
    print("=" * 80)
    print(f"图像输出目录: {fig_dir}")
    print("已生成主要图表：")
    print("1. system_accuracy.png")
    print("2. epoch_time_compare.png")
    print("3. samples_per_sec_compare.png")
    print("4. comm_mb_compare.png")
    print("5. scalability_workers.png")
    print("6. train_loss_curves.png")
    print("7. test_loss_curves.png")
    print("=" * 80)


def main() -> None:
    """
    主函数。
    """

    args = parse_args()

    plot_all(
        raw_dir=args.raw_dir,
        summary_path=args.summary,
        fig_dir=args.fig_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()