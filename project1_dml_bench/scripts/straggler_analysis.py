"""
straggler_analysis.py

功能：
1. 读取 Sync-SGD / Async-SGD 在普通 delay 和 straggler delay 下的 CSV；
2. 汇总 final test accuracy、virtual time、communication rounds、communication MB、staleness；
3. 绘制 straggler 对比图；
4. 输出 results/tables/straggler_summary.csv 与 results/figures/*.png。

运行示例：
python scripts/straggler_analysis.py --model mlp --epochs 10 --batch-size 64 --lr 0.01 --num-workers 4 --seed 42
"""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt


def build_csv_path(
    result_dir: Path,
    algorithm: str,
    model: str,
    num_workers: int,
    delay_tag: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> Path:
    """
    根据当前 DML-Bench 的日志命名规则生成 CSV 路径。
    """

    filename = (
        f"{algorithm}_{model}"
        f"_workers{num_workers}"
        f"_delay{delay_tag}"
        f"_epochs{epochs}"
        f"_bs{batch_size}"
        f"_lr{lr}"
        f"_seed{seed}.csv"
    )
    return result_dir / filename


def load_experiment(csv_path: Path, label: str) -> pd.DataFrame:
    """
    读取单个实验 CSV，并添加 label 字段。
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["label"] = label
    df["source_csv"] = str(csv_path)
    return df


def summarize_final(df: pd.DataFrame, label: str) -> Dict[str, float]:
    """
    从一个实验日志中提取最后一个 epoch 的关键结果。
    """

    last = df.iloc[-1]

    return {
        "label": label,
        "algorithm": last.get("algorithm", "unknown"),
        "worker_delays": last.get("worker_delays", "unknown"),
        "final_epoch": int(last["epoch"]),
        "final_test_acc": float(last["test_acc"]),
        "final_train_acc": float(last["train_acc"]),
        "final_test_loss": float(last["test_loss"]),
        "comm_round": int(last["comm_round"]),
        "comm_mb": float(last["comm_mb"]),
        "virtual_time": float(last["virtual_time"]),
        "epoch_virtual_time": float(last.get("epoch_virtual_time", 0.0)),
        "avg_staleness": float(last.get("avg_staleness", 0.0)),
        "max_staleness": float(last.get("max_staleness", 0.0)),
    }


def plot_test_acc_vs_epoch(experiments: Dict[str, pd.DataFrame], save_path: Path) -> None:
    """
    绘制 test accuracy 随 epoch 变化的对比图。
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for label, df in experiments.items():
        plt.plot(df["epoch"], df["test_acc"], marker="o", linewidth=1.8, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Straggler Experiment: Test Accuracy vs Epoch")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_test_acc_vs_virtual_time(experiments: Dict[str, pd.DataFrame], save_path: Path) -> None:
    """
    绘制 test accuracy 随虚拟时间变化的对比图。
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for label, df in experiments.items():
        plt.plot(df["virtual_time"], df["test_acc"], marker="o", linewidth=1.8, label=label)

    plt.xlabel("Virtual Time")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Straggler Experiment: Test Accuracy vs Virtual Time")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_final_virtual_time(summary_df: pd.DataFrame, save_path: Path) -> None:
    """
    绘制最终虚拟时间柱状图。
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["label"], summary_df["virtual_time"])
    plt.xlabel("Experiment")
    plt.ylabel("Final Virtual Time")
    plt.title("Final Virtual Time under Straggler Settings")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_async_staleness(experiments: Dict[str, pd.DataFrame], save_path: Path) -> None:
    """
    绘制 Async-SGD 的 staleness 曲线。
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    has_async = False

    for label, df in experiments.items():
        algorithm = str(df.iloc[-1].get("algorithm", ""))
        if algorithm != "async_sgd":
            continue
        if "avg_staleness" not in df.columns:
            continue

        has_async = True
        plt.plot(df["epoch"], df["avg_staleness"], marker="o", linewidth=1.8, label=f"{label} avg")

        if "max_staleness" in df.columns:
            plt.plot(df["epoch"], df["max_staleness"], marker="s", linewidth=1.8, label=f"{label} max")

    if not has_async:
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel("Staleness")
    plt.title("Async-SGD Staleness under Straggler Settings")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DML-Bench straggler experiments.")

    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--equal-delay", type=str, default="1-1-1-1")
    parser.add_argument("--straggler-delay", type=str, default="1-1-1-5")
    parser.add_argument("--result-dir", type=str, default="results/raw")
    parser.add_argument("--table-dir", type=str, default="results/tables")
    parser.add_argument("--figure-dir", type=str, default="results/figures")

    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    table_dir = Path(args.table_dir)
    figure_dir = Path(args.figure_dir)

    specs = [
        ("sync_sgd", args.equal_delay, "Sync equal delay"),
        ("sync_sgd", args.straggler_delay, "Sync straggler"),
        ("async_sgd", args.equal_delay, "Async equal delay"),
        ("async_sgd", args.straggler_delay, "Async straggler"),
    ]

    experiments: Dict[str, pd.DataFrame] = {}
    summaries: List[Dict[str, float]] = []

    for algorithm, delay_tag, label in specs:
        csv_path = build_csv_path(
            result_dir=result_dir,
            algorithm=algorithm,
            model=args.model,
            num_workers=args.num_workers,
            delay_tag=delay_tag,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )

        df = load_experiment(csv_path, label)
        experiments[label] = df
        summaries.append(summarize_final(df, label))

    table_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    summary_path = table_dir / "straggler_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    plot_test_acc_vs_epoch(
        experiments=experiments,
        save_path=figure_dir / "straggler_test_acc_vs_epoch.png",
    )

    plot_test_acc_vs_virtual_time(
        experiments=experiments,
        save_path=figure_dir / "straggler_test_acc_vs_virtual_time.png",
    )

    plot_final_virtual_time(
        summary_df=summary_df,
        save_path=figure_dir / "straggler_final_virtual_time.png",
    )

    plot_async_staleness(
        experiments=experiments,
        save_path=figure_dir / "straggler_async_staleness.png",
    )

    print("=" * 80)
    print("Straggler summary saved.")
    print(f"summary table: {summary_path}")
    print(f"figures dir  : {figure_dir}")
    print("=" * 80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
