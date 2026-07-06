"""
summary_all_experiments.py

功能：
1. 汇总 DML-Bench 中所有已运行实验的 CSV 日志；
2. 自动读取 results/raw/*.csv；
3. 生成统一实验汇总表 results/tables/summary.csv；
4. 绘制算法准确率对比图、通信轮数对比图、通信量对比图、虚拟时间对比图；
5. 绘制 Local SGD 的 local_steps 对最终准确率影响图；
6. 绘制 straggler 场景下 Sync-SGD 与 Async-SGD 的虚拟时间对比图；
7. 绘制 Async-SGD 的 staleness 曲线图。

运行示例：
python scripts/summary_all_experiments.py

指定路径示例：
python scripts/summary_all_experiments.py --raw-dir results/raw --tables-dir results/tables --figures-dir results/figures
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: Path) -> None:
    """
    如果目录不存在，则自动创建。
    """
    path.mkdir(parents=True, exist_ok=True)


def read_csv_safely(path: Path) -> Optional[pd.DataFrame]:
    """
    安全读取 CSV 文件。

    若文件为空或格式损坏，则跳过，不中断整个汇总流程。
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"[Skip empty] {path}")
            return None

        df["source_file"] = path.name
        return df

    except Exception as exc:
        print(f"[Skip broken] {path}: {exc}")
        return None


def infer_algorithm_from_file(path: Path, df: pd.DataFrame) -> str:
    """
    从 CSV 内容或文件名推断算法名称。
    """
    if "algorithm" in df.columns and pd.notna(df["algorithm"].iloc[-1]):
        return str(df["algorithm"].iloc[-1])

    name = path.name.lower()

    if "centralized" in name:
        return "centralized_sgd"
    if "sync_sgd" in name:
        return "sync_sgd"
    if "local_sgd" in name:
        return "local_sgd"
    if "async_sgd" in name:
        return "async_sgd"

    return "unknown"


def get_value(row: pd.Series, key: str, default=None):
    """
    从一行记录中读取字段；若不存在，则返回默认值。
    """
    if key in row.index and pd.notna(row[key]):
        return row[key]
    return default


def build_label(algorithm: str, row: pd.Series, file_name: str) -> str:
    """
    为图例和汇总表生成可读标签。
    """
    if algorithm == "centralized_sgd":
        return "Centralized SGD"

    if algorithm == "sync_sgd":
        delay = get_value(row, "worker_delays", None)
        if delay is not None:
            return f"Sync-SGD delay={delay}"
        return "Sync-SGD"

    if algorithm == "local_sgd":
        local_steps = get_value(row, "local_steps", None)
        if local_steps is not None:
            return f"Local-SGD E={int(local_steps)}"
        return "Local-SGD"

    if algorithm == "async_sgd":
        delay = get_value(row, "worker_delays", None)
        if delay is not None:
            return f"Async-SGD delay={delay}"
        return "Async-SGD"

    return file_name.replace(".csv", "")


def load_histories(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    读取 results/raw 目录中的全部实验 CSV。
    """
    histories: Dict[str, pd.DataFrame] = {}

    csv_paths = sorted(raw_dir.glob("*.csv"))

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    for path in csv_paths:
        df = read_csv_safely(path)

        if df is None:
            continue

        algorithm = infer_algorithm_from_file(path, df)
        final_row = df.iloc[-1]
        label = build_label(algorithm, final_row, path.name)

        # 防止同名实验覆盖。
        if label in histories:
            label = f"{label} ({path.stem})"

        histories[label] = df

    if not histories:
        raise RuntimeError(f"No valid CSV files loaded from {raw_dir}")

    return histories


def make_summary_table(histories: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    根据每个实验的最后一行生成最终结果汇总表。
    """
    records: List[Dict] = []

    for label, df in histories.items():
        final_row = df.iloc[-1]
        source_file = str(get_value(final_row, "source_file", ""))
        algorithm = infer_algorithm_from_file(Path(source_file), df)

        record = {
            "label": label,
            "algorithm": algorithm,
            "source_file": source_file,
            "final_epoch": get_value(final_row, "epoch", None),
            "final_train_loss": get_value(final_row, "train_loss", None),
            "final_train_acc": get_value(final_row, "train_acc", None),
            "final_test_loss": get_value(final_row, "test_loss", None),
            "final_test_acc": get_value(final_row, "test_acc", None),
            "comm_round": get_value(final_row, "comm_round", None),
            "comm_mb": get_value(final_row, "comm_mb", None),
            "virtual_time": get_value(final_row, "virtual_time", None),
            "epoch_virtual_time": get_value(final_row, "epoch_virtual_time", None),
            "num_workers": get_value(final_row, "num_workers", None),
            "batch_size": get_value(
                final_row,
                "batch_size",
                get_value(final_row, "batch_size_per_worker", None),
            ),
            "local_steps": get_value(final_row, "local_steps", None),
            "worker_delays": get_value(final_row, "worker_delays", None),
            "avg_staleness": get_value(final_row, "avg_staleness", None),
            "max_staleness": get_value(final_row, "max_staleness", None),
            "server_version": get_value(final_row, "server_version", None),
        }

        records.append(record)

    summary_df = pd.DataFrame(records)

    order = {
        "centralized_sgd": 0,
        "sync_sgd": 1,
        "local_sgd": 2,
        "async_sgd": 3,
        "unknown": 9,
    }

    summary_df["_order"] = summary_df["algorithm"].map(order).fillna(9)
    summary_df = summary_df.sort_values(["_order", "label"]).drop(columns=["_order"])

    return summary_df


def has_columns(df: pd.DataFrame, cols: List[str]) -> bool:
    """
    判断 DataFrame 是否包含指定列。
    """
    return all(col in df.columns for col in cols)


def plot_curves(
    histories: Dict[str, pd.DataFrame],
    x_col: str,
    y_col: str,
    save_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """
    绘制多实验曲线图。
    """
    plt.figure(figsize=(8, 5))

    plotted = 0

    for label, df in histories.items():
        if not has_columns(df, [x_col, y_col]):
            continue

        plt.plot(
            df[x_col],
            df[y_col],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=label,
        )
        plotted += 1

    if plotted == 0:
        print(f"[Skip figure] Missing columns for {save_path.name}: {x_col}, {y_col}")
        plt.close()
        return

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Figure saved] {save_path}")


def plot_final_bar(
    summary_df: pd.DataFrame,
    value_col: str,
    save_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """
    绘制最终指标柱状图。
    """
    if value_col not in summary_df.columns:
        print(f"[Skip figure] Missing column {value_col}")
        return

    df = summary_df.dropna(subset=[value_col]).copy()

    if df.empty:
        print(f"[Skip figure] No valid data for {value_col}")
        return

    plt.figure(figsize=(9, 5))
    plt.bar(df["label"], df[value_col])
    plt.title(title)
    plt.xlabel("Experiment")
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Figure saved] {save_path}")


def plot_local_steps(summary_df: pd.DataFrame, save_path: Path) -> None:
    """
    绘制 Local SGD 中 local_steps 与最终准确率的关系。
    """
    if "algorithm" not in summary_df.columns or "local_steps" not in summary_df.columns:
        return

    df = summary_df[
        (summary_df["algorithm"] == "local_sgd")
        & summary_df["local_steps"].notna()
        & summary_df["final_test_acc"].notna()
    ].copy()

    if df.empty:
        print("[Skip figure] No Local-SGD records with local_steps.")
        return

    df["local_steps"] = df["local_steps"].astype(int)
    df = df.sort_values("local_steps")

    plt.figure(figsize=(7, 5))
    plt.plot(df["local_steps"], df["final_test_acc"], marker="o", linewidth=1.8)
    plt.title("Local SGD: Final Test Accuracy vs Local Steps")
    plt.xlabel("Local steps E")
    plt.ylabel("Final test accuracy (%)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Figure saved] {save_path}")


def plot_straggler_virtual_time(summary_df: pd.DataFrame, save_path: Path) -> None:
    """
    绘制 straggler 场景下的最终虚拟时间。
    """
    if "worker_delays" not in summary_df.columns:
        return

    df = summary_df[
        summary_df["algorithm"].isin(["sync_sgd", "async_sgd"])
        & summary_df["worker_delays"].notna()
        & summary_df["virtual_time"].notna()
    ].copy()

    if df.empty:
        print("[Skip figure] No straggler records.")
        return

    plt.figure(figsize=(8, 5))
    plt.bar(df["label"], df["virtual_time"])
    plt.title("Straggler Effect: Final Virtual Time")
    plt.xlabel("Experiment")
    plt.ylabel("Virtual time")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Figure saved] {save_path}")


def plot_async_staleness(histories: Dict[str, pd.DataFrame], save_path: Path) -> None:
    """
    绘制 Async-SGD 的平均 staleness 和最大 staleness 曲线。
    """
    plt.figure(figsize=(8, 5))

    plotted = 0

    for label, df in histories.items():
        if "async" not in label.lower():
            continue

        if not has_columns(df, ["epoch", "avg_staleness", "max_staleness"]):
            continue

        plt.plot(
            df["epoch"],
            df["avg_staleness"],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=f"{label} avg",
        )

        plt.plot(
            df["epoch"],
            df["max_staleness"],
            marker="x",
            linewidth=1.5,
            markersize=3,
            label=f"{label} max",
        )

        plotted += 1

    if plotted == 0:
        print("[Skip figure] No Async-SGD staleness records.")
        plt.close()
        return

    plt.title("Async-SGD Staleness")
    plt.xlabel("Epoch")
    plt.ylabel("Staleness")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Figure saved] {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize all DML-Bench experiment results.")

    parser.add_argument("--raw-dir", type=str, default="../results/raw")
    parser.add_argument("--tables-dir", type=str, default="results/tables")
    parser.add_argument("--figures-dir", type=str, default="results/figures")

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)

    ensure_dir(tables_dir)
    ensure_dir(figures_dir)

    histories = load_histories(raw_dir)
    summary_df = make_summary_table(histories)

    summary_path = tables_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("DML-Bench summary saved.")
    print(f"summary table: {summary_path}")
    print(f"figures dir  : {figures_dir}")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    plot_curves(
        histories=histories,
        x_col="epoch",
        y_col="test_acc",
        save_path=figures_dir / "algorithm_test_acc_vs_epoch.png",
        title="Test Accuracy vs Epoch",
        xlabel="Epoch",
        ylabel="Test accuracy (%)",
    )

    plot_curves(
        histories=histories,
        x_col="comm_round",
        y_col="test_acc",
        save_path=figures_dir / "algorithm_test_acc_vs_comm_round.png",
        title="Test Accuracy vs Communication Rounds",
        xlabel="Communication rounds",
        ylabel="Test accuracy (%)",
    )

    plot_curves(
        histories=histories,
        x_col="comm_mb",
        y_col="test_acc",
        save_path=figures_dir / "algorithm_test_acc_vs_comm_mb.png",
        title="Test Accuracy vs Communication Cost",
        xlabel="Communication cost (MB)",
        ylabel="Test accuracy (%)",
    )

    plot_curves(
        histories=histories,
        x_col="virtual_time",
        y_col="test_acc",
        save_path=figures_dir / "algorithm_test_acc_vs_virtual_time.png",
        title="Test Accuracy vs Virtual Time",
        xlabel="Virtual time",
        ylabel="Test accuracy (%)",
    )

    plot_final_bar(
        summary_df=summary_df,
        value_col="final_test_acc",
        save_path=figures_dir / "final_test_accuracy.png",
        title="Final Test Accuracy",
        ylabel="Final test accuracy (%)",
    )

    plot_final_bar(
        summary_df=summary_df,
        value_col="comm_mb",
        save_path=figures_dir / "final_communication_cost.png",
        title="Final Communication Cost",
        ylabel="Communication cost (MB)",
    )

    plot_local_steps(
        summary_df=summary_df,
        save_path=figures_dir / "local_steps_final_acc.png",
    )

    plot_straggler_virtual_time(
        summary_df=summary_df,
        save_path=figures_dir / "straggler_final_virtual_time_overall.png",
    )

    plot_async_staleness(
        histories=histories,
        save_path=figures_dir / "async_staleness_curve.png",
    )


if __name__ == "__main__":
    main()