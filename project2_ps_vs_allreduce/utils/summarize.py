"""
实验结果汇总脚本。

本文件负责：
1. 读取 results/raw/ 下的所有 CSV 日志；
2. 提取每个实验最后一个 epoch 的结果；
3. 计算平均 epoch time、平均 samples/s 等统计指标；
4. 生成 results/tables/summary.csv；
5. 为后续画图和报告撰写提供统一汇总表。

运行示例：
python -m utils.summarize --raw-dir results/raw --output results/tables/summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


REQUIRED_COLUMNS = [
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
    "model_params",
    "model_size_mb",
    "comm_mb",
    "seed",
]


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """

    parser = argparse.ArgumentParser(
        description="Summarize raw experiment CSV logs"
    )

    parser.add_argument(
        "--raw-dir",
        type=str,
        default="results/raw",
        help="原始 CSV 日志目录",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/tables/summary.csv",
        help="汇总表输出路径",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="是否启用严格检查。启用后，缺少必要字段会直接报错",
    )

    return parser.parse_args()


def find_csv_files(raw_dir: str | Path) -> List[Path]:
    """
    查找原始结果目录下的 CSV 文件。

    参数
    ----
    raw_dir : str | Path
        原始结果目录。

    返回
    ----
    List[Path]
        CSV 文件路径列表。
    """

    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"原始结果目录不存在: {raw_dir}")

    csv_files = sorted(raw_dir.glob("*.csv"))

    # 排除可能已经生成的 summary.csv，避免重复汇总
    csv_files = [
        path for path in csv_files
        if path.name.lower() != "summary.csv"
    ]

    return csv_files


def check_required_columns(
    df: pd.DataFrame,
    csv_path: Path,
    strict: bool = False,
) -> bool:
    """
    检查 CSV 是否包含必要字段。

    参数
    ----
    df : pd.DataFrame
        读取到的 CSV 表格。
    csv_path : Path
        当前 CSV 文件路径。
    strict : bool
        是否启用严格模式。

    返回
    ----
    bool
        字段是否满足要求。
    """

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if not missing_columns:
        return True

    message = (
        f"文件 {csv_path} 缺少必要字段: {missing_columns}"
    )

    if strict:
        raise ValueError(message)

    print(f"[警告] {message}，该文件将被跳过。")
    return False


def safe_get(
    row: pd.Series,
    key: str,
    default: Optional[float | str] = None,
):
    """
    安全读取 Series 中的字段。

    参数
    ----
    row : pd.Series
        一行数据。
    key : str
        字段名。
    default : Optional[float | str]
        缺失时的默认值。

    返回
    ----
    Any
        字段值或默认值。
    """

    if key not in row:
        return default

    value = row[key]

    if pd.isna(value):
        return default

    return value


def summarize_one_file(csv_path: Path, strict: bool = False) -> Optional[Dict]:
    """
    汇总单个 CSV 文件。

    参数
    ----
    csv_path : Path
        CSV 文件路径。
    strict : bool
        是否启用严格字段检查。

    返回
    ----
    Optional[Dict]
        单个实验的汇总结果。如果文件无效，则返回 None。
    """

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        if strict:
            raise
        print(f"[警告] 读取文件失败: {csv_path}, 原因: {exc}")
        return None

    if df.empty:
        if strict:
            raise ValueError(f"CSV 文件为空: {csv_path}")
        print(f"[警告] CSV 文件为空，跳过: {csv_path}")
        return None

    if not check_required_columns(df, csv_path, strict=strict):
        return None

    # 按 epoch 排序，确保最后一行确实是最终 epoch
    df = df.sort_values("epoch").reset_index(drop=True)

    first_row = df.iloc[0]
    final_row = df.iloc[-1]

    summary = {
        "system": str(safe_get(first_row, "system", "")),
        "dataset": str(safe_get(first_row, "dataset", "")),
        "model": str(safe_get(first_row, "model", "")),
        "num_workers": int(safe_get(first_row, "num_workers", 1)),
        "epochs": int(safe_get(final_row, "epoch", len(df))),
        "final_train_loss": float(safe_get(final_row, "train_loss", 0.0)),
        "final_train_acc": float(safe_get(final_row, "train_acc", 0.0)),
        "final_test_loss": float(safe_get(final_row, "test_loss", 0.0)),
        "final_test_acc": float(safe_get(final_row, "test_acc", 0.0)),
        "mean_epoch_time": float(df["epoch_time"].mean()),
        "total_elapsed_time": float(safe_get(final_row, "elapsed_time", 0.0)),
        "mean_samples_per_sec": float(df["samples_per_sec"].mean()),
        "final_comm_mb": float(safe_get(final_row, "comm_mb", 0.0)),
        "mean_comm_mb": float(df["comm_mb"].mean()),
        "model_params": int(safe_get(final_row, "model_params", 0)),
        "model_size_mb": float(safe_get(final_row, "model_size_mb", 0.0)),
        "seed": int(safe_get(first_row, "seed", 0)),
        "lr": float(safe_get(first_row, "lr", 0.0)),
        "batch_size": int(safe_get(first_row, "batch_size", 0)),
        "log_file": csv_path.name,
    }

    return summary


def build_system_label(row: pd.Series) -> str:
    """
    构造展示用系统名称。

    参数
    ----
    row : pd.Series
        summary 表中的一行。

    返回
    ----
    str
        展示标签。
    """

    system = str(row["system"]).lower()
    workers = int(row["num_workers"])

    if system == "single":
        return "Single"

    if system == "ps":
        return f"PS-{workers}"

    if system == "ddp":
        return f"DDP-{workers}"

    return f"{system}-{workers}"


def sort_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    对汇总表排序。

    排序顺序：
    1. Single；
    2. PS；
    3. DDP；
    4. worker 数从小到大。
    """

    system_order = {
        "single": 0,
        "ps": 1,
        "ddp": 2,
    }

    df = df.copy()

    df["_system_order"] = df["system"].str.lower().map(system_order).fillna(99)
    df["_worker_order"] = df["num_workers"].astype(int)

    df = df.sort_values(
        by=["dataset", "model", "seed", "_system_order", "_worker_order"]
    )

    df = df.drop(columns=["_system_order", "_worker_order"])

    return df.reset_index(drop=True)


def summarize_all(
    raw_dir: str | Path,
    output: str | Path,
    strict: bool = False,
) -> pd.DataFrame:
    """
    汇总所有实验结果。

    参数
    ----
    raw_dir : str | Path
        原始 CSV 日志目录。
    output : str | Path
        汇总表输出路径。
    strict : bool
        是否启用严格模式。

    返回
    ----
    pd.DataFrame
        汇总结果表。
    """

    csv_files = find_csv_files(raw_dir)

    if not csv_files:
        raise FileNotFoundError(f"没有在 {raw_dir} 找到任何 CSV 文件")

    summaries = []

    for csv_path in csv_files:
        summary = summarize_one_file(csv_path, strict=strict)

        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise RuntimeError("没有可用于汇总的有效 CSV 文件")

    summary_df = pd.DataFrame(summaries)

    summary_df["system_label"] = summary_df.apply(build_system_label, axis=1)

    summary_df = sort_summary(summary_df)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output, index=False, encoding="utf-8-sig")

    return summary_df


def print_summary_table(summary_df: pd.DataFrame) -> None:
    """
    打印简洁版汇总表。
    """

    display_columns = [
        "system_label",
        "final_test_acc",
        "mean_epoch_time",
        "mean_samples_per_sec",
        "final_comm_mb",
        "total_elapsed_time",
    ]

    existing_columns = [
        col for col in display_columns
        if col in summary_df.columns
    ]

    printable_df = summary_df[existing_columns].copy()

    if "final_test_acc" in printable_df.columns:
        printable_df["final_test_acc"] = printable_df["final_test_acc"] * 100

    print("=" * 80)
    print("实验汇总结果")
    print("=" * 80)
    print(printable_df.to_string(index=False))
    print("=" * 80)


def main() -> None:
    """
    主函数。
    """

    args = parse_args()

    summary_df = summarize_all(
        raw_dir=args.raw_dir,
        output=args.output,
        strict=args.strict,
    )

    print_summary_table(summary_df)

    print(f"汇总表已保存到: {args.output}")


if __name__ == "__main__":
    main()