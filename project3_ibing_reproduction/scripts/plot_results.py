#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/plot_results.py

文件作用：
    该文件用于读取 MPI benchmark 的 CSV 结果，并生成论文复现报告中可用的图表。

默认输入：
    results/tables/mpi_benchmark_all.csv

默认输出目录：
    results/figures/

默认输出汇总表：
    results/tables/mpi_speedup_summary.csv

当前支持生成的图：
    1. 每个 world_size 下，不同算法随 data_size 变化的通信时间折线图；
    2. 每个 world_size 下，不同算法在不同 data_size 上的通信时间柱状图；
    3. Ring 与 IBing 的理论通信步数对比图；
    4. Ring vs IBing 的 speedup 折线图；
    5. Ring vs IBing 的优化率折线图。

输入 CSV 来源：
    一般由 scripts/run_benchmark.py 自动生成：

        python scripts/run_benchmark.py

    或者由 src/mpi/benchmark.py 单独生成：

        mpiexec -n 5 python src/mpi/benchmark.py --algo all --data_sizes_mb 1 10 50 --output results/raw/mpi_benchmark_n5.csv

注意：
    本文件不是 MPI 程序，不需要使用 mpiexec 启动。

正确运行方式：
    python scripts/plot_results.py

依赖：
    pip install pandas matplotlib
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = "results/tables/mpi_benchmark_all.csv"
DEFAULT_OUTPUT_DIR = "results/figures"
DEFAULT_SPEEDUP_OUTPUT = "results/tables/mpi_speedup_summary.csv"

DEFAULT_ALGO_ORDER = ["ring", "ibing", "mpi_allreduce"]

ALGO_LABELS = {
    "ring": "Ring",
    "ibing": "IBing",
    "mpi_allreduce": "MPI_Allreduce",
}


@dataclass(frozen=True)
class PlotConfig:
    """
    绘图配置。

    Attributes:
        input_path:
            benchmark CSV 输入路径。

        output_dir:
            图像输出目录。

        speedup_output:
            Ring vs IBing 加速比汇总 CSV 输出路径。

        formats:
            图像输出格式，例如 ["png"] 或 ["png", "pdf"]。

        dpi:
            图片分辨率。

        algos:
            需要绘制的算法列表。

        log_x:
            是否对 data_size 横轴使用对数坐标。

        show:
            是否在绘图后弹出窗口显示。

        title_suffix:
            图标题后缀，可用于标注实验环境。
    """

    input_path: Path
    output_dir: Path
    speedup_output: Path
    formats: List[str]
    dpi: int
    algos: List[str]
    log_x: bool
    show: bool
    title_suffix: str


def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件位于：
        scripts/plot_results.py

    因此：
        当前文件 parent        = scripts
        当前文件 parent.parent = 项目根目录

    Returns:
        项目根目录路径。
    """

    current_file = Path(__file__).resolve()
    scripts_dir = current_file.parent
    project_root = scripts_dir.parent
    return project_root


def resolve_path(path: str | Path) -> Path:
    """
    将相对路径解析为相对于项目根目录的路径。

    Args:
        path:
            输入路径。

    Returns:
        解析后的绝对路径或相对项目根目录路径。
    """

    path = Path(path)

    if path.is_absolute():
        return path

    return get_project_root() / path


def ensure_output_dirs(config: PlotConfig) -> None:
    """
    创建输出目录。

    Args:
        config:
            绘图配置。
    """

    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.speedup_output.parent:
        config.speedup_output.parent.mkdir(parents=True, exist_ok=True)


def validate_input_file(input_path: Path) -> None:
    """
    检查输入 CSV 是否存在。

    Args:
        input_path:
            输入 CSV 路径。

    Raises:
        FileNotFoundError:
            当文件不存在时抛出异常。
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}\n"
            f"Please run scripts/run_benchmark.py first."
        )


def load_results(input_path: Path) -> pd.DataFrame:
    """
    读取 benchmark CSV。

    Args:
        input_path:
            CSV 输入路径。

    Returns:
        pandas DataFrame。
    """

    validate_input_file(input_path)

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(f"Input CSV is empty: {input_path}")

    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化 CSV 数据类型和字段。

    必需字段：
        algo
        world_size
        data_size_mb_actual
        chunk_size
        step_count
        correct
        mean_ms
        std_ms
        min_ms
        max_ms

    Args:
        df:
            原始 DataFrame。

    Returns:
        标准化后的 DataFrame。
    """

    required_columns = [
        "algo",
        "world_size",
        "data_size_mb_actual",
        "chunk_size",
        "step_count",
        "correct",
        "mean_ms",
        "std_ms",
        "min_ms",
        "max_ms",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in input CSV: {missing_columns}\n"
            f"Existing columns: {list(df.columns)}"
        )

    df = df.copy()

    numeric_columns = [
        "world_size",
        "data_size_mb_actual",
        "chunk_size",
        "step_count",
        "mean_ms",
        "std_ms",
        "min_ms",
        "max_ms",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["algo"] = df["algo"].astype(str)

    # correct 有时从 CSV 读入为字符串 "True"/"False"。
    if df["correct"].dtype == object:
        df["correct"] = df["correct"].astype(str).str.lower().isin(["true", "1", "yes"])

    df = df.dropna(
        subset=[
            "world_size",
            "data_size_mb_actual",
            "chunk_size",
            "step_count",
            "mean_ms",
        ]
    )

    df["world_size"] = df["world_size"].astype(int)
    df["chunk_size"] = df["chunk_size"].astype(int)
    df["step_count"] = df["step_count"].astype(int)

    return df


def filter_algorithms(df: pd.DataFrame, algos: Iterable[str]) -> pd.DataFrame:
    """
    只保留指定算法。

    Args:
        df:
            输入 DataFrame。

        algos:
            需要保留的算法。

    Returns:
        筛选后的 DataFrame。
    """

    algos = list(algos)
    return df[df["algo"].isin(algos)].copy()


def aggregate_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    合并重复测试结果。

    一般情况下 scripts/run_benchmark.py 生成的 CSV 不会重复。
    但如果用户多次合并或手动拼接 CSV，可能存在同一组：
        algo, world_size, data_size_mb_actual, chunk_size, dtype

    出现多行的情况。

    这里对重复项做均值聚合，避免绘图时报错。

    Args:
        df:
            输入 DataFrame。

    Returns:
        聚合后的 DataFrame。
    """

    group_cols = [
        "algo",
        "world_size",
        "data_size_mb_actual",
        "chunk_size",
    ]

    if "dtype" in df.columns:
        group_cols.append("dtype")

    agg_dict = {
        "step_count": "first",
        "correct": "all",
        "mean_ms": "mean",
        "std_ms": "mean",
        "min_ms": "min",
        "max_ms": "max",
    }

    if "data_size_mb_target" in df.columns:
        agg_dict["data_size_mb_target"] = "first"

    aggregated = (
        df.groupby(group_cols, as_index=False)
        .agg(agg_dict)
        .sort_values(["world_size", "data_size_mb_actual", "algo"])
    )

    return aggregated


def get_available_algorithms(df: pd.DataFrame, requested_algos: List[str]) -> List[str]:
    """
    返回实际存在且需要绘制的算法列表。

    Args:
        df:
            输入 DataFrame。

        requested_algos:
            用户请求的算法顺序。

    Returns:
        实际可绘制的算法列表。
    """

    available = set(df["algo"].unique().tolist())

    algos = [algo for algo in requested_algos if algo in available]

    return algos


def get_world_sizes(df: pd.DataFrame) -> List[int]:
    """
    获取所有 world_size。

    Args:
        df:
            输入 DataFrame。

    Returns:
        world_size 排序列表。
    """

    return sorted(df["world_size"].unique().astype(int).tolist())


def format_data_size_label(value: float) -> str:
    """
    格式化 data size 标签。

    Args:
        value:
            数据大小，MiB。

    Returns:
        标签字符串。
    """

    if math.isclose(value, round(value), rel_tol=0, abs_tol=1e-9):
        return f"{int(round(value))}"

    return f"{value:.3f}"


def algo_label(algo: str) -> str:
    """
    将算法内部名转换为显示名。

    Args:
        algo:
            算法内部名称。

    Returns:
        显示名称。
    """

    return ALGO_LABELS.get(algo, algo)


def set_common_axes_style(ax) -> None:
    """
    设置通用坐标轴样式。

    Args:
        ax:
            matplotlib Axes。
    """

    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="both", labelsize=10)


def save_figure(fig, output_base: Path, formats: List[str], dpi: int) -> None:
    """
    保存图像。

    Args:
        fig:
            matplotlib Figure。

        output_base:
            不含扩展名的输出路径。

        formats:
            输出格式列表。

        dpi:
            图片分辨率。
    """

    for fmt in formats:
        output_path = output_base.with_suffix(f".{fmt}")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"[OK] Figure saved: {output_path}")


def plot_time_line_by_world_size(
    df: pd.DataFrame,
    config: PlotConfig,
    algos: List[str],
) -> None:
    """
    为每个 world_size 生成通信时间折线图。

    横轴：
        data_size_mb_actual

    纵轴：
        mean_ms

    每条线：
        一个算法。

    Args:
        df:
            benchmark 结果。

        config:
            绘图配置。

        algos:
            算法列表。
    """

    world_sizes = get_world_sizes(df)

    for world_size in world_sizes:
        sub_df = df[df["world_size"] == world_size].copy()

        if sub_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for algo in algos:
            algo_df = (
                sub_df[sub_df["algo"] == algo]
                .sort_values("data_size_mb_actual")
                .copy()
            )

            if algo_df.empty:
                continue

            ax.plot(
                algo_df["data_size_mb_actual"],
                algo_df["mean_ms"],
                marker="o",
                label=algo_label(algo),
            )

        title = f"All-Reduce time comparison, N={world_size}"
        if config.title_suffix:
            title += f" ({config.title_suffix})"

        ax.set_title(title)
        ax.set_xlabel("Data size per rank (MiB)")
        ax.set_ylabel("Mean time (ms)")

        if config.log_x:
            ax.set_xscale("log")

        set_common_axes_style(ax)
        ax.legend()

        output_base = config.output_dir / f"time_line_n{world_size}"
        save_figure(fig, output_base, config.formats, config.dpi)

        if config.show:
            plt.show()

        plt.close(fig)


def plot_time_bar_by_world_size(
    df: pd.DataFrame,
    config: PlotConfig,
    algos: List[str],
) -> None:
    """
    为每个 world_size 生成通信时间柱状图。

    横轴：
        data_size_mb_actual

    每组柱子：
        不同算法。

    纵轴：
        mean_ms

    Args:
        df:
            benchmark 结果。

        config:
            绘图配置。

        algos:
            算法列表。
    """

    world_sizes = get_world_sizes(df)

    for world_size in world_sizes:
        sub_df = df[df["world_size"] == world_size].copy()

        if sub_df.empty:
            continue

        data_sizes = sorted(sub_df["data_size_mb_actual"].unique().tolist())
        x = np.arange(len(data_sizes))

        num_algos = max(len(algos), 1)
        width = 0.8 / num_algos

        fig, ax = plt.subplots(figsize=(8, 5))

        for idx, algo in enumerate(algos):
            values = []

            for data_size in data_sizes:
                row = sub_df[
                    (sub_df["algo"] == algo)
                    & (np.isclose(sub_df["data_size_mb_actual"], data_size))
                ]

                if row.empty:
                    values.append(np.nan)
                else:
                    values.append(float(row.iloc[0]["mean_ms"]))

            offset = (idx - (num_algos - 1) / 2) * width

            ax.bar(
                x + offset,
                values,
                width=width,
                label=algo_label(algo),
            )

        labels = [format_data_size_label(size) for size in data_sizes]

        title = f"All-Reduce time by data size, N={world_size}"
        if config.title_suffix:
            title += f" ({config.title_suffix})"

        ax.set_title(title)
        ax.set_xlabel("Data size per rank (MiB)")
        ax.set_ylabel("Mean time (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        set_common_axes_style(ax)
        ax.legend()

        output_base = config.output_dir / f"time_bar_n{world_size}"
        save_figure(fig, output_base, config.formats, config.dpi)

        if config.show:
            plt.show()

        plt.close(fig)


def build_step_count_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造 Ring 与 IBing 的步数对比 DataFrame。

    优先使用 CSV 中已有的 step_count。
    如果缺失或异常，则根据理论公式补充：
        Ring  = 2(N - 1)
        IBing = N - 1

    Args:
        df:
            benchmark 结果。

    Returns:
        步数对比 DataFrame。
    """

    rows = []
    world_sizes = get_world_sizes(df)

    for world_size in world_sizes:
        for algo in ["ring", "ibing"]:
            sub_df = df[
                (df["world_size"] == world_size)
                & (df["algo"] == algo)
            ]

            if not sub_df.empty:
                step_count = int(sub_df.iloc[0]["step_count"])
            else:
                if algo == "ring":
                    step_count = 2 * (world_size - 1)
                else:
                    step_count = world_size - 1

            rows.append(
                {
                    "world_size": world_size,
                    "algo": algo,
                    "step_count": step_count,
                }
            )

    return pd.DataFrame(rows)


def plot_step_count_comparison(
    df: pd.DataFrame,
    config: PlotConfig,
) -> None:
    """
    绘制 Ring 与 IBing 的理论通信步数对比图。

    Args:
        df:
            benchmark 结果。

        config:
            绘图配置。
    """

    step_df = build_step_count_dataframe(df)

    if step_df.empty:
        return

    world_sizes = sorted(step_df["world_size"].unique().tolist())
    algos = ["ring", "ibing"]

    x = np.arange(len(world_sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, algo in enumerate(algos):
        values = []

        for world_size in world_sizes:
            row = step_df[
                (step_df["world_size"] == world_size)
                & (step_df["algo"] == algo)
            ]

            if row.empty:
                values.append(np.nan)
            else:
                values.append(int(row.iloc[0]["step_count"]))

        offset = (idx - 0.5) * width

        ax.bar(
            x + offset,
            values,
            width=width,
            label=algo_label(algo),
        )

    title = "Theoretical communication steps"
    if config.title_suffix:
        title += f" ({config.title_suffix})"

    ax.set_title(title)
    ax.set_xlabel("World size")
    ax.set_ylabel("Communication steps")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in world_sizes])

    set_common_axes_style(ax)
    ax.legend()

    output_base = config.output_dir / "step_count_comparison"
    save_figure(fig, output_base, config.formats, config.dpi)

    if config.show:
        plt.show()

    plt.close(fig)


def compute_speedup_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 Ring vs IBing 的加速比和优化率。

    对于同一个 world_size 和 data_size：

        speedup = ring_mean_ms / ibing_mean_ms

        opt_rate_percent =
            (ring_mean_ms - ibing_mean_ms) / ring_mean_ms * 100

    当 speedup > 1 时，表示 IBing 比 Ring 快。
    当 opt_rate_percent > 0 时，表示 IBing 比 Ring 有优化。
    当 opt_rate_percent < 0 时，表示 IBing 比 Ring 退化。

    Args:
        df:
            benchmark 结果。

    Returns:
        speedup summary DataFrame。
    """

    rows = []

    world_sizes = get_world_sizes(df)

    for world_size in world_sizes:
        sub_df = df[df["world_size"] == world_size]

        data_sizes = sorted(sub_df["data_size_mb_actual"].unique().tolist())

        for data_size in data_sizes:
            ring_df = sub_df[
                (sub_df["algo"] == "ring")
                & (np.isclose(sub_df["data_size_mb_actual"], data_size))
            ]

            ibing_df = sub_df[
                (sub_df["algo"] == "ibing")
                & (np.isclose(sub_df["data_size_mb_actual"], data_size))
            ]

            if ring_df.empty or ibing_df.empty:
                continue

            ring_mean_ms = float(ring_df.iloc[0]["mean_ms"])
            ibing_mean_ms = float(ibing_df.iloc[0]["mean_ms"])

            if ibing_mean_ms <= 0 or ring_mean_ms <= 0:
                continue

            speedup = ring_mean_ms / ibing_mean_ms
            opt_rate_percent = (ring_mean_ms - ibing_mean_ms) / ring_mean_ms * 100.0

            rows.append(
                {
                    "world_size": world_size,
                    "data_size_mb": data_size,
                    "ring_mean_ms": ring_mean_ms,
                    "ibing_mean_ms": ibing_mean_ms,
                    "speedup": speedup,
                    "opt_rate_percent": opt_rate_percent,
                }
            )

    return pd.DataFrame(rows)


def save_speedup_summary(speedup_df: pd.DataFrame, output_path: Path) -> None:
    """
    保存 Ring vs IBing 加速比汇总表。

    Args:
        speedup_df:
            加速比 DataFrame。

        output_path:
            输出 CSV 路径。
    """

    if speedup_df.empty:
        print("[WARN] Speedup summary is empty. No CSV saved.")
        return

    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    speedup_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Speedup summary saved: {output_path}")


def plot_speedup_vs_data_size(
    speedup_df: pd.DataFrame,
    config: PlotConfig,
) -> None:
    """
    绘制 Ring vs IBing speedup 图。

    横轴：
        data_size_mb

    纵轴：
        speedup = Ring time / IBing time

    解释：
        speedup > 1 表示 IBing 更快；
        speedup = 1 表示二者持平；
        speedup < 1 表示 IBing 更慢。

    Args:
        speedup_df:
            加速比 DataFrame。

        config:
            绘图配置。
    """

    if speedup_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    world_sizes = sorted(speedup_df["world_size"].unique().tolist())

    for world_size in world_sizes:
        sub_df = (
            speedup_df[speedup_df["world_size"] == world_size]
            .sort_values("data_size_mb")
            .copy()
        )

        ax.plot(
            sub_df["data_size_mb"],
            sub_df["speedup"],
            marker="o",
            label=f"N={world_size}",
        )

    ax.axhline(1.0, linestyle="--", linewidth=1)

    title = "Ring vs IBing speedup"
    if config.title_suffix:
        title += f" ({config.title_suffix})"

    ax.set_title(title)
    ax.set_xlabel("Data size per rank (MiB)")
    ax.set_ylabel("Speedup, Ring time / IBing time")

    if config.log_x:
        ax.set_xscale("log")

    set_common_axes_style(ax)
    ax.legend()

    output_base = config.output_dir / "ring_vs_ibing_speedup"
    save_figure(fig, output_base, config.formats, config.dpi)

    if config.show:
        plt.show()

    plt.close(fig)


def plot_opt_rate_vs_data_size(
    speedup_df: pd.DataFrame,
    config: PlotConfig,
) -> None:
    """
    绘制 Ring vs IBing 优化率图。

    横轴：
        data_size_mb

    纵轴：
        opt_rate_percent

    解释：
        opt_rate_percent > 0 表示 IBing 比 Ring 快；
        opt_rate_percent = 0 表示二者持平；
        opt_rate_percent < 0 表示 IBing 比 Ring 慢。

    Args:
        speedup_df:
            加速比 DataFrame。

        config:
            绘图配置。
    """

    if speedup_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    world_sizes = sorted(speedup_df["world_size"].unique().tolist())

    for world_size in world_sizes:
        sub_df = (
            speedup_df[speedup_df["world_size"] == world_size]
            .sort_values("data_size_mb")
            .copy()
        )

        ax.plot(
            sub_df["data_size_mb"],
            sub_df["opt_rate_percent"],
            marker="o",
            label=f"N={world_size}",
        )

    ax.axhline(0.0, linestyle="--", linewidth=1)

    title = "IBing optimization rate over Ring"
    if config.title_suffix:
        title += f" ({config.title_suffix})"

    ax.set_title(title)
    ax.set_xlabel("Data size per rank (MiB)")
    ax.set_ylabel("Optimization rate (%)")

    if config.log_x:
        ax.set_xscale("log")

    set_common_axes_style(ax)
    ax.legend()

    output_base = config.output_dir / "ring_vs_ibing_opt_rate"
    save_figure(fig, output_base, config.formats, config.dpi)

    if config.show:
        plt.show()

    plt.close(fig)


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    打印输入数据摘要。

    Args:
        df:
            benchmark 结果。
    """

    print("=" * 100)
    print("Benchmark Result Summary")
    print("=" * 100)
    print(f"rows        : {len(df)}")
    print(f"algorithms  : {sorted(df['algo'].unique().tolist())}")
    print(f"world_sizes : {get_world_sizes(df)}")
    print(
        "data_sizes  : "
        f"{[round(x, 6) for x in sorted(df['data_size_mb_actual'].unique().tolist())]}"
    )
    print("=" * 100)


def print_speedup_summary(speedup_df: pd.DataFrame) -> None:
    """
    打印 Ring vs IBing 加速比摘要。

    Args:
        speedup_df:
            加速比 DataFrame。
    """

    if speedup_df.empty:
        print("[WARN] No Ring vs IBing speedup data available.")
        return

    print()
    print("=" * 100)
    print("Ring vs IBing Speedup Summary")
    print("=" * 100)

    header = (
        f"{'N':>5} | "
        f"{'Data(MiB)':>10} | "
        f"{'Ring(ms)':>10} | "
        f"{'IBing(ms)':>10} | "
        f"{'Speedup':>10} | "
        f"{'OptRate':>10}"
    )

    print(header)
    print("-" * len(header))

    for _, row in speedup_df.sort_values(["world_size", "data_size_mb"]).iterrows():
        print(
            f"{int(row['world_size']):>5} | "
            f"{float(row['data_size_mb']):>10.3f} | "
            f"{float(row['ring_mean_ms']):>10.3f} | "
            f"{float(row['ibing_mean_ms']):>10.3f} | "
            f"{float(row['speedup']):>9.3f}x | "
            f"{float(row['opt_rate_percent']):>9.2f}%"
        )

    print("=" * 100)


def generate_all_plots(config: PlotConfig) -> None:
    """
    生成全部图表。

    Args:
        config:
            绘图配置。
    """

    ensure_output_dirs(config)

    df = load_results(config.input_path)
    df = normalize_dataframe(df)
    df = filter_algorithms(df, config.algos)
    df = aggregate_duplicate_rows(df)

    if df.empty:
        raise ValueError("No valid benchmark rows found after filtering.")

    algos = get_available_algorithms(df, config.algos)

    if not algos:
        raise ValueError(
            f"No requested algorithms found in CSV. Requested={config.algos}, "
            f"available={sorted(df['algo'].unique().tolist())}"
        )

    print_dataset_summary(df)

    plot_time_line_by_world_size(
        df=df,
        config=config,
        algos=algos,
    )

    plot_time_bar_by_world_size(
        df=df,
        config=config,
        algos=algos,
    )

    plot_step_count_comparison(
        df=df,
        config=config,
    )

    speedup_df = compute_speedup_summary(df)

    save_speedup_summary(
        speedup_df=speedup_df,
        output_path=config.speedup_output,
    )

    print_speedup_summary(speedup_df)

    plot_speedup_vs_data_size(
        speedup_df=speedup_df,
        config=config,
    )

    plot_opt_rate_vs_data_size(
        speedup_df=speedup_df,
        config=config,
    )


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace。
    """

    parser = argparse.ArgumentParser(
        description="Plot MPI benchmark results for Ring, IBing, and MPI_Allreduce."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=(
            "Input benchmark CSV path. "
            "Default: results/tables/mpi_benchmark_all.csv"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Figure output directory. "
            "Default: results/figures"
        ),
    )

    parser.add_argument(
        "--speedup_output",
        type=str,
        default=DEFAULT_SPEEDUP_OUTPUT,
        help=(
            "Ring vs IBing speedup summary CSV output path. "
            "Default: results/tables/mpi_speedup_summary.csv"
        ),
    )

    parser.add_argument(
        "--formats",
        type=str,
        nargs="*",
        default=["png"],
        help=(
            "Figure formats. "
            "Example: --formats png pdf. "
            "Default: png."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI. Default: 200.",
    )

    parser.add_argument(
        "--algos",
        type=str,
        nargs="*",
        default=DEFAULT_ALGO_ORDER,
        choices=DEFAULT_ALGO_ORDER,
        help=(
            "Algorithms to plot. "
            "Default: ring ibing mpi_allreduce."
        ),
    )

    parser.add_argument(
        "--log_x",
        action="store_true",
        help="Use log scale for data size x-axis.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving.",
    )

    parser.add_argument(
        "--title_suffix",
        type=str,
        default="",
        help=(
            "Optional suffix added to figure titles. "
            "Example: --title_suffix \"Windows MS-MPI\""
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    程序入口。

    执行流程：
        1. 解析参数；
        2. 读取 benchmark CSV；
        3. 标准化字段；
        4. 生成通信时间图；
        5. 生成通信步数图；
        6. 生成 Ring vs IBing speedup 图；
        7. 保存 speedup summary CSV。
    """

    args = parse_args()

    config = PlotConfig(
        input_path=resolve_path(args.input),
        output_dir=resolve_path(args.output_dir),
        speedup_output=resolve_path(args.speedup_output),
        formats=args.formats,
        dpi=args.dpi,
        algos=args.algos,
        log_x=args.log_x,
        show=args.show,
        title_suffix=args.title_suffix,
    )

    try:
        generate_all_plots(config)
    except Exception as exc:
        print("[FAIL] Failed to plot benchmark results.")
        print(f"Reason: {exc}")
        raise SystemExit(1) from exc

    print()
    print("=" * 100)
    print("[PASS] Plot generation completed.")
    print("=" * 100)
    print(f"Input CSV       : {config.input_path}")
    print(f"Figure directory: {config.output_dir}")
    print(f"Speedup CSV     : {config.speedup_output}")
    print("=" * 100)


if __name__ == "__main__":
    main()