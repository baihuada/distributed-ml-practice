#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/simulator/compare_steps.py

文件作用：
    该文件用于比较标准 Ring All-Reduce 与 IBing All-Reduce 的通信步数。

    当前阶段的重点不是比较真实通信时间，而是先验证论文中的核心理论结论：

        Ring All-Reduce 总通信步数：
            2(N - 1)

        IBing All-Reduce 总通信步数：
            N - 1

        因此 IBing 相比 Ring 的通信步数减少比例为：
            50%

    该文件会输出不同 world_size 下的通信步数对比表，并可选保存为 CSV 文件。

使用示例：
    使用默认 world_size 列表：
        python src/simulator/compare_steps.py

    指定多个 world_size：
        python src/simulator/compare_steps.py --world_sizes 3 4 5 6 7 8

    自动生成 2 到 16 的 world_size：
        python src/simulator/compare_steps.py --min_world_size 2 --max_world_size 16

    保存 CSV：
        python src/simulator/compare_steps.py --world_sizes 3 4 5 6 7 8 --output results/tables/step_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


# ---------------------------------------------------------------------
# 路径处理
#
# 当从项目根目录运行：
#     python src/simulator/compare_steps.py
#
# Python 通常可以直接找到同目录下的 ring_sim.py 和 ibing_schedule.py。
# 但为了兼容从其他目录、IDE 或测试框架中运行，这里手动加入项目路径。
# ---------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
SIMULATOR_DIR = CURRENT_FILE.parent
SRC_DIR = SIMULATOR_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

for path in (PROJECT_ROOT, SRC_DIR, SIMULATOR_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


try:
    from ibing_schedule import get_total_steps as get_ibing_total_steps
    from ibing_schedule import get_reduce_steps as get_ibing_reduce_steps
    from ring_sim import get_total_steps as get_ring_total_steps
    from ring_sim import get_reduce_scatter_steps as get_ring_reduce_scatter_steps
    from ring_sim import get_allgather_steps as get_ring_allgather_steps
except ImportError:
    from src.simulator.ibing_schedule import get_total_steps as get_ibing_total_steps
    from src.simulator.ibing_schedule import get_reduce_steps as get_ibing_reduce_steps
    from src.simulator.ring_sim import get_total_steps as get_ring_total_steps
    from src.simulator.ring_sim import (
        get_reduce_scatter_steps as get_ring_reduce_scatter_steps,
    )
    from src.simulator.ring_sim import get_allgather_steps as get_ring_allgather_steps


DEFAULT_WORLD_SIZES = [2, 3, 4, 5, 6, 7, 8, 16]


@dataclass(frozen=True)
class StepComparisonResult:
    """
    保存某个 world_size 下 Ring 和 IBing 的通信步数对比结果。

    Attributes:
        world_size:
            worker 总数 N。

        ring_reduce_scatter_steps:
            Ring All-Reduce 的 Reduce-Scatter 阶段步数，即 N - 1。

        ring_allgather_steps:
            Ring All-Reduce 的 All-Gather 阶段步数，即 N - 1。

        ring_total_steps:
            Ring All-Reduce 的总通信步数，即 2(N - 1)。

        ibing_reduce_steps:
            IBing 的 Reduce-Scatter 阶段步数。
            当前实现中采用 N // 2。

        ibing_allgather_steps:
            IBing 的 All-Gather 阶段步数。

        ibing_total_steps:
            IBing 的总通信步数，即 N - 1。

        reduced_steps:
            Ring 总步数减去 IBing 总步数。

        reduction_rate:
            IBing 相比 Ring 的通信步数减少比例，单位为百分比。

        speedup_by_steps:
            仅从通信步数角度计算的理论加速比：
                ring_total_steps / ibing_total_steps

            注意：
                这不是实际运行时间加速比，只是步数层面的理论比例。
    """

    world_size: int
    ring_reduce_scatter_steps: int
    ring_allgather_steps: int
    ring_total_steps: int
    ibing_reduce_steps: int
    ibing_allgather_steps: int
    ibing_total_steps: int
    reduced_steps: int
    reduction_rate: float
    speedup_by_steps: float

    def as_dict(self) -> dict[str, int | float]:
        """
        将结果转换为字典形式，方便保存为 CSV。

        Returns:
            包含当前 world_size 对比结果的字典。
        """

        return {
            "world_size": self.world_size,
            "ring_reduce_scatter_steps": self.ring_reduce_scatter_steps,
            "ring_allgather_steps": self.ring_allgather_steps,
            "ring_total_steps": self.ring_total_steps,
            "ibing_reduce_steps": self.ibing_reduce_steps,
            "ibing_allgather_steps": self.ibing_allgather_steps,
            "ibing_total_steps": self.ibing_total_steps,
            "reduced_steps": self.reduced_steps,
            "reduction_rate_percent": self.reduction_rate,
            "speedup_by_steps": self.speedup_by_steps,
        }


def validate_world_size(world_size: int) -> None:
    """
    检查 world_size 是否合法。

    Args:
        world_size:
            worker 总数。

    Raises:
        ValueError:
            当 world_size 小于 2 时抛出异常。
    """

    if world_size < 2:
        raise ValueError(f"world_size must be >= 2, but got {world_size}.")


def compare_steps_for_world_size(world_size: int) -> StepComparisonResult:
    """
    计算某个 world_size 下 Ring 与 IBing 的通信步数对比结果。

    对于 N 个 worker：

    Ring All-Reduce：
        Reduce-Scatter 阶段需要 N - 1 步；
        All-Gather 阶段需要 N - 1 步；
        总步数为 2(N - 1)。

    IBing All-Reduce：
        通过双向交错通信，总步数为 N - 1。

    Args:
        world_size:
            worker 总数 N。

    Returns:
        StepComparisonResult:
            当前 world_size 下的通信步数对比结果。
    """

    validate_world_size(world_size)

    ring_reduce_scatter_steps = get_ring_reduce_scatter_steps(world_size)
    ring_allgather_steps = get_ring_allgather_steps(world_size)
    ring_total_steps = get_ring_total_steps(world_size)

    ibing_reduce_steps = get_ibing_reduce_steps(world_size)
    ibing_total_steps = get_ibing_total_steps(world_size)
    ibing_allgather_steps = ibing_total_steps - ibing_reduce_steps

    reduced_steps = ring_total_steps - ibing_total_steps

    reduction_rate = reduced_steps / ring_total_steps * 100.0
    speedup_by_steps = ring_total_steps / ibing_total_steps

    return StepComparisonResult(
        world_size=world_size,
        ring_reduce_scatter_steps=ring_reduce_scatter_steps,
        ring_allgather_steps=ring_allgather_steps,
        ring_total_steps=ring_total_steps,
        ibing_reduce_steps=ibing_reduce_steps,
        ibing_allgather_steps=ibing_allgather_steps,
        ibing_total_steps=ibing_total_steps,
        reduced_steps=reduced_steps,
        reduction_rate=reduction_rate,
        speedup_by_steps=speedup_by_steps,
    )


def compare_steps(world_sizes: Iterable[int]) -> List[StepComparisonResult]:
    """
    批量计算多个 world_size 下的通信步数对比结果。

    Args:
        world_sizes:
            需要比较的 worker 数量列表。

    Returns:
        results:
            每个 world_size 对应的 StepComparisonResult 列表。
    """

    results: List[StepComparisonResult] = []

    for world_size in world_sizes:
        result = compare_steps_for_world_size(world_size)
        results.append(result)

    return results


def format_results_table(results: List[StepComparisonResult]) -> str:
    """
    将通信步数对比结果格式化为表格字符串。

    Args:
        results:
            通信步数对比结果列表。

    Returns:
        格式化后的表格字符串。
    """

    lines = []

    header = (
        f"{'N':>6} | "
        f"{'Ring RS':>8} | "
        f"{'Ring AG':>8} | "
        f"{'Ring Total':>11} | "
        f"{'IBing RS':>8} | "
        f"{'IBing AG':>8} | "
        f"{'IBing Total':>12} | "
        f"{'Reduced':>8} | "
        f"{'Reduction':>10} | "
        f"{'Step Speedup':>12}"
    )

    separator = "-" * len(header)

    lines.append(separator)
    lines.append(header)
    lines.append(separator)

    for result in results:
        line = (
            f"{result.world_size:>6} | "
            f"{result.ring_reduce_scatter_steps:>8} | "
            f"{result.ring_allgather_steps:>8} | "
            f"{result.ring_total_steps:>11} | "
            f"{result.ibing_reduce_steps:>8} | "
            f"{result.ibing_allgather_steps:>8} | "
            f"{result.ibing_total_steps:>12} | "
            f"{result.reduced_steps:>8} | "
            f"{result.reduction_rate:>9.2f}% | "
            f"{result.speedup_by_steps:>12.2f}x"
        )

        lines.append(line)

    lines.append(separator)

    return "\n".join(lines)


def print_results(results: List[StepComparisonResult]) -> None:
    """
    打印通信步数对比结果。

    Args:
        results:
            通信步数对比结果列表。
    """

    print("=" * 100)
    print("Ring vs IBing Communication Step Comparison")
    print("=" * 100)
    print()
    print(format_results_table(results))
    print()
    print("Notes:")
    print("  Ring RS      : Ring Reduce-Scatter steps")
    print("  Ring AG      : Ring All-Gather steps")
    print("  Ring Total   : total Ring communication steps, 2(N - 1)")
    print("  IBing RS     : IBing Reduce-Scatter steps")
    print("  IBing AG     : IBing All-Gather steps")
    print("  IBing Total  : total IBing communication steps, N - 1")
    print("  Reduction    : step reduction rate compared with Ring")
    print("  Step Speedup : Ring Total / IBing Total")
    print()
    print("Conclusion:")
    print("  IBing reduces the communication steps from 2(N - 1) to N - 1.")
    print("  Therefore, the theoretical step-level reduction is 50%.")
    print("=" * 100)


def save_results_to_csv(
    results: List[StepComparisonResult],
    output_path: str | Path,
) -> None:
    """
    将通信步数对比结果保存为 CSV 文件。

    Args:
        results:
            通信步数对比结果列表。

        output_path:
            CSV 输出路径。
    """

    output_path = Path(output_path)

    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "world_size",
        "ring_reduce_scatter_steps",
        "ring_allgather_steps",
        "ring_total_steps",
        "ibing_reduce_steps",
        "ibing_allgather_steps",
        "ibing_total_steps",
        "reduced_steps",
        "reduction_rate_percent",
        "speedup_by_steps",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result.as_dict())

    print(f"[OK] Step comparison results saved to: {output_path}")


def parse_world_sizes(args: argparse.Namespace) -> List[int]:
    """
    根据命令行参数解析 world_size 列表。

    优先级：
        1. 如果指定 --world_sizes，则使用该列表；
        2. 否则如果指定 --min_world_size 和 --max_world_size，
           则生成闭区间 [min_world_size, max_world_size]；
        3. 如果都没有指定，则使用 DEFAULT_WORLD_SIZES。

    Args:
        args:
            argparse 解析得到的参数。

    Returns:
        world_sizes:
            需要比较的 world_size 列表。
    """

    if args.world_sizes is not None and len(args.world_sizes) > 0:
        world_sizes = args.world_sizes
    elif args.min_world_size is not None or args.max_world_size is not None:
        min_n = args.min_world_size if args.min_world_size is not None else 2
        max_n = args.max_world_size if args.max_world_size is not None else 16

        if min_n > max_n:
            raise ValueError(
                f"min_world_size must be <= max_world_size, "
                f"but got min={min_n}, max={max_n}."
            )

        world_sizes = list(range(min_n, max_n + 1))
    else:
        world_sizes = DEFAULT_WORLD_SIZES

    # 去重并排序，保证输出稳定。
    world_sizes = sorted(set(world_sizes))

    for world_size in world_sizes:
        validate_world_size(world_size)

    return world_sizes


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace:
            命令行参数对象。
    """

    parser = argparse.ArgumentParser(
        description="Compare communication steps of Ring and IBing All-Reduce."
    )

    parser.add_argument(
        "--world_sizes",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Specific world sizes to compare. "
            "Example: --world_sizes 3 4 5 6 7 8"
        ),
    )

    parser.add_argument(
        "--min_world_size",
        type=int,
        default=None,
        help=(
            "Minimum world size. "
            "Used with --max_world_size to generate a range."
        ),
    )

    parser.add_argument(
        "--max_world_size",
        type=int,
        default=None,
        help=(
            "Maximum world size. "
            "Used with --min_world_size to generate a range."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional CSV output path. "
            "Example: --output results/tables/step_comparison.csv"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    程序入口。

    执行流程：
        1. 解析命令行参数；
        2. 得到需要比较的 world_size 列表；
        3. 计算 Ring 与 IBing 的通信步数；
        4. 打印对比表；
        5. 如果指定 --output，则保存 CSV 文件。
    """

    args = parse_args()

    try:
        world_sizes = parse_world_sizes(args)
        results = compare_steps(world_sizes)
    except Exception as exc:
        print("[FAIL] Failed to compare communication steps.")
        print(f"Reason: {exc}")
        raise SystemExit(1) from exc

    print_results(results)

    if args.output is not None:
        save_results_to_csv(
            results=results,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()