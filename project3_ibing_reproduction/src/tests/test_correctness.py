#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/tests/test_correctness.py

文件作用：
    该文件用于自动化测试单进程模拟版 All-Reduce 算法的正确性。

当前测试对象：
    1. src/simulator/ibing_sim.py
    2. src/simulator/ring_sim.py

测试内容：
    对不同 world_size 批量运行 IBing 和 Ring 的单进程模拟，并检查：
        1. 最终每个 rank 是否都得到理论 All-Reduce 结果；
        2. 所有 rank 的最终结果是否完全一致；
        3. IBing 的总通信步数是否为 N - 1；
        4. Ring 的总通信步数是否为 2(N - 1)；
        5. Reduce-Scatter 和 All-Gather 的阶段步数是否合理。

使用示例：
    默认同时测试 IBing 和 Ring：
        python src/tests/test_correctness.py

    只测试 IBing：
        python src/tests/test_correctness.py --algo ibing

    只测试 Ring：
        python src/tests/test_correctness.py --algo ring

    测试指定 world_size：
        python src/tests/test_correctness.py --world_size 5

    测试多个 world_size：
        python src/tests/test_correctness.py --world_sizes 3 4 5 6 7 8

    打印详细结果：
        python src/tests/test_correctness.py --algo both --world_sizes 3 4 5 6 7 8 --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Literal, Optional


# ---------------------------------------------------------------------
# 路径处理
#
# 当从项目根目录运行：
#     python src/tests/test_correctness.py
#
# Python 默认不一定能正确找到 src/simulator/ 下的文件。
# 因此这里手动把项目根目录和 src 目录加入 sys.path。
# ---------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
TESTS_DIR = CURRENT_FILE.parent
SRC_DIR = TESTS_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

for path in (PROJECT_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


try:
    from simulator.ibing_sim import (
        SimulationResult as IbingSimulationResult,
        simulate_ibing_allreduce,
    )
    from simulator.ring_sim import (
        SimulationResult as RingSimulationResult,
        simulate_ring_allreduce,
    )
except ImportError:
    from src.simulator.ibing_sim import (
        SimulationResult as IbingSimulationResult,
        simulate_ibing_allreduce,
    )
    from src.simulator.ring_sim import (
        SimulationResult as RingSimulationResult,
        simulate_ring_allreduce,
    )


AlgorithmName = Literal["ibing", "ring", "both"]

DEFAULT_WORLD_SIZES = [3, 4, 5, 6, 7, 8]


class CorrectnessTestError(AssertionError):
    """
    正确性测试失败时抛出的异常。

    继承 AssertionError，方便后续接入 pytest 时识别为断言失败。
    """

    pass


def assert_equal(actual, expected, message: str) -> None:
    """
    检查 actual 是否等于 expected。

    Args:
        actual:
            实际值。

        expected:
            期望值。

        message:
            断言失败时显示的信息。

    Raises:
        CorrectnessTestError:
            当 actual != expected 时抛出。
    """

    if actual != expected:
        raise CorrectnessTestError(
            f"{message}\n"
            f"  expected: {expected}\n"
            f"  actual  : {actual}"
        )


def assert_true(condition: bool, message: str) -> None:
    """
    检查 condition 是否为 True。

    Args:
        condition:
            待检查条件。

        message:
            断言失败时显示的信息。

    Raises:
        CorrectnessTestError:
            当 condition 为 False 时抛出。
    """

    if not condition:
        raise CorrectnessTestError(message)


def validate_final_workers(
    final_workers: List[List[int]],
    expected_result: List[int],
    world_size: int,
    algo_name: str,
) -> None:
    """
    检查所有 rank 的最终结果是否正确。

    正确条件：
        1. 每个 rank 的最终结果都等于 expected_result；
        2. 所有 rank 的最终结果彼此一致。

    Args:
        final_workers:
            模拟结束后所有 rank 的数据。

        expected_result:
            理论 All-Reduce 结果。

        world_size:
            worker 总数。

        algo_name:
            当前测试算法名称，用于错误信息显示。

    Raises:
        CorrectnessTestError:
            当任意 rank 结果不正确时抛出。
    """

    assert_true(
        condition=len(final_workers) == world_size,
        message=(
            f"[{algo_name}][world_size={world_size}] "
            f"final_workers size mismatch."
        ),
    )

    for rank, rank_data in enumerate(final_workers):
        assert_equal(
            actual=rank_data,
            expected=expected_result,
            message=(
                f"[{algo_name}][world_size={world_size}] "
                f"final data mismatch at rank {rank}."
            ),
        )

    reference = final_workers[0]

    for rank, rank_data in enumerate(final_workers):
        assert_equal(
            actual=rank_data,
            expected=reference,
            message=(
                f"[{algo_name}][world_size={world_size}] "
                f"final workers are not identical. "
                f"Mismatch at rank {rank}."
            ),
        )


def validate_ibing_result(result: IbingSimulationResult) -> None:
    """
    对一次 IBing 模拟结果进行正确性检查。

    检查内容：
        1. total_steps 是否为 N - 1；
        2. reduce_steps + allgather_steps 是否等于 total_steps；
        3. result.is_correct 是否为 True；
        4. 所有 rank 最终结果是否等于 expected_result。

    Args:
        result:
            simulate_ibing_allreduce 返回的模拟结果。

    Raises:
        CorrectnessTestError:
            当任意检查不通过时抛出。
    """

    world_size = result.world_size

    expected_total_steps = world_size - 1

    assert_equal(
        actual=result.total_steps,
        expected=expected_total_steps,
        message=f"[IBing][world_size={world_size}] total_steps check failed.",
    )

    assert_equal(
        actual=result.reduce_steps + result.allgather_steps,
        expected=result.total_steps,
        message=f"[IBing][world_size={world_size}] phase steps check failed.",
    )

    assert_true(
        condition=result.is_correct,
        message=f"[IBing][world_size={world_size}] result.is_correct is False.",
    )

    validate_final_workers(
        final_workers=result.final_workers,
        expected_result=result.expected_result,
        world_size=world_size,
        algo_name="IBing",
    )


def validate_ring_result(result: RingSimulationResult) -> None:
    """
    对一次 Ring 模拟结果进行正确性检查。

    检查内容：
        1. reduce_scatter_steps 是否为 N - 1；
        2. allgather_steps 是否为 N - 1；
        3. total_steps 是否为 2(N - 1)；
        4. result.is_correct 是否为 True；
        5. 所有 rank 最终结果是否等于 expected_result。

    Args:
        result:
            simulate_ring_allreduce 返回的模拟结果。

    Raises:
        CorrectnessTestError:
            当任意检查不通过时抛出。
    """

    world_size = result.world_size

    expected_phase_steps = world_size - 1
    expected_total_steps = 2 * (world_size - 1)

    assert_equal(
        actual=result.reduce_scatter_steps,
        expected=expected_phase_steps,
        message=(
            f"[Ring][world_size={world_size}] "
            f"reduce_scatter_steps check failed."
        ),
    )

    assert_equal(
        actual=result.allgather_steps,
        expected=expected_phase_steps,
        message=f"[Ring][world_size={world_size}] allgather_steps check failed.",
    )

    assert_equal(
        actual=result.total_steps,
        expected=expected_total_steps,
        message=f"[Ring][world_size={world_size}] total_steps check failed.",
    )

    assert_true(
        condition=result.is_correct,
        message=f"[Ring][world_size={world_size}] result.is_correct is False.",
    )

    validate_final_workers(
        final_workers=result.final_workers,
        expected_result=result.expected_result,
        world_size=world_size,
        algo_name="Ring",
    )


def run_ibing_correctness_test(
    world_size: int,
    value_scale: int = 10,
    verbose: bool = False,
) -> IbingSimulationResult:
    """
    运行一次 IBing 正确性测试。

    Args:
        world_size:
            worker 总数 N。

        value_scale:
            初始化数据时使用的数据间隔。

        verbose:
            是否打印详细测试结果。

    Returns:
        IbingSimulationResult:
            当前 world_size 下的 IBing 模拟结果。
    """

    result = simulate_ibing_allreduce(
        world_size=world_size,
        value_scale=value_scale,
        verbose=False,
        trace=False,
    )

    validate_ibing_result(result)

    if verbose:
        print(f"[PASS] IBing correctness test passed for world_size={world_size}")
        print(f"       total_steps     = {result.total_steps}")
        print(f"       reduce_steps    = {result.reduce_steps}")
        print(f"       allgather_steps = {result.allgather_steps}")
        print(f"       expected_result = {result.expected_result}")

    return result


def run_ring_correctness_test(
    world_size: int,
    value_scale: int = 10,
    verbose: bool = False,
) -> RingSimulationResult:
    """
    运行一次 Ring 正确性测试。

    Args:
        world_size:
            worker 总数 N。

        value_scale:
            初始化数据时使用的数据间隔。

        verbose:
            是否打印详细测试结果。

    Returns:
        RingSimulationResult:
            当前 world_size 下的 Ring 模拟结果。
    """

    result = simulate_ring_allreduce(
        world_size=world_size,
        value_scale=value_scale,
        verbose=False,
        trace=False,
    )

    validate_ring_result(result)

    if verbose:
        print(f"[PASS] Ring correctness test passed for world_size={world_size}")
        print(f"       reduce_scatter_steps = {result.reduce_scatter_steps}")
        print(f"       allgather_steps      = {result.allgather_steps}")
        print(f"       total_steps          = {result.total_steps}")
        print(f"       expected_result      = {result.expected_result}")

    return result


def run_batch_tests(
    world_sizes: Iterable[int],
    algo: AlgorithmName,
    value_scale: int = 10,
    verbose: bool = False,
) -> tuple[List[IbingSimulationResult], List[RingSimulationResult]]:
    """
    批量运行多个 world_size 下的正确性测试。

    Args:
        world_sizes:
            需要测试的 worker 数量列表。

        algo:
            测试算法，可选：
                "ibing"：只测试 IBing；
                "ring"：只测试 Ring；
                "both"：同时测试 IBing 和 Ring。

        value_scale:
            初始化数据时使用的数据间隔。

        verbose:
            是否打印详细测试信息。

    Returns:
        ibing_results:
            IBing 测试结果列表。

        ring_results:
            Ring 测试结果列表。

    Raises:
        CorrectnessTestError:
            当任意测试失败时抛出。
    """

    ibing_results: List[IbingSimulationResult] = []
    ring_results: List[RingSimulationResult] = []

    for world_size in world_sizes:
        if algo in ("ibing", "both"):
            ibing_result = run_ibing_correctness_test(
                world_size=world_size,
                value_scale=value_scale,
                verbose=verbose,
            )
            ibing_results.append(ibing_result)

        if algo in ("ring", "both"):
            ring_result = run_ring_correctness_test(
                world_size=world_size,
                value_scale=value_scale,
                verbose=verbose,
            )
            ring_results.append(ring_result)

    return ibing_results, ring_results


def parse_world_sizes(args: argparse.Namespace) -> List[int]:
    """
    根据命令行参数解析需要测试的 world_size 列表。

    优先级：
        1. 如果指定 --world_size，则只测试该单个 world_size；
        2. 如果指定 --world_sizes，则测试该列表；
        3. 如果都没有指定，则使用默认列表 DEFAULT_WORLD_SIZES。

    Args:
        args:
            argparse 解析得到的命令行参数。

    Returns:
        world_sizes:
            需要测试的 worker 数量列表。
    """

    if args.world_size is not None:
        return [args.world_size]

    if args.world_sizes is not None and len(args.world_sizes) > 0:
        return args.world_sizes

    return DEFAULT_WORLD_SIZES


def print_header(
    world_sizes: List[int],
    algo: AlgorithmName,
    value_scale: int,
) -> None:
    """
    打印测试开始信息。

    Args:
        world_sizes:
            本次需要测试的 worker 数量列表。

        algo:
            当前测试算法。

        value_scale:
            初始化数据间隔。
    """

    print("=" * 80)
    print("All-Reduce Correctness Test")
    print("=" * 80)
    print(f"algo        : {algo}")
    print(f"world_sizes : {world_sizes}")
    print(f"value_scale : {value_scale}")
    print("=" * 80)


def print_ibing_summary(results: List[IbingSimulationResult]) -> None:
    """
    打印 IBing 测试结果摘要。

    Args:
        results:
            IBing 测试结果列表。
    """

    if not results:
        return

    print()
    print("-" * 80)
    print("IBing Summary")
    print("-" * 80)

    for result in results:
        print(
            f"[PASS] world_size={result.world_size}, "
            f"total_steps={result.total_steps}, "
            f"reduce_steps={result.reduce_steps}, "
            f"allgather_steps={result.allgather_steps}"
        )


def print_ring_summary(results: List[RingSimulationResult]) -> None:
    """
    打印 Ring 测试结果摘要。

    Args:
        results:
            Ring 测试结果列表。
    """

    if not results:
        return

    print()
    print("-" * 80)
    print("Ring Summary")
    print("-" * 80)

    for result in results:
        print(
            f"[PASS] world_size={result.world_size}, "
            f"total_steps={result.total_steps}, "
            f"reduce_scatter_steps={result.reduce_scatter_steps}, "
            f"allgather_steps={result.allgather_steps}"
        )


def print_step_comparison(
    ibing_results: List[IbingSimulationResult],
    ring_results: List[RingSimulationResult],
) -> None:
    """
    打印 IBing 和 Ring 的通信步数对比。

    只有当两个结果列表都非空时才打印。

    Args:
        ibing_results:
            IBing 测试结果列表。

        ring_results:
            Ring 测试结果列表。
    """

    if not ibing_results or not ring_results:
        return

    ibing_by_n = {result.world_size: result for result in ibing_results}
    ring_by_n = {result.world_size: result for result in ring_results}

    common_world_sizes = sorted(set(ibing_by_n.keys()) & set(ring_by_n.keys()))

    if not common_world_sizes:
        return

    print()
    print("-" * 80)
    print("Step Comparison")
    print("-" * 80)
    print(f"{'N':>6} | {'Ring Steps':>12} | {'IBing Steps':>12} | {'Reduction':>10}")
    print("-" * 80)

    for world_size in common_world_sizes:
        ring_steps = ring_by_n[world_size].total_steps
        ibing_steps = ibing_by_n[world_size].total_steps

        reduction = (ring_steps - ibing_steps) / ring_steps * 100.0

        print(
            f"{world_size:>6} | "
            f"{ring_steps:>12} | "
            f"{ibing_steps:>12} | "
            f"{reduction:>9.2f}%"
        )


def print_footer(
    ibing_results: List[IbingSimulationResult],
    ring_results: List[RingSimulationResult],
) -> None:
    """
    打印测试结束摘要。

    Args:
        ibing_results:
            IBing 测试结果列表。

        ring_results:
            Ring 测试结果列表。
    """

    total_cases = len(ibing_results) + len(ring_results)

    print_ibing_summary(ibing_results)
    print_ring_summary(ring_results)
    print_step_comparison(ibing_results, ring_results)

    print()
    print("=" * 80)
    print("Correctness Test Summary")
    print("=" * 80)
    print(f"Total passed cases: {total_cases}")
    print("[PASS] All selected correctness tests passed.")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace:
            命令行参数对象。
    """

    parser = argparse.ArgumentParser(
        description="Run correctness tests for Ring and IBing single-process simulations."
    )

    parser.add_argument(
        "--algo",
        type=str,
        choices=["ibing", "ring", "both"],
        default="both",
        help=(
            "Algorithm to test. "
            "Choices: ibing, ring, both. "
            "Default: both."
        ),
    )

    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help=(
            "Test a single world_size. "
            "Example: --world_size 5"
        ),
    )

    parser.add_argument(
        "--world_sizes",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Test multiple world sizes. "
            "Example: --world_sizes 3 4 5 6 7 8"
        ),
    )

    parser.add_argument(
        "--value_scale",
        type=int,
        default=10,
        help=(
            "Scale used to initialize worker data. "
            "Default: workers[rank][chunk] = rank * 10 + chunk."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each world_size.",
    )

    return parser.parse_args()


def main() -> None:
    """
    程序入口。

    执行流程：
        1. 解析命令行参数；
        2. 确定测试算法和 world_size 列表；
        3. 批量运行 IBing / Ring 正确性测试；
        4. 如果全部通过，打印 PASS；
        5. 如果任意失败，打印错误并返回非 0 退出码。
    """

    args = parse_args()

    algo: AlgorithmName = args.algo
    world_sizes = parse_world_sizes(args)

    print_header(
        world_sizes=world_sizes,
        algo=algo,
        value_scale=args.value_scale,
    )

    try:
        ibing_results, ring_results = run_batch_tests(
            world_sizes=world_sizes,
            algo=algo,
            value_scale=args.value_scale,
            verbose=args.verbose,
        )
    except Exception as exc:
        print("[FAIL] Correctness test failed.")
        print(f"Reason: {exc}")
        raise SystemExit(1) from exc

    print_footer(
        ibing_results=ibing_results,
        ring_results=ring_results,
    )


if __name__ == "__main__":
    main()