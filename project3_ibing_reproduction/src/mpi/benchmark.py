#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/mpi/benchmark.py

文件作用：
    该文件用于 benchmark MPI 版 All-Reduce 算法的通信时间。

当前支持算法：
    1. ring
        MPI 版标准 Ring All-Reduce，对应 src/mpi/ring_mpi.py

    2. ibing
        MPI 版 IBing All-Reduce，对应 src/mpi/ibing_mpi.py

    3. mpi_allreduce
        MPI 自带 Allreduce，作为参考基线。

    4. all
        同时测试 ring、ibing、mpi_allreduce。

运行方式：
    Windows / MS-MPI:
        mpiexec -n 5 python src/mpi/benchmark.py --algo all --data_sizes_mb 1 10 --repeat 30 --warmup 5

    Linux / OpenMPI:
        mpirun -np 5 python src/mpi/benchmark.py --algo all --data_sizes_mb 1 10 --repeat 30 --warmup 5

保存结果：
        mpiexec -n 5 python src/mpi/benchmark.py ^
            --algo all ^
            --data_sizes_mb 1 10 50 ^
            --repeat 30 ^
            --warmup 5 ^
            --output results/raw/mpi_benchmark.csv

说明：
    本文件测试的是“真实 MPI 多进程通信时间”，不是单进程模拟时间。

    计时方式：
        每次通信前后使用 comm.Barrier() 同步；
        每个 rank 得到本地耗时；
        使用 MPI.MAX 取所有 rank 中最大的耗时作为该轮耗时。

    原因：
        All-Reduce 的完成时间取决于最慢 rank。
        因此使用 max elapsed time 更符合分布式通信整体耗时定义。

注意：
    1. data_size_mb 表示每个 rank 的总数据大小，单位按 MiB 计算，即 1 MiB = 1024 * 1024 bytes。
    2. 每个 rank 的数据会被切成 world_size 个 chunk。
    3. chunk_size 会根据 data_size_mb、world_size 和 dtype 自动计算。
    4. 如果显式传入 --chunk_size，则优先使用 chunk_size，忽略 data_size_mb 自动计算。
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional

import numpy as np
from mpi4py import MPI


# ---------------------------------------------------------------------
# 路径处理
#
# 当前文件位于：
#     src/mpi/benchmark.py
#
# 为了让脚本无论从项目根目录还是其他目录运行，都能正确导入：
#     src/mpi/ibing_mpi.py
#     src/mpi/ring_mpi.py
# ---------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
MPI_DIR = CURRENT_FILE.parent
SRC_DIR = MPI_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

for path in (PROJECT_ROOT, SRC_DIR, MPI_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


try:
    from ibing_mpi import ibing_allreduce_inplace
    from ibing_mpi import get_total_steps as get_ibing_total_steps
    from ring_mpi import ring_allreduce_inplace
    from ring_mpi import get_total_steps as get_ring_total_steps
except ImportError:
    from src.mpi.ibing_mpi import ibing_allreduce_inplace
    from src.mpi.ibing_mpi import get_total_steps as get_ibing_total_steps
    from src.mpi.ring_mpi import ring_allreduce_inplace
    from src.mpi.ring_mpi import get_total_steps as get_ring_total_steps


AlgorithmName = Literal["ring", "ibing", "mpi_allreduce", "all"]


@dataclass(frozen=True)
class BenchmarkCase:
    """
    表示一个 benchmark 测试用例。

    Attributes:
        algo:
            算法名称。

        world_size:
            MPI 进程数。

        data_size_mb:
            每个 rank 的目标总数据大小，单位 MiB。
            如果用户显式指定 chunk_size，则该值主要用于记录。

        actual_data_size_mb:
            根据 chunk_size 反算出的实际数据大小，单位 MiB。

        chunk_size:
            每个 chunk 中的元素数量。

        dtype_name:
            numpy 数据类型名称，例如 float32 或 float64。

        warmup:
            预热轮数，不计入最终统计。

        repeat:
            正式计时轮数。

        value_scale:
            初始化数据使用的 rank 间隔。
    """

    algo: str
    world_size: int
    data_size_mb: float
    actual_data_size_mb: float
    chunk_size: int
    dtype_name: str
    warmup: int
    repeat: int
    value_scale: int


@dataclass(frozen=True)
class BenchmarkResult:
    """
    保存一个 benchmark 测试结果。

    Attributes:
        case:
            对应的测试用例。

        step_count:
            当前算法的理论通信步数。
            mpi_allreduce 不按 Ring 步数建模，记为 -1。

        correct:
            正确性检查是否通过。

        times_ms:
            每轮正式测试的耗时，单位 ms。
            每轮耗时使用所有 rank 的 max elapsed time。

        mean_ms:
            平均耗时。

        std_ms:
            标准差。

        min_ms:
            最小耗时。

        max_ms:
            最大耗时。
    """

    case: BenchmarkCase
    step_count: int
    correct: bool
    times_ms: List[float]
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, int | float | str | bool]:
        """
        转换为字典，便于保存 CSV。

        Returns:
            当前 benchmark 结果的字典形式。
        """

        return {
            "algo": self.case.algo,
            "world_size": self.case.world_size,
            "data_size_mb_target": self.case.data_size_mb,
            "data_size_mb_actual": self.case.actual_data_size_mb,
            "chunk_size": self.case.chunk_size,
            "dtype": self.case.dtype_name,
            "warmup": self.case.warmup,
            "repeat": self.case.repeat,
            "value_scale": self.case.value_scale,
            "step_count": self.step_count,
            "correct": self.correct,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


def validate_world_size(world_size: int) -> None:
    """
    检查 MPI 进程数是否合法。

    Args:
        world_size:
            MPI 进程数。

    Raises:
        ValueError:
            当 world_size 小于 2 时抛出异常。
    """

    if world_size < 2:
        raise ValueError(f"world_size must be >= 2, but got {world_size}.")


def validate_chunk_size(chunk_size: int) -> None:
    """
    检查 chunk_size 是否合法。

    Args:
        chunk_size:
            每个 chunk 的元素数量。

    Raises:
        ValueError:
            当 chunk_size 小于 1 时抛出异常。
    """

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, but got {chunk_size}.")


def validate_repeat_and_warmup(repeat: int, warmup: int) -> None:
    """
    检查 repeat 和 warmup 是否合法。

    Args:
        repeat:
            正式测试轮数。

        warmup:
            预热轮数。

    Raises:
        ValueError:
            当 repeat 小于 1 或 warmup 小于 0 时抛出。
    """

    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, but got {repeat}.")

    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, but got {warmup}.")


def parse_dtype(dtype_name: str) -> np.dtype:
    """
    将字符串转换为 numpy dtype。

    Args:
        dtype_name:
            数据类型名称。

    Returns:
        numpy dtype。
    """

    if dtype_name == "float32":
        return np.float32

    if dtype_name == "float64":
        return np.float64

    raise ValueError(
        f"Unsupported dtype: {dtype_name}. "
        f"Supported dtypes: float32, float64."
    )


def dtype_nbytes(dtype: np.dtype) -> int:
    """
    返回 dtype 的字节数。

    Args:
        dtype:
            numpy dtype。

    Returns:
        每个元素占用的字节数。
    """

    return int(np.dtype(dtype).itemsize)


def compute_chunk_size_from_data_size(
    data_size_mb: float,
    world_size: int,
    dtype: np.dtype,
) -> int:
    """
    根据目标数据大小计算每个 chunk 的元素数量。

    每个 rank 的总数据形状为：

        [world_size, chunk_size]

    因此每个 rank 的总元素数量为：

        world_size * chunk_size

    每个 rank 的总字节数为：

        world_size * chunk_size * dtype_nbytes

    所以：

        chunk_size =
            floor(data_size_bytes / (world_size * dtype_nbytes))

    Args:
        data_size_mb:
            每个 rank 的目标数据大小，单位 MiB。

        world_size:
            MPI 进程数。

        dtype:
            numpy dtype。

    Returns:
        每个 chunk 的元素数量。
    """

    if data_size_mb <= 0:
        raise ValueError(f"data_size_mb must be > 0, but got {data_size_mb}.")

    validate_world_size(world_size)

    bytes_per_rank = data_size_mb * 1024 * 1024
    bytes_per_value = dtype_nbytes(dtype)

    chunk_size = int(bytes_per_rank // (world_size * bytes_per_value))
    chunk_size = max(chunk_size, 1)

    return chunk_size


def compute_actual_data_size_mb(
    world_size: int,
    chunk_size: int,
    dtype: np.dtype,
) -> float:
    """
    根据实际 chunk_size 计算每个 rank 的实际数据大小。

    Args:
        world_size:
            MPI 进程数。

        chunk_size:
            每个 chunk 中的元素数量。

        dtype:
            numpy dtype。

    Returns:
        每个 rank 的实际数据大小，单位 MiB。
    """

    validate_world_size(world_size)
    validate_chunk_size(chunk_size)

    total_bytes = world_size * chunk_size * dtype_nbytes(dtype)
    return total_bytes / (1024 * 1024)


def resolve_algorithms(algo: AlgorithmName) -> List[str]:
    """
    将命令行算法选项解析为实际算法列表。

    Args:
        algo:
            "ring"、"ibing"、"mpi_allreduce" 或 "all"。

    Returns:
        需要测试的算法列表。
    """

    if algo == "all":
        return ["ring", "ibing", "mpi_allreduce"]

    return [algo]


def create_initial_chunks(
    rank: int,
    world_size: int,
    chunk_size: int,
    value_scale: int,
    dtype: np.dtype,
) -> np.ndarray:
    """
    初始化当前 rank 的数据块。

    数据形状：

        [world_size, chunk_size]

    初始化规则：

        chunks[chunk_id, :] = rank * value_scale + chunk_id

    Args:
        rank:
            当前 rank。

        world_size:
            MPI 进程数。

        chunk_size:
            每个 chunk 的元素数量。

        value_scale:
            初始化 rank 间隔。

        dtype:
            numpy dtype。

    Returns:
        当前 rank 的初始化 chunks。
    """

    chunks = np.empty((world_size, chunk_size), dtype=dtype)

    for chunk_id in range(world_size):
        chunks[chunk_id, :] = rank * value_scale + chunk_id

    return chunks


def compute_expected_chunks(
    world_size: int,
    chunk_size: int,
    value_scale: int,
    dtype: np.dtype,
) -> np.ndarray:
    """
    计算理论 All-Reduce 结果。

    对于 chunk_id：

        expected[chunk_id, :] =
            sum(rank * value_scale + chunk_id for rank in range(world_size))

    Args:
        world_size:
            MPI 进程数。

        chunk_size:
            每个 chunk 的元素数量。

        value_scale:
            初始化 rank 间隔。

        dtype:
            numpy dtype。

    Returns:
        理论 All-Reduce 结果。
    """

    expected = np.empty((world_size, chunk_size), dtype=dtype)

    for chunk_id in range(world_size):
        value = 0

        for rank in range(world_size):
            value += rank * value_scale + chunk_id

        expected[chunk_id, :] = value

    return expected


def run_mpi_allreduce_inplace(
    chunks: np.ndarray,
    comm: MPI.Comm,
) -> np.ndarray:
    """
    使用 MPI 自带 Allreduce 执行全局求和。

    输入 chunks 的形状为：

        [world_size, chunk_size]

    MPI_Allreduce 直接对整个二维数组展平后的连续内存做 SUM。
    最终结果再 reshape 回原形状。

    Args:
        chunks:
            当前 rank 的本地数据。

        comm:
            MPI 通信子。

    Returns:
        All-Reduce 后的 chunks。
    """

    send_buf = np.ascontiguousarray(chunks.reshape(-1))
    recv_buf = np.empty_like(send_buf)

    comm.Allreduce(
        send_buf,
        recv_buf,
        op=MPI.SUM,
    )

    chunks[:, :] = recv_buf.reshape(chunks.shape)

    return chunks


def run_algorithm_inplace(
    algo: str,
    chunks: np.ndarray,
    comm: MPI.Comm,
) -> np.ndarray:
    """
    根据 algo 运行对应 All-Reduce 算法。

    Args:
        algo:
            算法名称，ring / ibing / mpi_allreduce。

        chunks:
            当前 rank 的本地数据，会被原地更新。

        comm:
            MPI 通信子。

    Returns:
        All-Reduce 后的 chunks。
    """

    if algo == "ring":
        return ring_allreduce_inplace(chunks, comm)

    if algo == "ibing":
        return ibing_allreduce_inplace(chunks, comm)

    if algo == "mpi_allreduce":
        return run_mpi_allreduce_inplace(chunks, comm)

    raise ValueError(f"Unsupported algo: {algo}")


def get_step_count(algo: str, world_size: int) -> int:
    """
    返回某个算法的理论通信步数。

    Args:
        algo:
            算法名称。

        world_size:
            MPI 进程数。

    Returns:
        理论通信步数。
        对 mpi_allreduce 返回 -1，因为其内部算法由 MPI 实现决定。
    """

    if algo == "ring":
        return get_ring_total_steps(world_size)

    if algo == "ibing":
        return get_ibing_total_steps(world_size)

    if algo == "mpi_allreduce":
        return -1

    raise ValueError(f"Unsupported algo: {algo}")


def check_correctness_once(
    algo: str,
    world_size: int,
    chunk_size: int,
    value_scale: int,
    dtype: np.dtype,
    comm: MPI.Comm,
) -> bool:
    """
    对某个算法执行一次正确性检查。

    Args:
        algo:
            算法名称。

        world_size:
            MPI 进程数。

        chunk_size:
            每个 chunk 中的元素数量。

        value_scale:
            初始化 rank 间隔。

        dtype:
            numpy dtype。

        comm:
            MPI 通信子。

    Returns:
        True 表示所有 rank 均正确；
        False 表示至少一个 rank 错误。
    """

    rank = comm.Get_rank()

    chunks = create_initial_chunks(
        rank=rank,
        world_size=world_size,
        chunk_size=chunk_size,
        value_scale=value_scale,
        dtype=dtype,
    )

    expected = compute_expected_chunks(
        world_size=world_size,
        chunk_size=chunk_size,
        value_scale=value_scale,
        dtype=dtype,
    )

    comm.Barrier()

    run_algorithm_inplace(
        algo=algo,
        chunks=chunks,
        comm=comm,
    )

    comm.Barrier()

    local_correct = bool(np.allclose(chunks, expected, atol=1e-6, rtol=0.0))
    correct_count = comm.allreduce(int(local_correct), op=MPI.SUM)

    return correct_count == world_size


def run_one_timed_iteration(
    algo: str,
    world_size: int,
    chunk_size: int,
    value_scale: int,
    dtype: np.dtype,
    comm: MPI.Comm,
) -> float:
    """
    运行一次正式或预热迭代，并返回所有 rank 中最大的耗时。

    Args:
        algo:
            算法名称。

        world_size:
            MPI 进程数。

        chunk_size:
            每个 chunk 中的元素数量。

        value_scale:
            初始化 rank 间隔。

        dtype:
            numpy dtype。

        comm:
            MPI 通信子。

    Returns:
        本轮通信耗时，单位 ms。
        返回值为所有 rank 本地耗时中的最大值。
    """

    rank = comm.Get_rank()

    chunks = create_initial_chunks(
        rank=rank,
        world_size=world_size,
        chunk_size=chunk_size,
        value_scale=value_scale,
        dtype=dtype,
    )

    comm.Barrier()
    start_time = time.perf_counter()

    run_algorithm_inplace(
        algo=algo,
        chunks=chunks,
        comm=comm,
    )

    comm.Barrier()
    elapsed_seconds_local = time.perf_counter() - start_time

    elapsed_seconds_global = comm.allreduce(
        elapsed_seconds_local,
        op=MPI.MAX,
    )

    return elapsed_seconds_global * 1000.0


def run_benchmark_case(
    case: BenchmarkCase,
    dtype: np.dtype,
    comm: MPI.Comm,
    skip_check: bool = False,
) -> BenchmarkResult:
    """
    运行一个 benchmark 测试用例。

    执行流程：
        1. 可选正确性检查；
        2. warmup 轮预热；
        3. repeat 轮正式计时；
        4. 统计 mean/std/min/max。

    Args:
        case:
            benchmark 用例。

        dtype:
            numpy dtype。

        comm:
            MPI 通信子。

        skip_check:
            是否跳过正确性检查。

    Returns:
        BenchmarkResult。
    """

    correct = True

    if not skip_check:
        correct = check_correctness_once(
            algo=case.algo,
            world_size=case.world_size,
            chunk_size=case.chunk_size,
            value_scale=case.value_scale,
            dtype=dtype,
            comm=comm,
        )

    for _ in range(case.warmup):
        run_one_timed_iteration(
            algo=case.algo,
            world_size=case.world_size,
            chunk_size=case.chunk_size,
            value_scale=case.value_scale,
            dtype=dtype,
            comm=comm,
        )

    times_ms: List[float] = []

    for _ in range(case.repeat):
        elapsed_ms = run_one_timed_iteration(
            algo=case.algo,
            world_size=case.world_size,
            chunk_size=case.chunk_size,
            value_scale=case.value_scale,
            dtype=dtype,
            comm=comm,
        )
        times_ms.append(elapsed_ms)

    mean_ms = statistics.mean(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) >= 2 else 0.0
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    step_count = get_step_count(
        algo=case.algo,
        world_size=case.world_size,
    )

    return BenchmarkResult(
        case=case,
        step_count=step_count,
        correct=correct,
        times_ms=times_ms,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
    )


def build_cases(
    algorithms: Iterable[str],
    data_sizes_mb: Iterable[float],
    explicit_chunk_size: Optional[int],
    world_size: int,
    dtype: np.dtype,
    dtype_name: str,
    warmup: int,
    repeat: int,
    value_scale: int,
) -> List[BenchmarkCase]:
    """
    构造 benchmark 测试用例。

    Args:
        algorithms:
            算法列表。

        data_sizes_mb:
            每个 rank 的目标数据大小列表。

        explicit_chunk_size:
            如果用户显式指定 chunk_size，则使用该值。

        world_size:
            MPI 进程数。

        dtype:
            numpy dtype。

        dtype_name:
            dtype 名称。

        warmup:
            预热轮数。

        repeat:
            正式计时轮数。

        value_scale:
            初始化 rank 间隔。

    Returns:
        BenchmarkCase 列表。
    """

    cases: List[BenchmarkCase] = []

    for data_size_mb in data_sizes_mb:
        if explicit_chunk_size is not None:
            chunk_size = explicit_chunk_size
        else:
            chunk_size = compute_chunk_size_from_data_size(
                data_size_mb=data_size_mb,
                world_size=world_size,
                dtype=dtype,
            )

        validate_chunk_size(chunk_size)

        actual_data_size_mb = compute_actual_data_size_mb(
            world_size=world_size,
            chunk_size=chunk_size,
            dtype=dtype,
        )

        for algo in algorithms:
            cases.append(
                BenchmarkCase(
                    algo=algo,
                    world_size=world_size,
                    data_size_mb=data_size_mb,
                    actual_data_size_mb=actual_data_size_mb,
                    chunk_size=chunk_size,
                    dtype_name=dtype_name,
                    warmup=warmup,
                    repeat=repeat,
                    value_scale=value_scale,
                )
            )

    return cases


def print_result_table(results: List[BenchmarkResult]) -> None:
    """
    打印 benchmark 结果表。

    Args:
        results:
            benchmark 结果列表。
    """

    if not results:
        return

    print()
    print("=" * 120)
    print("MPI All-Reduce Benchmark Results")
    print("=" * 120)

    header = (
        f"{'Algo':>14} | "
        f"{'N':>4} | "
        f"{'Data(MiB)':>10} | "
        f"{'ChunkSize':>10} | "
        f"{'Steps':>7} | "
        f"{'Mean(ms)':>10} | "
        f"{'Std(ms)':>10} | "
        f"{'Min(ms)':>10} | "
        f"{'Max(ms)':>10} | "
        f"{'Correct':>8}"
    )

    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result.case.algo:>14} | "
            f"{result.case.world_size:>4} | "
            f"{result.case.actual_data_size_mb:>10.3f} | "
            f"{result.case.chunk_size:>10} | "
            f"{result.step_count:>7} | "
            f"{result.mean_ms:>10.3f} | "
            f"{result.std_ms:>10.3f} | "
            f"{result.min_ms:>10.3f} | "
            f"{result.max_ms:>10.3f} | "
            f"{str(result.correct):>8}"
        )

    print("=" * 120)


def print_speedup_summary(results: List[BenchmarkResult]) -> None:
    """
    打印 Ring 与 IBing 的简单加速比和优化率。

    只有同一 data_size 下同时存在 ring 与 ibing 结果时才打印。

    Args:
        results:
            benchmark 结果列表。
    """

    by_data_size: dict[float, dict[str, BenchmarkResult]] = {}

    for result in results:
        key = result.case.actual_data_size_mb
        by_data_size.setdefault(key, {})
        by_data_size[key][result.case.algo] = result

    rows = []

    for data_size, algo_results in sorted(by_data_size.items()):
        if "ring" in algo_results and "ibing" in algo_results:
            ring_time = algo_results["ring"].mean_ms
            ibing_time = algo_results["ibing"].mean_ms

            speedup = ring_time / ibing_time
            opt_rate = (ring_time - ibing_time) / ring_time * 100.0

            rows.append((data_size, ring_time, ibing_time, speedup, opt_rate))

    if not rows:
        return

    print()
    print("=" * 100)
    print("Ring vs IBing Speedup Summary")
    print("=" * 100)

    header = (
        f"{'Data(MiB)':>10} | "
        f"{'Ring Mean(ms)':>14} | "
        f"{'IBing Mean(ms)':>15} | "
        f"{'Speedup':>10} | "
        f"{'OptRate':>10}"
    )

    print(header)
    print("-" * len(header))

    for data_size, ring_time, ibing_time, speedup, opt_rate in rows:
        print(
            f"{data_size:>10.3f} | "
            f"{ring_time:>14.3f} | "
            f"{ibing_time:>15.3f} | "
            f"{speedup:>9.3f}x | "
            f"{opt_rate:>9.2f}%"
        )

    print("=" * 100)


def save_results_to_csv(
    results: List[BenchmarkResult],
    output_path: str | Path,
) -> None:
    """
    保存 benchmark 结果到 CSV。

    Args:
        results:
            benchmark 结果列表。

        output_path:
            CSV 输出路径。
    """

    output_path = Path(output_path)

    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "algo",
        "world_size",
        "data_size_mb_target",
        "data_size_mb_actual",
        "chunk_size",
        "dtype",
        "warmup",
        "repeat",
        "value_scale",
        "step_count",
        "correct",
        "mean_ms",
        "std_ms",
        "min_ms",
        "max_ms",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result.as_dict())

    print(f"[OK] Benchmark results saved to: {output_path}")


def parse_data_sizes(args: argparse.Namespace) -> List[float]:
    """
    解析 data_size 参数。

    Args:
        args:
            命令行参数。

    Returns:
        data_size_mb 列表。
    """

    if args.data_sizes_mb is not None and len(args.data_sizes_mb) > 0:
        return args.data_sizes_mb

    if args.data_size_mb is not None:
        return [args.data_size_mb]

    return [1.0]


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace。
    """

    parser = argparse.ArgumentParser(
        description="Benchmark MPI implementations of Ring, IBing, and MPI_Allreduce."
    )

    parser.add_argument(
        "--algo",
        type=str,
        choices=["ring", "ibing", "mpi_allreduce", "all"],
        default="all",
        help="Algorithm to benchmark. Default: all.",
    )

    parser.add_argument(
        "--data_size_mb",
        type=float,
        default=None,
        help="Single target data size per rank in MiB.",
    )

    parser.add_argument(
        "--data_sizes_mb",
        type=float,
        nargs="*",
        default=None,
        help="Multiple target data sizes per rank in MiB. Example: --data_sizes_mb 1 10 50",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help=(
            "Explicit number of elements per chunk. "
            "If set, this overrides --data_size_mb / --data_sizes_mb."
        ),
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
        help="Data type. Default: float32.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations. Default: 5.",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=30,
        help="Number of timed iterations. Default: 30.",
    )

    parser.add_argument(
        "--value_scale",
        type=int,
        default=10,
        help="Scale used to initialize worker data. Default: 10.",
    )

    parser.add_argument(
        "--skip_check",
        action="store_true",
        help="Skip correctness check before benchmark.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV output path. Example: --output results/raw/mpi_benchmark.csv",
    )

    return parser.parse_args()


def main() -> None:
    """
    程序入口。

    执行流程：
        1. 获取 MPI rank/world_size；
        2. 解析参数；
        3. 构造 benchmark cases；
        4. 每个 rank 同步执行 benchmark；
        5. rank 0 打印结果并保存 CSV。
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    args = parse_args()

    validate_world_size(world_size)
    validate_repeat_and_warmup(
        repeat=args.repeat,
        warmup=args.warmup,
    )

    dtype = parse_dtype(args.dtype)

    if args.chunk_size is not None:
        validate_chunk_size(args.chunk_size)

    algorithms = resolve_algorithms(args.algo)
    data_sizes_mb = parse_data_sizes(args)

    cases = build_cases(
        algorithms=algorithms,
        data_sizes_mb=data_sizes_mb,
        explicit_chunk_size=args.chunk_size,
        world_size=world_size,
        dtype=dtype,
        dtype_name=args.dtype,
        warmup=args.warmup,
        repeat=args.repeat,
        value_scale=args.value_scale,
    )

    if rank == 0:
        print("=" * 120)
        print("MPI All-Reduce Benchmark")
        print("=" * 120)
        print(f"world_size     : {world_size}")
        print(f"algorithms     : {algorithms}")
        print(f"data_sizes_mb  : {data_sizes_mb}")
        print(f"chunk_size     : {args.chunk_size if args.chunk_size is not None else 'auto'}")
        print(f"dtype          : {args.dtype}")
        print(f"warmup         : {args.warmup}")
        print(f"repeat         : {args.repeat}")
        print(f"skip_check     : {args.skip_check}")
        print(f"total cases    : {len(cases)}")
        print("=" * 120)

    results: List[BenchmarkResult] = []

    for index, case in enumerate(cases, start=1):
        if rank == 0:
            print(
                f"[{index}/{len(cases)}] "
                f"algo={case.algo}, "
                f"data={case.actual_data_size_mb:.3f} MiB, "
                f"chunk_size={case.chunk_size} ... ",
                end="",
                flush=True,
            )

        result = run_benchmark_case(
            case=case,
            dtype=dtype,
            comm=comm,
            skip_check=args.skip_check,
        )

        results.append(result)

        if rank == 0:
            status = "PASS" if result.correct else "FAIL"
            print(
                f"{status}, "
                f"mean={result.mean_ms:.3f} ms, "
                f"std={result.std_ms:.3f} ms"
            )

    if rank == 0:
        print_result_table(results)
        print_speedup_summary(results)

        if args.output is not None:
            save_results_to_csv(
                results=results,
                output_path=args.output,
            )

    any_failed = any(not result.correct for result in results)
    failed_flag = comm.allreduce(int(any_failed), op=MPI.SUM)

    if failed_flag > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()