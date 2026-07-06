#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/mpi/ring_mpi.py

文件作用：
    该文件用于实现标准 Ring All-Reduce 的 MPI 多进程版本。

    它是 IBing 论文复现中的 MPI baseline，用于后续与 MPI 版 IBing 对比。

    标准 Ring All-Reduce 分为两个阶段：
        1. Reduce-Scatter 阶段：
            每个 rank 向右邻居发送一个 chunk；
            每个 rank 从左邻居接收一个 chunk；
            收到后执行加法 reduce。

        2. All-Gather 阶段：
            每个 rank 向右邻居发送一个已经完成聚合的 chunk；
            每个 rank 从左邻居接收一个 chunk；
            收到后直接保存，不再执行加法。

    对于 N 个 worker：
        Reduce-Scatter 步数 = N - 1
        All-Gather 步数     = N - 1
        总通信步数          = 2(N - 1)

运行示例：
    Windows / MS-MPI:
        mpiexec -n 5 python src/mpi/ring_mpi.py --chunk_size 4 --verbose

    Linux / WSL / OpenMPI:
        mpirun -np 5 python src/mpi/ring_mpi.py --chunk_size 4 --verbose

    只做正确性检查：
        mpiexec -n 5 python src/mpi/ring_mpi.py --chunk_size 1024 --check

初始化规则：
    每个 rank 持有一个二维数组：

        chunks[chunk_id, element_id]

    其中：
        chunk_id 取值范围为 [0, world_size - 1]
        element_id 取值范围为 [0, chunk_size - 1]

    默认初始化为：

        chunks[chunk_id, :] = rank * value_scale + chunk_id

    例如 world_size=5, chunk_size=4, value_scale=10 时：

        rank 0:
            chunk 0 = [0, 0, 0, 0]
            chunk 1 = [1, 1, 1, 1]
            ...

        rank 1:
            chunk 0 = [10, 10, 10, 10]
            chunk 1 = [11, 11, 11, 11]
            ...

    理论 All-Reduce 结果：
        expected[chunk_id, :] =
            sum(rank * value_scale + chunk_id for rank in all ranks)

    当 world_size=5, value_scale=10 时：
        chunk 0 = 0 + 10 + 20 + 30 + 40 = 100
        chunk 1 = 1 + 11 + 21 + 31 + 41 = 105
        chunk 2 = 2 + 12 + 22 + 32 + 42 = 110
        chunk 3 = 3 + 13 + 23 + 33 + 43 = 115
        chunk 4 = 4 + 14 + 24 + 34 + 44 = 120
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List

import numpy as np
from mpi4py import MPI


@dataclass(frozen=True)
class NeighborInfo:
    """
    保存当前 rank 的左右邻居信息。

    Attributes:
        rank:
            当前 MPI 进程编号。

        left:
            当前 rank 的左邻居编号。

        right:
            当前 rank 的右邻居编号。
    """

    rank: int
    left: int
    right: int


@dataclass(frozen=True)
class RingStepSchedule:
    """
    保存当前 rank 在某个 Ring step 中的通信调度。

    Attributes:
        rank:
            当前 MPI 进程编号。

        step:
            当前阶段内部 step，0-based。

        phase:
            当前阶段，取值为 "Reduce-Scatter" 或 "All-Gather"。

        left:
            当前 rank 的左邻居。

        right:
            当前 rank 的右邻居。

        send_chunk:
            当前 rank 需要发送给右邻居的 chunk 编号。

        recv_chunk:
            当前 rank 需要从左邻居接收的 chunk 编号。
    """

    rank: int
    step: int
    phase: str
    left: int
    right: int
    send_chunk: int
    recv_chunk: int


@dataclass(frozen=True)
class RingRunResult:
    """
    保存一次 MPI 版 Ring All-Reduce 的运行结果。

    Attributes:
        rank:
            当前进程 rank。

        world_size:
            MPI 总进程数。

        chunk_size:
            每个 chunk 中的元素数量。

        reduce_scatter_steps:
            Reduce-Scatter 阶段通信步数，即 N - 1。

        allgather_steps:
            All-Gather 阶段通信步数，即 N - 1。

        total_steps:
            总通信步数，即 2(N - 1)。

        elapsed_seconds:
            当前 rank 执行 Ring All-Reduce 的耗时。

        local_correct:
            当前 rank 的最终结果是否正确。

        global_correct:
            所有 rank 的最终结果是否全部正确。
    """

    rank: int
    world_size: int
    chunk_size: int
    reduce_scatter_steps: int
    allgather_steps: int
    total_steps: int
    elapsed_seconds: float
    local_correct: bool
    global_correct: bool


def validate_world_size(world_size: int) -> None:
    """
    检查 MPI 进程数是否合法。

    Args:
        world_size:
            MPI 总进程数。

    Raises:
        ValueError:
            当 world_size 小于 2 时抛出异常。
    """

    if world_size < 2:
        raise ValueError("world_size must be >= 2.")


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
        raise ValueError("chunk_size must be >= 1.")


def get_neighbors(rank: int, world_size: int) -> NeighborInfo:
    """
    计算当前 rank 在环形拓扑中的左右邻居。

    对于 rank=r：

        left(r) = (r - 1 + N) % N

        right(r) = (r + 1) % N

    Args:
        rank:
            当前 MPI 进程编号。

        world_size:
            MPI 总进程数 N。

    Returns:
        NeighborInfo:
            当前 rank 的左右邻居。
    """

    validate_world_size(world_size)

    left = (rank - 1 + world_size) % world_size
    right = (rank + 1) % world_size

    return NeighborInfo(rank=rank, left=left, right=right)


def get_reduce_scatter_steps(world_size: int) -> int:
    """
    返回 Ring All-Reduce 的 Reduce-Scatter 阶段步数。

    对于 N 个 worker：

        Reduce-Scatter steps = N - 1

    Args:
        world_size:
            worker 总数 N。

    Returns:
        Reduce-Scatter 阶段通信步数。
    """

    validate_world_size(world_size)
    return world_size - 1


def get_allgather_steps(world_size: int) -> int:
    """
    返回 Ring All-Reduce 的 All-Gather 阶段步数。

    对于 N 个 worker：

        All-Gather steps = N - 1

    Args:
        world_size:
            worker 总数 N。

    Returns:
        All-Gather 阶段通信步数。
    """

    validate_world_size(world_size)
    return world_size - 1


def get_total_steps(world_size: int) -> int:
    """
    返回 Ring All-Reduce 的总通信步数。

    对于 N 个 worker：

        total_steps = 2(N - 1)

    Args:
        world_size:
            worker 总数 N。

    Returns:
        Ring All-Reduce 总通信步数。
    """

    validate_world_size(world_size)
    return 2 * (world_size - 1)


def ring_reduce_scatter_schedule(
    rank: int,
    step: int,
    world_size: int,
) -> RingStepSchedule:
    """
    计算 Reduce-Scatter 阶段当前 rank 的通信调度。

    标准 Ring All-Reduce 在 Reduce-Scatter 阶段中，每个 rank：

        向右邻居发送 send_chunk；
        从左邻居接收 recv_chunk；
        收到后执行加法。

    对于 rank=r，step=i：

        send_chunk = (r - i + N) % N

        recv_chunk = (r - i - 1 + N) % N

    Args:
        rank:
            当前 MPI 进程编号。

        step:
            Reduce-Scatter 阶段内部 step，0-based。

        world_size:
            MPI 总进程数 N。

    Returns:
        RingStepSchedule:
            当前 rank 当前 step 的通信调度。
    """

    validate_world_size(world_size)

    if step < 0 or step >= world_size - 1:
        raise ValueError(
            f"step must be in [0, {world_size - 2}], but got step={step}."
        )

    neighbors = get_neighbors(rank, world_size)

    send_chunk = (rank - step + world_size) % world_size
    recv_chunk = (rank - step - 1 + world_size) % world_size

    return RingStepSchedule(
        rank=rank,
        step=step,
        phase="Reduce-Scatter",
        left=neighbors.left,
        right=neighbors.right,
        send_chunk=send_chunk,
        recv_chunk=recv_chunk,
    )


def ring_allgather_schedule(
    rank: int,
    step: int,
    world_size: int,
) -> RingStepSchedule:
    """
    计算 All-Gather 阶段当前 rank 的通信调度。

    Reduce-Scatter 结束后，每个 rank 持有一个已经完成全局 reduce 的 chunk。
    All-Gather 阶段负责把这些完整 chunk 沿环传播给所有 rank。

    对于 rank=r，step=i：

        send_chunk = (r - i + 1 + N) % N

        recv_chunk = (r - i + N) % N

    Args:
        rank:
            当前 MPI 进程编号。

        step:
            All-Gather 阶段内部 step，0-based。

        world_size:
            MPI 总进程数 N。

    Returns:
        RingStepSchedule:
            当前 rank 当前 step 的通信调度。
    """

    validate_world_size(world_size)

    if step < 0 or step >= world_size - 1:
        raise ValueError(
            f"step must be in [0, {world_size - 2}], but got step={step}."
        )

    neighbors = get_neighbors(rank, world_size)

    send_chunk = (rank - step + 1 + world_size) % world_size
    recv_chunk = (rank - step + world_size) % world_size

    return RingStepSchedule(
        rank=rank,
        step=step,
        phase="All-Gather",
        left=neighbors.left,
        right=neighbors.right,
        send_chunk=send_chunk,
        recv_chunk=recv_chunk,
    )


def create_initial_chunks(
    rank: int,
    world_size: int,
    chunk_size: int,
    value_scale: int = 10,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    为当前 rank 初始化本地 chunks。

    数据形状为：

        [world_size, chunk_size]

    其中：
        chunks[chunk_id, :] 表示当前 rank 保存的第 chunk_id 个数据块。

    初始化规则：

        chunks[chunk_id, :] = rank * value_scale + chunk_id

    Args:
        rank:
            当前 MPI 进程编号。

        world_size:
            MPI 总进程数 N。

        chunk_size:
            每个 chunk 中的元素数量。

        value_scale:
            不同 rank 之间的数据间隔。

        dtype:
            numpy 数据类型。

    Returns:
        chunks:
            当前 rank 的本地数据块。
    """

    validate_world_size(world_size)
    validate_chunk_size(chunk_size)

    chunks = np.empty((world_size, chunk_size), dtype=dtype)

    for chunk_id in range(world_size):
        value = rank * value_scale + chunk_id
        chunks[chunk_id, :] = value

    return chunks


def compute_expected_chunks(
    world_size: int,
    chunk_size: int,
    value_scale: int = 10,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    计算理论 All-Reduce 结果。

    对于某个 chunk_id，理论结果为：

        expected[chunk_id, :] =
            sum(rank * value_scale + chunk_id for rank in range(world_size))

    Args:
        world_size:
            MPI 总进程数。

        chunk_size:
            每个 chunk 的元素数量。

        value_scale:
            初始化时使用的数据间隔。

        dtype:
            numpy 数据类型。

    Returns:
        expected:
            理论 All-Reduce 结果，形状为 [world_size, chunk_size]。
    """

    validate_world_size(world_size)
    validate_chunk_size(chunk_size)

    expected = np.empty((world_size, chunk_size), dtype=dtype)

    for chunk_id in range(world_size):
        value = 0

        for rank in range(world_size):
            value += rank * value_scale + chunk_id

        expected[chunk_id, :] = value

    return expected


def summarize_chunks(chunks: np.ndarray) -> List[float]:
    """
    将 chunks 压缩成一维摘要，便于打印。

    当前初始化中，每个 chunk 内部所有元素相同。
    因此取每个 chunk 的第一个元素作为摘要。

    Args:
        chunks:
            数据块数组，形状为 [world_size, chunk_size]。

    Returns:
        每个 chunk 的第一个元素组成的列表。
    """

    return [float(chunks[chunk_id, 0]) for chunk_id in range(chunks.shape[0])]


def ring_reduce_scatter_inplace(
    chunks: np.ndarray,
    comm: MPI.Comm,
) -> None:
    """
    原地执行 Ring All-Reduce 的 Reduce-Scatter 阶段。

    每个 step 中，当前 rank：
        1. 从左邻居接收 recv_chunk；
        2. 向右邻居发送 send_chunk；
        3. 收到后执行：
               chunks[recv_chunk] += recv_buf

    Args:
        chunks:
            当前 rank 的本地数据块，形状为 [world_size, chunk_size]。

        comm:
            MPI 通信子。
    """

    rank = comm.Get_rank()
    world_size = comm.Get_size()

    reduce_scatter_steps = get_reduce_scatter_steps(world_size)

    for step in range(reduce_scatter_steps):
        schedule = ring_reduce_scatter_schedule(
            rank=rank,
            step=step,
            world_size=world_size,
        )

        send_buf = chunks[schedule.send_chunk].copy()
        recv_buf = np.empty_like(chunks[schedule.recv_chunk])

        tag = step

        requests = [
            comm.Irecv(
                recv_buf,
                source=schedule.left,
                tag=tag,
            ),
            comm.Isend(
                send_buf,
                dest=schedule.right,
                tag=tag,
            ),
        ]

        MPI.Request.Waitall(requests)

        chunks[schedule.recv_chunk] += recv_buf


def ring_allgather_inplace(
    chunks: np.ndarray,
    comm: MPI.Comm,
) -> None:
    """
    原地执行 Ring All-Reduce 的 All-Gather 阶段。

    每个 step 中，当前 rank：
        1. 从左邻居接收 recv_chunk；
        2. 向右邻居发送 send_chunk；
        3. 收到后执行：
               chunks[recv_chunk] = recv_buf

    Args:
        chunks:
            当前 rank 的本地数据块，形状为 [world_size, chunk_size]。

        comm:
            MPI 通信子。
    """

    rank = comm.Get_rank()
    world_size = comm.Get_size()

    allgather_steps = get_allgather_steps(world_size)

    for step in range(allgather_steps):
        schedule = ring_allgather_schedule(
            rank=rank,
            step=step,
            world_size=world_size,
        )

        send_buf = chunks[schedule.send_chunk].copy()
        recv_buf = np.empty_like(chunks[schedule.recv_chunk])

        # 用偏移 tag 区分 Reduce-Scatter 和 All-Gather，避免消息错配。
        tag = 100000 + step

        requests = [
            comm.Irecv(
                recv_buf,
                source=schedule.left,
                tag=tag,
            ),
            comm.Isend(
                send_buf,
                dest=schedule.right,
                tag=tag,
            ),
        ]

        MPI.Request.Waitall(requests)

        chunks[schedule.recv_chunk] = recv_buf


def ring_allreduce_inplace(
    chunks: np.ndarray,
    comm: MPI.Comm,
) -> np.ndarray:
    """
    原地执行完整的 MPI 版 Ring All-Reduce。

    执行顺序：
        1. Reduce-Scatter 阶段；
        2. All-Gather 阶段。

    Args:
        chunks:
            当前 rank 的本地数据块，形状为 [world_size, chunk_size]。

        comm:
            MPI 通信子。

    Returns:
        chunks:
            原地更新后的 All-Reduce 结果。
    """

    rank = comm.Get_rank()
    world_size = comm.Get_size()

    validate_world_size(world_size)

    if chunks.ndim != 2:
        raise ValueError(
            f"chunks must be a 2D array with shape [world_size, chunk_size], "
            f"but got shape={chunks.shape}."
        )

    if chunks.shape[0] != world_size:
        raise ValueError(
            f"chunks.shape[0] must equal world_size. "
            f"chunks.shape[0]={chunks.shape[0]}, world_size={world_size}, rank={rank}."
        )

    ring_reduce_scatter_inplace(
        chunks=chunks,
        comm=comm,
    )

    ring_allgather_inplace(
        chunks=chunks,
        comm=comm,
    )

    return chunks


def check_local_correctness(
    chunks: np.ndarray,
    expected: np.ndarray,
    atol: float = 1e-6,
) -> bool:
    """
    检查当前 rank 的最终结果是否正确。

    Args:
        chunks:
            当前 rank 的最终数据。

        expected:
            理论 All-Reduce 结果。

        atol:
            浮点比较绝对误差。

    Returns:
        True 表示当前 rank 正确；
        False 表示当前 rank 错误。
    """

    return bool(np.allclose(chunks, expected, atol=atol, rtol=0.0))


def ordered_print(comm: MPI.Comm, message: str) -> None:
    """
    按 rank 顺序打印信息，避免多进程输出互相穿插。

    Args:
        comm:
            MPI 通信子。

        message:
            当前 rank 需要打印的字符串。
    """

    rank = comm.Get_rank()
    world_size = comm.Get_size()

    for current_rank in range(world_size):
        comm.Barrier()

        if rank == current_rank:
            print(message, flush=True)

    comm.Barrier()


def run_ring_mpi(
    chunk_size: int,
    value_scale: int = 10,
    dtype: np.dtype = np.float32,
    verbose: bool = False,
) -> RingRunResult:
    """
    执行一次 MPI 版 Ring All-Reduce，并检查正确性。

    Args:
        chunk_size:
            每个 chunk 的元素数量。

        value_scale:
            初始化数据时使用的数据间隔。

        dtype:
            numpy 数据类型。

        verbose:
            是否打印每个 rank 的详细输入输出摘要。

    Returns:
        RingRunResult:
            当前 rank 的运行结果。
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    validate_world_size(world_size)
    validate_chunk_size(chunk_size)

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

    initial_summary = summarize_chunks(chunks)

    reduce_scatter_steps = get_reduce_scatter_steps(world_size)
    allgather_steps = get_allgather_steps(world_size)
    total_steps = get_total_steps(world_size)

    comm.Barrier()
    start_time = time.perf_counter()

    ring_allreduce_inplace(
        chunks=chunks,
        comm=comm,
    )

    comm.Barrier()
    elapsed_seconds = time.perf_counter() - start_time

    local_correct = check_local_correctness(
        chunks=chunks,
        expected=expected,
    )

    correct_count = comm.allreduce(
        int(local_correct),
        op=MPI.SUM,
    )

    global_correct = correct_count == world_size

    final_summary = summarize_chunks(chunks)
    expected_summary = summarize_chunks(expected)

    if verbose:
        message = (
            f"[rank {rank}]\n"
            f"  initial  = {initial_summary}\n"
            f"  final    = {final_summary}\n"
            f"  expected = {expected_summary}\n"
            f"  local_correct = {local_correct}\n"
        )

        ordered_print(comm, message)

    return RingRunResult(
        rank=rank,
        world_size=world_size,
        chunk_size=chunk_size,
        reduce_scatter_steps=reduce_scatter_steps,
        allgather_steps=allgather_steps,
        total_steps=total_steps,
        elapsed_seconds=elapsed_seconds,
        local_correct=local_correct,
        global_correct=global_correct,
    )


def parse_dtype(dtype_name: str) -> np.dtype:
    """
    将命令行输入的数据类型字符串转换为 numpy dtype。

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


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace:
            命令行参数对象。
    """

    parser = argparse.ArgumentParser(
        description="Run MPI implementation of standard Ring All-Reduce."
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="Number of elements in each chunk. Default: 1024.",
    )

    parser.add_argument(
        "--value_scale",
        type=int,
        default=10,
        help=(
            "Scale used to initialize worker data. "
            "Default: chunks[chunk_id, :] = rank * 10 + chunk_id."
        ),
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Data type used in communication. Default: float32.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print initial/final/expected summaries for each rank.",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Run correctness check. "
            "This flag is kept for readability; correctness is always checked."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    程序入口。

    执行流程：
        1. 解析命令行参数；
        2. 初始化当前 rank 的数据；
        3. 执行 MPI 版 Ring All-Reduce；
        4. 检查当前 rank 和全局正确性；
        5. rank 0 打印摘要；
        6. 如果任意 rank 错误，则返回非 0 退出码。
    """

    args = parse_args()
    dtype = parse_dtype(args.dtype)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    try:
        result = run_ring_mpi(
            chunk_size=args.chunk_size,
            value_scale=args.value_scale,
            dtype=dtype,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"[rank {rank}] [FAIL] Runtime error: {exc}", flush=True)
        raise SystemExit(1) from exc

    if rank == 0:
        print("=" * 80)
        print("MPI Ring All-Reduce Summary")
        print("=" * 80)
        print(f"world_size           : {result.world_size}")
        print(f"chunk_size           : {result.chunk_size}")
        print(f"reduce_scatter_steps : {result.reduce_scatter_steps}")
        print(f"allgather_steps      : {result.allgather_steps}")
        print(f"total_steps          : {result.total_steps}")
        print(f"elapsed_seconds      : {result.elapsed_seconds:.6f}")
        print(f"global_correct       : {result.global_correct}")

        if result.global_correct:
            print("[PASS] MPI Ring correctness check passed.")
        else:
            print("[FAIL] MPI Ring correctness check failed.")

        print("=" * 80)

    if not result.global_correct:
        raise SystemExit(1)


if __name__ == "__main__":
    main()