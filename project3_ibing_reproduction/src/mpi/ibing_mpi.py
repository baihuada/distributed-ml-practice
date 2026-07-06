#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/mpi/ibing_mpi.py

文件作用：
    该文件用于实现 IBing All-Reduce 的 MPI 多进程版本。

    前面阶段已经完成：
        1. src/simulator/ibing_schedule.py
           验证 IBing 的 send_chunk / recv_chunk 调度公式。

        2. src/simulator/ibing_sim.py
           在单进程中模拟多个 worker，验证 IBing 数据流正确性。

    本文件进入真正的 MPI 多进程通信阶段：
        1. 每个 MPI 进程对应一个 worker；
        2. 每个 worker 只知道自己的 rank、world_size、左邻居、右邻居；
        3. 每一步同时向左右两个方向发送不同 chunk；
        4. 使用非阻塞通信 MPI.Irecv / MPI.Isend；
        5. Reduce-Scatter 阶段执行加法；
        6. All-Gather 阶段只复制和转发；
        7. 最后验证每个 rank 是否得到正确的 All-Reduce 结果。

运行示例：
    Linux / WSL / OpenMPI:
        mpirun -np 5 python src/mpi/ibing_mpi.py --chunk_size 4 --verbose

    Windows / MS-MPI:
        mpiexec -n 5 python src/mpi/ibing_mpi.py --chunk_size 4 --verbose

    只做正确性检查：
        mpiexec -n 5 python src/mpi/ibing_mpi.py --chunk_size 1024 --check

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

    理论 All-Reduce 结果为：

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
from typing import List, Tuple

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
            当前 rank 的左邻居。

        right:
            当前 rank 的右邻居。
    """

    rank: int
    left: int
    right: int


@dataclass(frozen=True)
class IbingStepSchedule:
    """
    保存当前 rank 在某个 step 下的 IBing 通信调度。

    Attributes:
        rank:
            当前 rank。

        step:
            当前通信步，0-based。

        phase:
            当前阶段，取值为 "Reduce-Scatter" 或 "All-Gather"。

        left:
            当前 rank 的左邻居。

        right:
            当前 rank 的右邻居。

        send_chunk_1:
            方向 1 中，当前 rank 发给右邻居的 chunk 编号。

        recv_chunk_1:
            方向 1 中，当前 rank 从左邻居接收的 chunk 编号。

        send_chunk_2:
            方向 2 中，当前 rank 发给左邻居的 chunk 编号。

        recv_chunk_2:
            方向 2 中，当前 rank 从右邻居接收的 chunk 编号。
    """

    rank: int
    step: int
    phase: str
    left: int
    right: int
    send_chunk_1: int
    recv_chunk_1: int
    send_chunk_2: int
    recv_chunk_2: int


@dataclass(frozen=True)
class IbingRunResult:
    """
    保存一次 MPI 版 IBing 运行结果。

    Attributes:
        rank:
            当前进程 rank。

        world_size:
            MPI 总进程数。

        chunk_size:
            每个 chunk 中的元素数量。

        total_steps:
            IBing 总通信步数，即 N - 1。

        reduce_steps:
            Reduce-Scatter 阶段步数。

        allgather_steps:
            All-Gather 阶段步数。

        elapsed_seconds:
            当前 rank 执行 IBing All-Reduce 的耗时。

        local_correct:
            当前 rank 的最终结果是否正确。

        global_correct:
            所有 rank 的最终结果是否全部正确。
    """

    rank: int
    world_size: int
    chunk_size: int
    total_steps: int
    reduce_steps: int
    allgather_steps: int
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


def get_total_steps(world_size: int) -> int:
    """
    返回 IBing 总通信步数。

    对于 N 个 worker，IBing 总通信步数为：

        N - 1

    Args:
        world_size:
            worker 总数 N。

    Returns:
        IBing 总通信步数。
    """

    validate_world_size(world_size)
    return world_size - 1


def get_reduce_steps(world_size: int) -> int:
    """
    返回 IBing Reduce-Scatter 阶段步数。

    当前实现与前面的单进程模拟保持一致：

        reduce_steps = N // 2

    例如：
        N=5:
            total_steps = 4
            reduce_steps = 2
            allgather_steps = 2

        N=4:
            total_steps = 3
            reduce_steps = 2
            allgather_steps = 1

    Args:
        world_size:
            worker 总数 N。

    Returns:
        Reduce-Scatter 阶段步数。
    """

    validate_world_size(world_size)
    return world_size // 2


def get_phase(step: int, world_size: int) -> str:
    """
    判断当前 step 属于 Reduce-Scatter 还是 All-Gather。

    Args:
        step:
            当前通信步，0-based。

        world_size:
            worker 总数 N。

    Returns:
        "Reduce-Scatter" 或 "All-Gather"。
    """

    reduce_steps = get_reduce_steps(world_size)

    if step < reduce_steps:
        return "Reduce-Scatter"

    return "All-Gather"


def ibing_schedule(rank: int, step: int, world_size: int) -> IbingStepSchedule:
    """
    计算当前 rank 在当前 step 中的 IBing 通信调度。

    IBing 每一步同时做两个方向通信：

    方向 1：
        当前 rank 向右邻居发送 send_chunk_1；
        当前 rank 从左邻居接收 recv_chunk_1。

    方向 2：
        当前 rank 向左邻居发送 send_chunk_2；
        当前 rank 从右邻居接收 recv_chunk_2。

    核心公式：

        recv_chunk_1 = (rank - step - 1 + world_size) % world_size

        send_chunk_1 = (rank - step + world_size) % world_size

        recv_chunk_2 = (rank + step + world_size + 2) % world_size

        send_chunk_2 = (rank + step + world_size + 1) % world_size

    Args:
        rank:
            当前 MPI 进程编号。

        step:
            当前通信步，0-based。

        world_size:
            MPI 总进程数 N。

    Returns:
        IbingStepSchedule:
            当前 rank 当前 step 的通信调度。
    """

    validate_world_size(world_size)

    neighbors = get_neighbors(rank, world_size)

    recv_chunk_1 = (rank - step - 1 + world_size) % world_size
    send_chunk_1 = (rank - step + world_size) % world_size

    recv_chunk_2 = (rank + step + world_size + 2) % world_size
    send_chunk_2 = (rank + step + world_size + 1) % world_size

    phase = get_phase(step, world_size)

    return IbingStepSchedule(
        rank=rank,
        step=step,
        phase=phase,
        left=neighbors.left,
        right=neighbors.right,
        send_chunk_1=send_chunk_1,
        recv_chunk_1=recv_chunk_1,
        send_chunk_2=send_chunk_2,
        recv_chunk_2=recv_chunk_2,
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
    将 chunks 压缩成一个便于打印的一维摘要。

    当前初始化中，每个 chunk 内部所有元素相同。
    因此可以取每个 chunk 的第一个元素作为摘要。

    Args:
        chunks:
            数据块数组，形状为 [world_size, chunk_size]。

    Returns:
        每个 chunk 的第一个元素组成的列表。
    """

    return [float(chunks[chunk_id, 0]) for chunk_id in range(chunks.shape[0])]


def ibing_allreduce_inplace(
    chunks: np.ndarray,
    comm: MPI.Comm,
) -> np.ndarray:
    """
    原地执行 MPI 版 IBing All-Reduce。

    每个 rank 调用该函数时，只传入自己的本地 chunks。
    函数执行结束后，chunks 会被原地更新为 All-Reduce 后的结果。

    通信逻辑：
        对于每个 step：

        方向 1：
            当前 rank 向右邻居发送 send_chunk_1；
            当前 rank 从左邻居接收 recv_chunk_1。

        方向 2：
            当前 rank 向左邻居发送 send_chunk_2；
            当前 rank 从右邻居接收 recv_chunk_2。

        Reduce-Scatter 阶段：
            chunks[recv_chunk] += recv_buf

        All-Gather 阶段：
            chunks[recv_chunk] = recv_buf

    非阻塞通信顺序：
        1. 先挂起两个 Irecv；
        2. 再启动两个 Isend；
        3. 使用 MPI.Request.Waitall 等待全部完成；
        4. 再更新本地 chunks。

    Args:
        chunks:
            当前 rank 的本地数据块，形状为 [world_size, chunk_size]。

        comm:
            MPI 通信子，通常为 MPI.COMM_WORLD。

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
            f"chunks.shape[0]={chunks.shape[0]}, world_size={world_size}."
        )

    total_steps = get_total_steps(world_size)
    reduce_steps = get_reduce_steps(world_size)

    for step in range(total_steps):
        schedule = ibing_schedule(
            rank=rank,
            step=step,
            world_size=world_size,
        )

        # 关键点：
        #   非阻塞发送时，发送缓冲区在通信完成前不能被修改。
        #   所以这里必须 copy 一份发送数据，避免后续 reduce 或 allgather 更新原数组。
        send_buf_1 = chunks[schedule.send_chunk_1].copy()
        send_buf_2 = chunks[schedule.send_chunk_2].copy()

        recv_buf_1 = np.empty_like(chunks[schedule.recv_chunk_1])
        recv_buf_2 = np.empty_like(chunks[schedule.recv_chunk_2])

        # 使用不同 tag 区分两个方向，避免消息错配。
        #
        # 方向 1：
        #   left -> current -> right
        #
        # 方向 2：
        #   right -> current -> left
        tag_direction_1 = step * 2
        tag_direction_2 = step * 2 + 1

        requests = [
            # 方向 1：从左邻居接收
            comm.Irecv(
                recv_buf_1,
                source=schedule.left,
                tag=tag_direction_1,
            ),

            # 方向 2：从右邻居接收
            comm.Irecv(
                recv_buf_2,
                source=schedule.right,
                tag=tag_direction_2,
            ),

            # 方向 1：向右邻居发送
            comm.Isend(
                send_buf_1,
                dest=schedule.right,
                tag=tag_direction_1,
            ),

            # 方向 2：向左邻居发送
            comm.Isend(
                send_buf_2,
                dest=schedule.left,
                tag=tag_direction_2,
            ),
        ]

        MPI.Request.Waitall(requests)

        if step < reduce_steps:
            # Reduce-Scatter 阶段：
            # 收到 partial sum 后，与本地对应 chunk 累加。
            #
            # 注意：
            #   当左右两个方向收到同一个 chunk_id 时，不能去重。
            #   它们来自不同方向，代表不同 partial sum，应连续加两次。
            chunks[schedule.recv_chunk_1] += recv_buf_1
            chunks[schedule.recv_chunk_2] += recv_buf_2
        else:
            # All-Gather 阶段：
            # 收到的 chunk 已经是完整 reduce 结果，直接保存。
            chunks[schedule.recv_chunk_1] = recv_buf_1
            chunks[schedule.recv_chunk_2] = recv_buf_2

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


def run_ibing_mpi(
    chunk_size: int,
    value_scale: int = 10,
    dtype: np.dtype = np.float32,
    verbose: bool = False,
) -> IbingRunResult:
    """
    执行一次 MPI 版 IBing All-Reduce，并检查正确性。

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
        IbingRunResult:
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

    total_steps = get_total_steps(world_size)
    reduce_steps = get_reduce_steps(world_size)
    allgather_steps = total_steps - reduce_steps

    # 所有进程同步后再开始计时。
    comm.Barrier()
    start_time = time.perf_counter()

    ibing_allreduce_inplace(
        chunks=chunks,
        comm=comm,
    )

    comm.Barrier()
    elapsed_seconds = time.perf_counter() - start_time

    local_correct = check_local_correctness(
        chunks=chunks,
        expected=expected,
    )

    # 将每个 rank 的正确性结果汇总到所有 rank。
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

    return IbingRunResult(
        rank=rank,
        world_size=world_size,
        chunk_size=chunk_size,
        total_steps=total_steps,
        reduce_steps=reduce_steps,
        allgather_steps=allgather_steps,
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
        description="Run MPI implementation of IBing All-Reduce."
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
        3. 执行 MPI 版 IBing All-Reduce；
        4. 检查当前 rank 和全局正确性；
        5. rank 0 打印摘要；
        6. 如果任意 rank 错误，则返回非 0 退出码。
    """

    args = parse_args()
    dtype = parse_dtype(args.dtype)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    try:
        result = run_ibing_mpi(
            chunk_size=args.chunk_size,
            value_scale=args.value_scale,
            dtype=dtype,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"[rank {rank}] [FAIL] Runtime error: {exc}", flush=True)
        raise SystemExit(1) from exc

    # 只让 rank 0 打印总摘要。
    if rank == 0:
        print("=" * 80)
        print("MPI IBing All-Reduce Summary")
        print("=" * 80)
        print(f"world_size      : {result.world_size}")
        print(f"chunk_size      : {result.chunk_size}")
        print(f"total_steps     : {result.total_steps}")
        print(f"reduce_steps    : {result.reduce_steps}")
        print(f"allgather_steps : {result.allgather_steps}")
        print(f"elapsed_seconds : {result.elapsed_seconds:.6f}")
        print(f"global_correct  : {result.global_correct}")

        if result.global_correct:
            print("[PASS] MPI IBing correctness check passed.")
        else:
            print("[FAIL] MPI IBing correctness check failed.")

        print("=" * 80)

    if not result.global_correct:
        raise SystemExit(1)


if __name__ == "__main__":
    main()