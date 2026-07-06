#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/simulator/ring_sim.py

文件作用：
    该文件用于在单进程中模拟标准 Ring All-Reduce 的完整数据流。

    它是 IBing 复现项目中的 baseline 文件，用来和 IBing All-Reduce 做对比。

    Ring All-Reduce 分为两个阶段：
        1. Reduce-Scatter 阶段：
            每个 rank 向右邻居发送一个 chunk；
            每个 rank 从左邻居接收一个 chunk；
            收到后对对应 chunk 执行加法 reduce。

        2. All-Gather 阶段：
            每个 rank 继续向右邻居发送一个已经聚合完成的 chunk；
            每个 rank 从左邻居接收一个 chunk；
            收到后直接保存，不再做加法。

    对于 N 个 worker：
        Reduce-Scatter 需要 N - 1 步；
        All-Gather 需要 N - 1 步；
        因此总通信步数为 2(N - 1)。

使用示例：
    只看最终正确性：
        python src/simulator/ring_sim.py --world_size 5

    打印初始状态、理论结果和最终状态：
        python src/simulator/ring_sim.py --world_size 5 --verbose

    打印每一步通信后的中间状态：
        python src/simulator/ring_sim.py --world_size 5 --trace

初始化规则：
    workers[rank][chunk_id] = rank * value_scale + chunk_id

    默认 value_scale = 10。

例如 world_size = 5 时：
    rank 0: [0, 1, 2, 3, 4]
    rank 1: [10, 11, 12, 13, 14]
    rank 2: [20, 21, 22, 23, 24]
    rank 3: [30, 31, 32, 33, 34]
    rank 4: [40, 41, 42, 43, 44]

理论 All-Reduce 结果：
    [100, 105, 110, 115, 120]

最终所有 rank 都应该得到：
    [100, 105, 110, 115, 120]
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple


Number = int
WorkerState = List[List[Number]]

# messages[(src_rank, dst_rank)] = message
MessageKey = Tuple[int, int]


@dataclass(frozen=True)
class NeighborInfo:
    """
    保存某个 rank 的左右邻居信息。

    Attributes:
        rank:
            当前 worker 编号。

        left:
            当前 worker 的左邻居编号。

        right:
            当前 worker 的右邻居编号。
    """

    rank: int
    left: int
    right: int


@dataclass(frozen=True)
class RingStepSchedule:
    """
    保存某个 rank 在某个 Ring All-Reduce step 中的通信调度。

    Attributes:
        rank:
            当前 worker 编号。

        step:
            当前阶段内部的 step，0-based。

        display_step:
            用于显示的 step，1-based。

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
    display_step: int
    phase: str
    left: int
    right: int
    send_chunk: int
    recv_chunk: int


@dataclass(frozen=True)
class SimulatedMessage:
    """
    表示 Ring All-Reduce 中一条模拟消息。

    Attributes:
        step:
            当前阶段内部的 step，0-based。

        phase:
            当前阶段，Reduce-Scatter 或 All-Gather。

        src_rank:
            发送方 rank。

        dst_rank:
            接收方 rank。

        chunk_id:
            消息携带的 chunk 编号。

        value:
            消息携带的数据值。
            当前模拟中为整数；后续可扩展为 numpy 数组。
    """

    step: int
    phase: str
    src_rank: int
    dst_rank: int
    chunk_id: int
    value: Number


@dataclass(frozen=True)
class SimulationResult:
    """
    保存一次 Ring All-Reduce 单进程模拟的结果。

    Attributes:
        world_size:
            worker 总数 N。

        reduce_scatter_steps:
            Reduce-Scatter 阶段通信步数，即 N - 1。

        allgather_steps:
            All-Gather 阶段通信步数，即 N - 1。

        total_steps:
            总通信步数，即 2(N - 1)。

        initial_workers:
            初始 worker 数据。

        final_workers:
            模拟结束后的 worker 数据。

        expected_result:
            理论 All-Reduce 正确结果。

        is_correct:
            final_workers 是否与 expected_result 完全一致。
    """

    world_size: int
    reduce_scatter_steps: int
    allgather_steps: int
    total_steps: int
    initial_workers: WorkerState
    final_workers: WorkerState
    expected_result: List[Number]
    is_correct: bool


def validate_world_size(world_size: int) -> None:
    """
    检查 worker 总数是否合法。

    Args:
        world_size:
            参与 All-Reduce 的 worker 数量。

    Raises:
        ValueError:
            当 world_size 小于 2 时抛出异常。
    """

    if world_size < 2:
        raise ValueError("world_size must be >= 2.")


def validate_rank(rank: int, world_size: int) -> None:
    """
    检查 rank 编号是否合法。

    Args:
        rank:
            当前 worker 编号。

        world_size:
            worker 总数。

    Raises:
        ValueError:
            当 rank 不在 [0, world_size - 1] 范围内时抛出异常。
    """

    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"rank must be in [0, {world_size - 1}], but got rank={rank}."
        )


def validate_step(step: int, world_size: int) -> None:
    """
    检查阶段内部 step 编号是否合法。

    Ring All-Reduce 的每个阶段都需要 N - 1 步。
    因此 step 的合法范围是 [0, N - 2]。

    Args:
        step:
            当前阶段内部 step，0-based。

        world_size:
            worker 总数 N。

    Raises:
        ValueError:
            当 step 不在合法范围内时抛出异常。
    """

    steps_per_phase = world_size - 1

    if step < 0 or step >= steps_per_phase:
        raise ValueError(
            f"step must be in [0, {steps_per_phase - 1}], but got step={step}."
        )


def get_neighbors(rank: int, world_size: int) -> NeighborInfo:
    """
    计算当前 rank 在环形拓扑中的左右邻居。

    对于 rank=r：
        left  = (r - 1 + N) % N
        right = (r + 1) % N

    Args:
        rank:
            当前 worker 编号。

        world_size:
            worker 总数 N。

    Returns:
        NeighborInfo:
            当前 rank 的左右邻居信息。
    """

    validate_world_size(world_size)
    validate_rank(rank, world_size)

    left = (rank - 1 + world_size) % world_size
    right = (rank + 1) % world_size

    return NeighborInfo(rank=rank, left=left, right=right)


def get_reduce_scatter_steps(world_size: int) -> int:
    """
    返回 Reduce-Scatter 阶段通信步数。

    标准 Ring All-Reduce 中：
        Reduce-Scatter 阶段需要 N - 1 步。

    Args:
        world_size:
            worker 总数 N。

    Returns:
        Reduce-Scatter 阶段步数。
    """

    validate_world_size(world_size)
    return world_size - 1


def get_allgather_steps(world_size: int) -> int:
    """
    返回 All-Gather 阶段通信步数。

    标准 Ring All-Reduce 中：
        All-Gather 阶段需要 N - 1 步。

    Args:
        world_size:
            worker 总数 N。

    Returns:
        All-Gather 阶段步数。
    """

    validate_world_size(world_size)
    return world_size - 1


def get_total_steps(world_size: int) -> int:
    """
    返回 Ring All-Reduce 的总通信步数。

    标准 Ring All-Reduce 总步数为：
        2(N - 1)

    Args:
        world_size:
            worker 总数 N。

    Returns:
        Ring All-Reduce 的总通信步数。
    """

    validate_world_size(world_size)
    return 2 * (world_size - 1)


def ring_reduce_scatter_schedule(
    rank: int,
    step: int,
    world_size: int,
) -> RingStepSchedule:
    """
    计算 Reduce-Scatter 阶段中某个 rank 在某个 step 的通信调度。

    标准 Ring All-Reduce 中，每个 rank 单向向右发送，从左接收。

    对于 rank=r，step=i：
        send_chunk = (r - i + N) % N
        recv_chunk = (r - i - 1 + N) % N

    发送方向：
        rank r -> right(r)

    接收方向：
        left(r) -> rank r

    Args:
        rank:
            当前 worker 编号。

        step:
            Reduce-Scatter 阶段内部 step，0-based。

        world_size:
            worker 总数 N。

    Returns:
        RingStepSchedule:
            当前 rank 当前 step 的 Reduce-Scatter 调度信息。
    """

    validate_world_size(world_size)
    validate_rank(rank, world_size)
    validate_step(step, world_size)

    neighbors = get_neighbors(rank, world_size)

    send_chunk = (rank - step + world_size) % world_size
    recv_chunk = (rank - step - 1 + world_size) % world_size

    return RingStepSchedule(
        rank=rank,
        step=step,
        display_step=step + 1,
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
    计算 All-Gather 阶段中某个 rank 在某个 step 的通信调度。

    Reduce-Scatter 结束后，每个 rank 持有一个已经完整 reduce 的 chunk。
    All-Gather 阶段负责把这些完整 chunk 沿环传播给所有 rank。

    对于 rank=r，step=i：
        send_chunk = (r - i + 1 + N) % N
        recv_chunk = (r - i + N) % N

    发送方向：
        rank r -> right(r)

    接收方向：
        left(r) -> rank r

    Args:
        rank:
            当前 worker 编号。

        step:
            All-Gather 阶段内部 step，0-based。

        world_size:
            worker 总数 N。

    Returns:
        RingStepSchedule:
            当前 rank 当前 step 的 All-Gather 调度信息。
    """

    validate_world_size(world_size)
    validate_rank(rank, world_size)
    validate_step(step, world_size)

    neighbors = get_neighbors(rank, world_size)

    send_chunk = (rank - step + 1 + world_size) % world_size
    recv_chunk = (rank - step + world_size) % world_size

    return RingStepSchedule(
        rank=rank,
        step=step,
        display_step=step + 1,
        phase="All-Gather",
        left=neighbors.left,
        right=neighbors.right,
        send_chunk=send_chunk,
        recv_chunk=recv_chunk,
    )


def create_initial_workers(world_size: int, value_scale: int = 10) -> WorkerState:
    """
    初始化所有 worker 的本地数据。

    初始化规则：
        workers[rank][chunk_id] = rank * value_scale + chunk_id

    Args:
        world_size:
            worker 总数 N。

        value_scale:
            不同 rank 之间的数据间隔。
            默认是 10，方便人工检查。

    Returns:
        workers:
            二维列表，形状为 [world_size][world_size]。
    """

    validate_world_size(world_size)

    workers: WorkerState = []

    for rank in range(world_size):
        rank_chunks = []

        for chunk_id in range(world_size):
            value = rank * value_scale + chunk_id
            rank_chunks.append(value)

        workers.append(rank_chunks)

    return workers


def compute_expected_result(initial_workers: WorkerState) -> List[Number]:
    """
    根据初始数据计算理论 All-Reduce 结果。

    对于 chunk j：
        expected[j] = sum(initial_workers[rank][j] for rank in all ranks)

    Args:
        initial_workers:
            所有 rank 的初始数据。

    Returns:
        expected_result:
            每个 chunk 的全局求和结果。
    """

    world_size = len(initial_workers)

    if world_size == 0:
        raise ValueError("initial_workers must not be empty.")

    num_chunks = len(initial_workers[0])
    expected_result: List[Number] = []

    for chunk_id in range(num_chunks):
        chunk_sum = 0

        for rank in range(world_size):
            chunk_sum += initial_workers[rank][chunk_id]

        expected_result.append(chunk_sum)

    return expected_result


def check_correctness(
    final_workers: WorkerState,
    expected_result: List[Number],
) -> bool:
    """
    检查 Ring All-Reduce 模拟结果是否正确。

    正确条件：
        每个 rank 的最终数据都必须等于 expected_result。

    Args:
        final_workers:
            模拟结束后的 worker 数据。

        expected_result:
            理论 All-Reduce 结果。

    Returns:
        True 表示所有 rank 都正确；
        False 表示至少一个 rank 不正确。
    """

    for rank_data in final_workers:
        if rank_data != expected_result:
            return False

    return True


def format_workers(workers: WorkerState) -> str:
    """
    将 workers 状态格式化为可读字符串。

    Args:
        workers:
            当前所有 rank 的数据状态。

    Returns:
        格式化后的字符串。
    """

    lines = []

    for rank, rank_data in enumerate(workers):
        lines.append(f"rank {rank}: {rank_data}")

    return "\n".join(lines)


def print_workers(title: str, workers: WorkerState) -> None:
    """
    打印所有 worker 当前保存的数据。

    Args:
        title:
            打印标题。

        workers:
            当前所有 rank 的数据状态。
    """

    print(title)
    print("-" * len(title))
    print(format_workers(workers))
    print()


def build_messages_for_reduce_scatter_step(
    snapshot_workers: WorkerState,
    step: int,
    world_size: int,
) -> Dict[MessageKey, SimulatedMessage]:
    """
    构造 Reduce-Scatter 阶段某个 step 中所有 rank 发出的消息。

    注意：
        必须使用 snapshot_workers 作为发送数据来源。

    原因：
        真实 Ring All-Reduce 中，同一个 step 内所有 rank 是并行通信的。
        单进程模拟时如果边遍历边更新 workers，会导致后续 rank 读取到
        当前 step 已更新过的数据，从而破坏真实并行语义。

    Args:
        snapshot_workers:
            当前 step 开始前的 worker 数据快照。

        step:
            Reduce-Scatter 阶段内部 step，0-based。

        world_size:
            worker 总数 N。

    Returns:
        messages:
            当前 step 中所有 rank 发出的消息。
    """

    messages: Dict[MessageKey, SimulatedMessage] = {}

    for rank in range(world_size):
        schedule = ring_reduce_scatter_schedule(
            rank=rank,
            step=step,
            world_size=world_size,
        )

        message = SimulatedMessage(
            step=step,
            phase=schedule.phase,
            src_rank=rank,
            dst_rank=schedule.right,
            chunk_id=schedule.send_chunk,
            value=snapshot_workers[rank][schedule.send_chunk],
        )

        messages[(message.src_rank, message.dst_rank)] = message

    return messages


def build_messages_for_allgather_step(
    snapshot_workers: WorkerState,
    step: int,
    world_size: int,
) -> Dict[MessageKey, SimulatedMessage]:
    """
    构造 All-Gather 阶段某个 step 中所有 rank 发出的消息。

    Args:
        snapshot_workers:
            当前 step 开始前的 worker 数据快照。

        step:
            All-Gather 阶段内部 step，0-based。

        world_size:
            worker 总数 N。

    Returns:
        messages:
            当前 step 中所有 rank 发出的消息。
    """

    messages: Dict[MessageKey, SimulatedMessage] = {}

    for rank in range(world_size):
        schedule = ring_allgather_schedule(
            rank=rank,
            step=step,
            world_size=world_size,
        )

        message = SimulatedMessage(
            step=step,
            phase=schedule.phase,
            src_rank=rank,
            dst_rank=schedule.right,
            chunk_id=schedule.send_chunk,
            value=snapshot_workers[rank][schedule.send_chunk],
        )

        messages[(message.src_rank, message.dst_rank)] = message

    return messages


def get_expected_received_message(
    messages: Dict[MessageKey, SimulatedMessage],
    schedule: RingStepSchedule,
) -> SimulatedMessage:
    """
    根据某个 rank 的调度信息，取出它当前 step 应该收到的消息。

    对当前 rank 来说：
        它从左邻居接收 recv_chunk。
        这条消息的 key 应该是：
            (left_rank, current_rank)

    Args:
        messages:
            当前 step 的所有消息。

        schedule:
            当前 rank 当前 step 的调度信息。

    Returns:
        message:
            当前 rank 从左邻居收到的消息。

    Raises:
        RuntimeError:
            如果消息不存在，或者 chunk 编号不匹配，说明调度实现有错误。
    """

    key = (schedule.left, schedule.rank)

    if key not in messages:
        raise RuntimeError(
            f"Missing message. "
            f"rank={schedule.rank}, step={schedule.step}, "
            f"phase={schedule.phase}, key={key}"
        )

    message = messages[key]

    if message.chunk_id != schedule.recv_chunk:
        raise RuntimeError(
            f"Chunk mismatch. "
            f"rank={schedule.rank}, step={schedule.step}, "
            f"phase={schedule.phase}, "
            f"expected={schedule.recv_chunk}, got={message.chunk_id}"
        )

    return message


def apply_reduce_scatter_update(
    workers: WorkerState,
    rank: int,
    message: SimulatedMessage,
) -> None:
    """
    在 Reduce-Scatter 阶段更新当前 rank 的数据。

    Reduce-Scatter 阶段规则：
        workers[rank][recv_chunk] += received_value

    Args:
        workers:
            当前所有 worker 的数据状态。

        rank:
            当前需要更新的 rank。

        message:
            当前 rank 从左邻居收到的消息。
    """

    workers[rank][message.chunk_id] += message.value


def apply_allgather_update(
    workers: WorkerState,
    rank: int,
    message: SimulatedMessage,
) -> None:
    """
    在 All-Gather 阶段更新当前 rank 的数据。

    All-Gather 阶段规则：
        workers[rank][recv_chunk] = received_value

    Args:
        workers:
            当前所有 worker 的数据状态。

        rank:
            当前需要更新的 rank。

        message:
            当前 rank 从左邻居收到的消息。
    """

    workers[rank][message.chunk_id] = message.value


def simulate_reduce_scatter_step(
    workers: WorkerState,
    step: int,
    world_size: int,
) -> None:
    """
    模拟 Ring All-Reduce 的一个 Reduce-Scatter step。

    执行流程：
        1. 复制当前 workers 作为快照；
        2. 每个 rank 从快照中取 send_chunk，向右邻居发送；
        3. 每个 rank 从左邻居接收 recv_chunk；
        4. 对接收到的 chunk 执行加法 reduce。

    Args:
        workers:
            当前所有 worker 的数据状态。
            该对象会被原地修改。

        step:
            Reduce-Scatter 阶段内部 step，0-based。

        world_size:
            worker 总数 N。
    """

    snapshot_workers = deepcopy(workers)

    messages = build_messages_for_reduce_scatter_step(
        snapshot_workers=snapshot_workers,
        step=step,
        world_size=world_size,
    )

    for rank in range(world_size):
        schedule = ring_reduce_scatter_schedule(
            rank=rank,
            step=step,
            world_size=world_size,
        )

        message = get_expected_received_message(
            messages=messages,
            schedule=schedule,
        )

        apply_reduce_scatter_update(
            workers=workers,
            rank=rank,
            message=message,
        )


def simulate_allgather_step(
    workers: WorkerState,
    step: int,
    world_size: int,
) -> None:
    """
    模拟 Ring All-Reduce 的一个 All-Gather step。

    执行流程：
        1. 复制当前 workers 作为快照；
        2. 每个 rank 从快照中取 send_chunk，向右邻居发送；
        3. 每个 rank 从左邻居接收 recv_chunk；
        4. 将收到的数据直接保存，不再加法。

    Args:
        workers:
            当前所有 worker 的数据状态。
            该对象会被原地修改。

        step:
            All-Gather 阶段内部 step，0-based。

        world_size:
            worker 总数 N。
    """

    snapshot_workers = deepcopy(workers)

    messages = build_messages_for_allgather_step(
        snapshot_workers=snapshot_workers,
        step=step,
        world_size=world_size,
    )

    for rank in range(world_size):
        schedule = ring_allgather_schedule(
            rank=rank,
            step=step,
            world_size=world_size,
        )

        message = get_expected_received_message(
            messages=messages,
            schedule=schedule,
        )

        apply_allgather_update(
            workers=workers,
            rank=rank,
            message=message,
        )


def simulate_ring_allreduce(
    world_size: int,
    value_scale: int = 10,
    verbose: bool = False,
    trace: bool = False,
) -> SimulationResult:
    """
    模拟完整的标准 Ring All-Reduce 流程。

    Args:
        world_size:
            worker 总数 N。

        value_scale:
            初始化 worker 数据时使用的数据间隔。

        verbose:
            是否打印初始数据、理论结果和最终数据。

        trace:
            是否打印每个 step 后的中间状态。

    Returns:
        SimulationResult:
            包含模拟结果、理论结果和正确性检查结果。
    """

    validate_world_size(world_size)

    reduce_scatter_steps = get_reduce_scatter_steps(world_size)
    allgather_steps = get_allgather_steps(world_size)
    total_steps = get_total_steps(world_size)

    initial_workers = create_initial_workers(
        world_size=world_size,
        value_scale=value_scale,
    )

    expected_result = compute_expected_result(initial_workers)

    workers = deepcopy(initial_workers)

    if verbose or trace:
        print("=" * 80)
        print("Ring All-Reduce Single-Process Simulation")
        print("=" * 80)
        print(f"world_size           : {world_size}")
        print(f"reduce_scatter_steps : {reduce_scatter_steps}")
        print(f"allgather_steps      : {allgather_steps}")
        print(f"total_steps          : {total_steps}")
        print("=" * 80)
        print()

        print_workers("Initial workers", workers)
        print(f"Expected all-reduce result:\n{expected_result}")
        print()

    for step in range(reduce_scatter_steps):
        simulate_reduce_scatter_step(
            workers=workers,
            step=step,
            world_size=world_size,
        )

        if trace:
            print_workers(
                title=(
                    f"After Reduce-Scatter step "
                    f"{step + 1}/{reduce_scatter_steps}"
                ),
                workers=workers,
            )

    for step in range(allgather_steps):
        simulate_allgather_step(
            workers=workers,
            step=step,
            world_size=world_size,
        )

        if trace:
            print_workers(
                title=f"After All-Gather step {step + 1}/{allgather_steps}",
                workers=workers,
            )

    is_correct = check_correctness(
        final_workers=workers,
        expected_result=expected_result,
    )

    if verbose or trace:
        print_workers("Final workers", workers)

    return SimulationResult(
        world_size=world_size,
        reduce_scatter_steps=reduce_scatter_steps,
        allgather_steps=allgather_steps,
        total_steps=total_steps,
        initial_workers=initial_workers,
        final_workers=workers,
        expected_result=expected_result,
        is_correct=is_correct,
    )


def print_summary(result: SimulationResult) -> None:
    """
    打印 Ring All-Reduce 模拟结果摘要。

    Args:
        result:
            Ring All-Reduce 模拟结果。
    """

    print("=" * 80)
    print("Simulation Summary")
    print("=" * 80)
    print(f"world_size           : {result.world_size}")
    print(f"reduce_scatter_steps : {result.reduce_scatter_steps}")
    print(f"allgather_steps      : {result.allgather_steps}")
    print(f"total_steps          : {result.total_steps}")
    print(f"expected_result      : {result.expected_result}")

    if result.is_correct:
        print("[PASS] Ring simulation correctness check passed.")
    else:
        print("[FAIL] Ring simulation correctness check failed.")

        print()
        print("Expected:")
        print(result.expected_result)

        print()
        print("Final workers:")
        print(format_workers(result.final_workers))

    print("=" * 80)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace:
            命令行参数对象。
    """

    parser = argparse.ArgumentParser(
        description="Simulate standard Ring All-Reduce in a single process."
    )

    parser.add_argument(
        "--world_size",
        type=int,
        required=True,
        help="Number of workers participating in Ring All-Reduce.",
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
        help="Print initial workers, expected result, and final workers.",
    )

    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print intermediate worker states after each step.",
    )

    return parser.parse_args()


def main() -> None:
    """
    程序入口。

    执行流程：
        1. 读取命令行参数；
        2. 初始化 worker 数据；
        3. 执行 Ring All-Reduce 单进程模拟；
        4. 检查最终结果是否正确；
        5. 打印摘要。
    """

    args = parse_args()

    result = simulate_ring_allreduce(
        world_size=args.world_size,
        value_scale=args.value_scale,
        verbose=args.verbose,
        trace=args.trace,
    )

    print_summary(result)

    if not result.is_correct:
        raise SystemExit(1)


if __name__ == "__main__":
    main()