#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/simulator/ibing_sim.py

文件作用：
    该文件用于在单进程中模拟 IBing All-Reduce 的完整数据流。

    上一个文件 ibing_schedule.py 只负责打印调度表，即：
        1. 每个 rank 的左右邻居是谁；
        2. 每一步发送哪个 chunk；
        3. 每一步接收哪个 chunk；
        4. 当前 step 属于 Reduce-Scatter 还是 All-Gather。

    本文件在调度表的基础上进一步执行“数据流模拟”：
        1. 初始化每个 rank 的本地数据；
        2. 按 IBing 调度模拟每一步双向通信；
        3. 在 Reduce-Scatter 阶段执行加法 reduce；
        4. 在 All-Gather 阶段执行数据复制和转发；
        5. 验证所有 rank 最终是否得到相同的 All-Reduce 结果。

使用示例：
    只看最终正确性：
        python src/simulator/ibing_sim.py --world_size 5

    打印初始状态、理论结果和最终状态：
        python src/simulator/ibing_sim.py --world_size 5 --verbose

    打印每一步通信后的中间状态：
        python src/simulator/ibing_sim.py --world_size 5 --trace

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
    chunk 0: 0 + 10 + 20 + 30 + 40 = 100
    chunk 1: 1 + 11 + 21 + 31 + 41 = 105
    chunk 2: 2 + 12 + 22 + 32 + 42 = 110
    chunk 3: 3 + 13 + 23 + 33 + 43 = 115
    chunk 4: 4 + 14 + 24 + 34 + 44 = 120

最终所有 rank 都应该得到：
    [100, 105, 110, 115, 120]
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------
# 导入 ibing_schedule.py 中已经实现好的调度函数。
#
# 说明：
#   当你从项目根目录运行：
#       python src/simulator/ibing_sim.py --world_size 5
#
#   Python 会优先在当前脚本所在目录 src/simulator/ 中查找模块，
#   因此直接 import ibing_schedule 通常可以正常工作。
#
#   为了兼容部分 IDE 或测试框架，这里额外提供 src.simulator 的导入方式。
# ---------------------------------------------------------------------
try:
    from ibing_schedule import (
        IbingStepSchedule,
        get_phase,
        get_reduce_steps,
        get_total_steps,
        ibing_schedule,
        validate_world_size,
    )
except ImportError:
    from src.simulator.ibing_schedule import (
        IbingStepSchedule,
        get_phase,
        get_reduce_steps,
        get_total_steps,
        ibing_schedule,
        validate_world_size,
    )


# 当前模拟中，每个 chunk 用一个整数表示。
# 后续进入 MPI 或真实梯度模拟时，可以把 int 扩展为 numpy 数组。
Number = int

# workers[rank][chunk_id] 表示某个 rank 当前保存的某个 chunk 的值。
WorkerState = List[List[Number]]

# 消息字典的 key：
#   (source_rank, destination_rank, direction)
MessageKey = Tuple[int, int, str]

# 方向标记：
#   RIGHTWARD 表示从当前 rank 发给右邻居；
#   LEFTWARD 表示从当前 rank 发给左邻居。
RIGHTWARD = "rightward"
LEFTWARD = "leftward"


@dataclass(frozen=True)
class SimulatedMessage:
    """
    表示一次模拟通信中的消息。

    Attributes:
        step:
            当前通信步，0-based 编号。

        src_rank:
            消息发送方 rank。

        dst_rank:
            消息接收方 rank。

        direction:
            消息方向。
            RIGHTWARD 表示发送方把数据发给右邻居。
            LEFTWARD 表示发送方把数据发给左邻居。

        chunk_id:
            当前消息携带的数据块编号。

        value:
            当前消息携带的数据值。
            在本文件中是整数；后续可扩展为 numpy 数组。
    """

    step: int
    src_rank: int
    dst_rank: int
    direction: str
    chunk_id: int
    value: Number


@dataclass(frozen=True)
class SimulationResult:
    """
    保存一次 IBing 单进程模拟的结果。

    Attributes:
        world_size:
            worker 总数 N。

        total_steps:
            IBing 总通信步数，即 N - 1。

        reduce_steps:
            Reduce-Scatter 阶段步数。

        allgather_steps:
            All-Gather 阶段步数。

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
    total_steps: int
    reduce_steps: int
    allgather_steps: int
    initial_workers: WorkerState
    final_workers: WorkerState
    expected_result: List[Number]
    is_correct: bool


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
            workers[rank][chunk_id] 表示 rank 对应 chunk 的当前值。
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

    All-Reduce 的目标是让每个 rank 最终都得到所有 rank 的对应 chunk 之和。

    对于 chunk j，理论结果为：
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
    检查 IBing 模拟结果是否正确。

    正确条件：
        每个 rank 的最终数据都必须等于 expected_result。

    Args:
        final_workers:
            模拟结束后的 worker 数据。

        expected_result:
            理论 All-Reduce 结果。

    Returns:
        True 表示所有 rank 都正确；
        False 表示至少有一个 rank 不正确。
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


def build_messages_for_step(
    snapshot_workers: WorkerState,
    step: int,
    world_size: int,
) -> Dict[MessageKey, SimulatedMessage]:
    """
    根据当前 step 的调度表，构造本轮所有 rank 发出的消息。

    注意：
        这里使用 snapshot_workers，而不是直接使用 workers。

    原因：
        在真实 MPI 中，同一个 step 中所有 rank 的通信是并行发生的。
        如果在单进程模拟中边遍历 rank 边更新 workers，就可能导致后面的 rank
        读取到本 step 已经更新过的数据，从而破坏并行通信语义。

    因此，每个 step 开始时必须先复制一份快照 snapshot_workers。
    本 step 中所有发送数据都从 snapshot_workers 中读取。

    Args:
        snapshot_workers:
            当前 step 开始前的 worker 数据快照。

        step:
            当前通信步，0-based。

        world_size:
            worker 总数 N。

    Returns:
        messages:
            当前 step 中所有模拟消息。
            key 为 (src_rank, dst_rank, direction)。
    """

    messages: Dict[MessageKey, SimulatedMessage] = {}

    for rank in range(world_size):
        schedule: IbingStepSchedule = ibing_schedule(
            rank=rank,
            step=step,
            world_size=world_size,
        )

        # 方向 1：
        # 当前 rank 把 send_chunk_1 发送给右邻居。
        rightward_message = SimulatedMessage(
            step=step,
            src_rank=rank,
            dst_rank=schedule.right,
            direction=RIGHTWARD,
            chunk_id=schedule.send_chunk_1,
            value=snapshot_workers[rank][schedule.send_chunk_1],
        )

        # 方向 2：
        # 当前 rank 把 send_chunk_2 发送给左邻居。
        leftward_message = SimulatedMessage(
            step=step,
            src_rank=rank,
            dst_rank=schedule.left,
            direction=LEFTWARD,
            chunk_id=schedule.send_chunk_2,
            value=snapshot_workers[rank][schedule.send_chunk_2],
        )

        messages[
            (
                rightward_message.src_rank,
                rightward_message.dst_rank,
                rightward_message.direction,
            )
        ] = rightward_message

        messages[
            (
                leftward_message.src_rank,
                leftward_message.dst_rank,
                leftward_message.direction,
            )
        ] = leftward_message

    return messages


def get_expected_received_messages(
    messages: Dict[MessageKey, SimulatedMessage],
    schedule: IbingStepSchedule,
) -> Tuple[SimulatedMessage, SimulatedMessage]:
    """
    根据某个 rank 的调度表，从消息池中取出它本 step 应该收到的两条消息。

    对当前 rank 来说：

    方向 1：
        当前 rank 从左邻居接收 recv_chunk_1。
        这条消息应该是左邻居向右发送过来的。
        因此 key 为：
            (left_rank, current_rank, RIGHTWARD)

    方向 2：
        当前 rank 从右邻居接收 recv_chunk_2。
        这条消息应该是右邻居向左发送过来的。
        因此 key 为：
            (right_rank, current_rank, LEFTWARD)

    Args:
        messages:
            当前 step 的所有消息。

        schedule:
            当前 rank 在当前 step 的调度信息。

    Returns:
        message_from_left:
            当前 rank 从左邻居收到的消息。

        message_from_right:
            当前 rank 从右邻居收到的消息。

    Raises:
        RuntimeError:
            如果消息不存在，说明调度或消息构造存在错误。
    """

    key_from_left = (schedule.left, schedule.rank, RIGHTWARD)
    key_from_right = (schedule.right, schedule.rank, LEFTWARD)

    if key_from_left not in messages:
        raise RuntimeError(
            f"Missing message from left. "
            f"rank={schedule.rank}, step={schedule.step}, key={key_from_left}"
        )

    if key_from_right not in messages:
        raise RuntimeError(
            f"Missing message from right. "
            f"rank={schedule.rank}, step={schedule.step}, key={key_from_right}"
        )

    message_from_left = messages[key_from_left]
    message_from_right = messages[key_from_right]

    # 额外检查：收到的 chunk 编号必须和调度公式中的 recv_chunk 对齐。
    if message_from_left.chunk_id != schedule.recv_chunk_1:
        raise RuntimeError(
            f"Chunk mismatch from left. "
            f"rank={schedule.rank}, step={schedule.step}, "
            f"expected={schedule.recv_chunk_1}, got={message_from_left.chunk_id}"
        )

    if message_from_right.chunk_id != schedule.recv_chunk_2:
        raise RuntimeError(
            f"Chunk mismatch from right. "
            f"rank={schedule.rank}, step={schedule.step}, "
            f"expected={schedule.recv_chunk_2}, got={message_from_right.chunk_id}"
        )

    return message_from_left, message_from_right


def apply_reduce_scatter_update(
    workers: WorkerState,
    rank: int,
    message_from_left: SimulatedMessage,
    message_from_right: SimulatedMessage,
) -> None:
    """
    在 Reduce-Scatter 阶段更新当前 rank 的数据。

    Reduce-Scatter 阶段的规则：
        收到数据后，与本地对应 chunk 执行加法。

    即：
        workers[rank][recv_chunk] += received_value

    注意：
        某些 step 中，左右两个方向可能收到同一个 chunk_id。
        例如 world_size=5 时，rank 0 在 step=1 会从左右两个方向都收到 chunk 3。
        这种情况下应该连续加两次，因为它们代表来自两个方向的不同 partial sum。

    Args:
        workers:
            当前所有 worker 的数据状态。

        rank:
            当前更新的 rank。

        message_from_left:
            当前 rank 从左邻居收到的消息。

        message_from_right:
            当前 rank 从右邻居收到的消息。
    """

    workers[rank][message_from_left.chunk_id] += message_from_left.value
    workers[rank][message_from_right.chunk_id] += message_from_right.value


def apply_allgather_update(
    workers: WorkerState,
    rank: int,
    message_from_left: SimulatedMessage,
    message_from_right: SimulatedMessage,
) -> None:
    """
    在 All-Gather 阶段更新当前 rank 的数据。

    All-Gather 阶段的规则：
        收到数据后直接保存，不再做加法。

    即：
        workers[rank][recv_chunk] = received_value

    Args:
        workers:
            当前所有 worker 的数据状态。

        rank:
            当前更新的 rank。

        message_from_left:
            当前 rank 从左邻居收到的消息。

        message_from_right:
            当前 rank 从右邻居收到的消息。
    """

    workers[rank][message_from_left.chunk_id] = message_from_left.value
    workers[rank][message_from_right.chunk_id] = message_from_right.value


def simulate_one_step(
    workers: WorkerState,
    step: int,
    world_size: int,
) -> None:
    """
    模拟 IBing 的一个通信 step。

    该函数完成以下操作：
        1. 复制当前 workers 作为快照；
        2. 根据快照构造本 step 的所有发送消息；
        3. 每个 rank 从消息池中取出自己应该收到的两条消息；
        4. 根据当前阶段执行 Reduce-Scatter 或 All-Gather 更新。

    Args:
        workers:
            当前所有 worker 的数据状态。
            该对象会被原地修改。

        step:
            当前通信步，0-based。

        world_size:
            worker 总数 N。
    """

    # 关键点：
    #   必须使用 step 开始前的快照来构造消息。
    #   否则单进程顺序模拟会破坏真实并行通信语义。
    snapshot_workers = deepcopy(workers)

    messages = build_messages_for_step(
        snapshot_workers=snapshot_workers,
        step=step,
        world_size=world_size,
    )

    phase = get_phase(step=step, world_size=world_size)

    for rank in range(world_size):
        schedule = ibing_schedule(
            rank=rank,
            step=step,
            world_size=world_size,
        )

        message_from_left, message_from_right = get_expected_received_messages(
            messages=messages,
            schedule=schedule,
        )

        if phase == "Reduce-Scatter":
            apply_reduce_scatter_update(
                workers=workers,
                rank=rank,
                message_from_left=message_from_left,
                message_from_right=message_from_right,
            )
        elif phase == "All-Gather":
            apply_allgather_update(
                workers=workers,
                rank=rank,
                message_from_left=message_from_left,
                message_from_right=message_from_right,
            )
        else:
            raise RuntimeError(f"Unknown phase: {phase}")


def simulate_ibing_allreduce(
    world_size: int,
    value_scale: int = 10,
    verbose: bool = False,
    trace: bool = False,
) -> SimulationResult:
    """
    模拟完整的 IBing All-Reduce 流程。

    Args:
        world_size:
            worker 总数 N。

        value_scale:
            初始化 worker 数据时使用的数据间隔。

        verbose:
            是否打印初始数据、理论结果和最终数据。

        trace:
            是否打印每个 step 之后的中间状态。
            trace=True 时会自动打印更详细的信息。

    Returns:
        SimulationResult:
            包含模拟结果、理论结果和正确性检查结果。
    """

    validate_world_size(world_size)

    total_steps = get_total_steps(world_size)
    reduce_steps = get_reduce_steps(world_size)
    allgather_steps = total_steps - reduce_steps

    initial_workers = create_initial_workers(
        world_size=world_size,
        value_scale=value_scale,
    )

    expected_result = compute_expected_result(initial_workers)

    # workers 是后续会被不断更新的模拟状态。
    workers = deepcopy(initial_workers)

    if verbose or trace:
        print("=" * 80)
        print("IBing All-Reduce Single-Process Simulation")
        print("=" * 80)
        print(f"world_size      : {world_size}")
        print(f"total_steps     : {total_steps}")
        print(f"reduce_steps    : {reduce_steps}")
        print(f"allgather_steps : {allgather_steps}")
        print("=" * 80)
        print()

        print_workers("Initial workers", workers)
        print(f"Expected all-reduce result:\n{expected_result}")
        print()

    for step in range(total_steps):
        phase = get_phase(step=step, world_size=world_size)

        simulate_one_step(
            workers=workers,
            step=step,
            world_size=world_size,
        )

        if trace:
            print_workers(
                title=f"After step {step + 1}/{total_steps} [{phase}]",
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
        total_steps=total_steps,
        reduce_steps=reduce_steps,
        allgather_steps=allgather_steps,
        initial_workers=initial_workers,
        final_workers=workers,
        expected_result=expected_result,
        is_correct=is_correct,
    )


def print_summary(result: SimulationResult) -> None:
    """
    打印模拟结果摘要。

    Args:
        result:
            IBing 模拟结果。
    """

    print("=" * 80)
    print("Simulation Summary")
    print("=" * 80)
    print(f"world_size      : {result.world_size}")
    print(f"total_steps     : {result.total_steps}")
    print(f"reduce_steps    : {result.reduce_steps}")
    print(f"allgather_steps : {result.allgather_steps}")
    print(f"expected_result : {result.expected_result}")

    if result.is_correct:
        print("[PASS] IBing simulation correctness check passed.")
    else:
        print("[FAIL] IBing simulation correctness check failed.")

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
        description="Simulate IBing All-Reduce in a single process."
    )

    parser.add_argument(
        "--world_size",
        type=int,
        required=True,
        help="Number of workers participating in IBing All-Reduce.",
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
        3. 执行 IBing 单进程模拟；
        4. 检查最终结果是否正确；
        5. 打印摘要。
    """

    args = parse_args()

    result = simulate_ibing_allreduce(
        world_size=args.world_size,
        value_scale=args.value_scale,
        verbose=args.verbose,
        trace=args.trace,
    )

    print_summary(result)

    # 如果正确性检查失败，使用非 0 退出码。
    # 这样后续写自动化测试或 shell 脚本时，可以直接根据退出码判断是否通过。
    if not result.is_correct:
        raise SystemExit(1)


if __name__ == "__main__":
    main()