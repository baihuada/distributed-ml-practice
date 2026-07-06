#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/simulator/ibing_schedule.py

文件作用：
    该文件用于模拟并打印 IBing All-Reduce 的通信调度过程。

    它不执行真实 MPI 通信，也不执行梯度求和，只做第一步复现中最重要的事情：
        1. 根据 rank 计算左右邻居；
        2. 根据 rank 和 step 计算每一步发送/接收的 chunk 编号；
        3. 标记当前 step 属于 Reduce-Scatter 还是 All-Gather；
        4. 打印完整调度表，帮助检查 IBing 的通信顺序是否正确。

使用示例：
    打印所有 rank 在 N=5 时的完整调度：
        python src/simulator/ibing_schedule.py --world_size 5

    只打印 rank 0 的调度：
        python src/simulator/ibing_schedule.py --world_size 5 --rank 0

核心公式：
    对于当前节点 rank=r，当前通信步 step=i，总节点数 N：

    方向 1：向右发送，从左接收
        send_chunk_1 = (r - i + N) % N
        recv_chunk_1 = (r - i - 1 + N) % N

    方向 2：向左发送，从右接收
        send_chunk_2 = (r + i + N + 1) % N
        recv_chunk_2 = (r + i + N + 2) % N
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class NeighborInfo:
    """
    保存某个 rank 的左右邻居信息。

    Attributes:
        rank: 当前 worker 的编号。
        left: 当前 worker 的左邻居编号。
        right: 当前 worker 的右邻居编号。
    """

    rank: int
    left: int
    right: int


@dataclass(frozen=True)
class IbingStepSchedule:
    """
    保存某个 rank 在某个 step 下的 IBing 通信调度信息。

    Attributes:
        rank: 当前 worker 的编号。
        step: 当前通信步，使用 0-based 编号。
        display_step: 用于打印显示的通信步，使用 1-based 编号。
        phase: 当前 step 所属阶段，Reduce-Scatter 或 All-Gather。
        left: 当前 rank 的左邻居。
        right: 当前 rank 的右邻居。
        send_chunk_1: 方向 1 中，当前 rank 发送给右邻居的 chunk 编号。
        recv_chunk_1: 方向 1 中，当前 rank 从左邻居接收的 chunk 编号。
        send_chunk_2: 方向 2 中，当前 rank 发送给左邻居的 chunk 编号。
        recv_chunk_2: 方向 2 中，当前 rank 从右邻居接收的 chunk 编号。
    """

    rank: int
    step: int
    display_step: int
    phase: str
    left: int
    right: int
    send_chunk_1: int
    recv_chunk_1: int
    send_chunk_2: int
    recv_chunk_2: int

    def as_dict(self) -> Dict[str, int | str]:
        """
        将调度结果转换为字典形式。

        Returns:
            包含当前 step 调度信息的字典。
        """

        return {
            "rank": self.rank,
            "step": self.step,
            "display_step": self.display_step,
            "phase": self.phase,
            "left": self.left,
            "right": self.right,
            "send_chunk_1": self.send_chunk_1,
            "recv_chunk_1": self.recv_chunk_1,
            "send_chunk_2": self.send_chunk_2,
            "recv_chunk_2": self.recv_chunk_2,
        }


def validate_world_size(world_size: int) -> None:
    """
    检查 worker 总数是否合法。

    Args:
        world_size: 参与 All-Reduce 的 worker 数量。

    Raises:
        ValueError: 当 world_size 小于 2 时抛出异常。
    """

    if world_size < 2:
        raise ValueError("world_size must be >= 2.")


def validate_rank(rank: int, world_size: int) -> None:
    """
    检查 rank 编号是否合法。

    Args:
        rank: 当前 worker 编号。
        world_size: worker 总数。

    Raises:
        ValueError: 当 rank 不在 [0, world_size - 1] 范围内时抛出异常。
    """

    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"rank must be in [0, {world_size - 1}], but got rank={rank}."
        )


def validate_step(step: int, world_size: int) -> None:
    """
    检查通信步编号是否合法。

    IBing 总通信步数为 N - 1。
    因此 step 的合法范围是 [0, N - 2]。

    Args:
        step: 当前通信步，0-based。
        world_size: worker 总数。

    Raises:
        ValueError: 当 step 不在合法范围内时抛出异常。
    """

    total_steps = world_size - 1

    if step < 0 or step >= total_steps:
        raise ValueError(
            f"step must be in [0, {total_steps - 1}], but got step={step}."
        )


def get_neighbors(rank: int, world_size: int) -> NeighborInfo:
    """
    计算当前 rank 在环形拓扑中的左右邻居。

    IBing 保持 Ring All-Reduce 的基本环形拓扑。
    每个 worker 只需要知道自己的左邻居和右邻居。

    对于 rank=r：
        left  = (r - 1 + N) % N
        right = (r + 1) % N

    Args:
        rank: 当前 worker 编号。
        world_size: worker 总数 N。

    Returns:
        NeighborInfo 对象，包含当前 rank 的左右邻居。
    """

    validate_world_size(world_size)
    validate_rank(rank, world_size)

    left = (rank - 1 + world_size) % world_size
    right = (rank + 1) % world_size

    return NeighborInfo(rank=rank, left=left, right=right)


def get_total_steps(world_size: int) -> int:
    """
    计算 IBing 的总通信步数。

    标准 Ring All-Reduce 的通信步数为：
        2 * (N - 1)

    IBing 通过双向交错通信，将总通信步数减少为：
        N - 1

    Args:
        world_size: worker 总数 N。

    Returns:
        IBing 的总通信步数。
    """

    validate_world_size(world_size)
    return world_size - 1


def get_reduce_steps(world_size: int) -> int:
    """
    计算 Reduce-Scatter 阶段的通信步数。

    论文中 N=5 时，前 2 步为 Reduce-Scatter，后 2 步为 All-Gather。
    对一般 N，这里采用：
        reduce_steps = N // 2

    例如：
        N=5，总步数为 4，Reduce-Scatter 步数为 2；
        N=4，总步数为 3，Reduce-Scatter 步数为 2；
        N=8，总步数为 7，Reduce-Scatter 步数为 4。

    Args:
        world_size: worker 总数 N。

    Returns:
        Reduce-Scatter 阶段步数。
    """

    validate_world_size(world_size)
    return world_size // 2


def get_phase(step: int, world_size: int) -> str:
    """
    判断当前 step 属于 Reduce-Scatter 还是 All-Gather。

    Reduce-Scatter 阶段：
        收到数据后，需要与本地对应 chunk 做加法。

    All-Gather 阶段：
        收到数据后，只保存或转发，不再做加法。

    Args:
        step: 当前通信步，0-based。
        world_size: worker 总数 N。

    Returns:
        字符串 "Reduce-Scatter" 或 "All-Gather"。
    """

    validate_step(step, world_size)

    reduce_steps = get_reduce_steps(world_size)

    if step < reduce_steps:
        return "Reduce-Scatter"

    return "All-Gather"


def ibing_schedule(rank: int, step: int, world_size: int) -> IbingStepSchedule:
    """
    计算某个 rank 在某个 step 中的 IBing 通信调度。

    每个 step 中，当前 rank 同时执行两个方向的通信：

    方向 1：
        当前 rank 向右邻居发送 send_chunk_1；
        当前 rank 从左邻居接收 recv_chunk_1。

    方向 2：
        当前 rank 向左邻居发送 send_chunk_2；
        当前 rank 从右邻居接收 recv_chunk_2。

    Args:
        rank: 当前 worker 编号。
        step: 当前通信步，0-based。
        world_size: worker 总数 N。

    Returns:
        IbingStepSchedule 对象，包含当前 step 的完整调度信息。
    """

    validate_world_size(world_size)
    validate_rank(rank, world_size)
    validate_step(step, world_size)

    neighbor_info = get_neighbors(rank, world_size)

    recv_chunk_1 = (rank - step - 1 + world_size) % world_size
    send_chunk_1 = (rank - step + world_size) % world_size

    recv_chunk_2 = (rank + step + world_size + 2) % world_size
    send_chunk_2 = (rank + step + world_size + 1) % world_size

    phase = get_phase(step, world_size)

    return IbingStepSchedule(
        rank=rank,
        step=step,
        display_step=step + 1,
        phase=phase,
        left=neighbor_info.left,
        right=neighbor_info.right,
        send_chunk_1=send_chunk_1,
        recv_chunk_1=recv_chunk_1,
        send_chunk_2=send_chunk_2,
        recv_chunk_2=recv_chunk_2,
    )


def build_rank_schedule(rank: int, world_size: int) -> List[IbingStepSchedule]:
    """
    构造某个 rank 在所有通信 step 中的调度表。

    Args:
        rank: 当前 worker 编号。
        world_size: worker 总数 N。

    Returns:
        当前 rank 的完整 IBing 调度列表。
    """

    validate_world_size(world_size)
    validate_rank(rank, world_size)

    total_steps = get_total_steps(world_size)

    return [
        ibing_schedule(rank=rank, step=step, world_size=world_size)
        for step in range(total_steps)
    ]


def build_all_schedules(
    world_size: int,
    target_rank: Optional[int] = None,
) -> Dict[int, List[IbingStepSchedule]]:
    """
    构造一个或多个 rank 的完整调度表。

    如果 target_rank 为 None，则构造所有 rank 的调度表。
    如果 target_rank 不为 None，则只构造指定 rank 的调度表。

    Args:
        world_size: worker 总数 N。
        target_rank: 需要打印的 rank；如果为 None，则打印所有 rank。

    Returns:
        字典形式的调度表：
            key 是 rank；
            value 是该 rank 的所有 step 调度信息。
    """

    validate_world_size(world_size)

    if target_rank is not None:
        validate_rank(target_rank, world_size)
        ranks = [target_rank]
    else:
        ranks = list(range(world_size))

    return {
        rank: build_rank_schedule(rank=rank, world_size=world_size)
        for rank in ranks
    }


def format_schedule_entry(entry: IbingStepSchedule, world_size: int) -> str:
    """
    将单个 step 的调度结果格式化为可读字符串。

    Args:
        entry: 某个 rank 在某个 step 的调度信息。
        world_size: worker 总数 N。

    Returns:
        格式化后的字符串。
    """

    total_steps = get_total_steps(world_size)

    lines = [
        f"rank={entry.rank}, "
        f"step={entry.display_step}/{total_steps}, "
        f"phase={entry.phase}",
        f"  neighbors: left={entry.left}, right={entry.right}",
        "  direction 1: "
        f"recv chunk {entry.recv_chunk_1} from left rank {entry.left}; "
        f"send chunk {entry.send_chunk_1} to right rank {entry.right}",
        "  direction 2: "
        f"recv chunk {entry.recv_chunk_2} from right rank {entry.right}; "
        f"send chunk {entry.send_chunk_2} to left rank {entry.left}",
    ]

    return "\n".join(lines)


def print_schedule(world_size: int, target_rank: Optional[int] = None) -> None:
    """
    打印 IBing 调度表。

    Args:
        world_size: worker 总数 N。
        target_rank: 指定打印某个 rank；如果为 None，则打印所有 rank。
    """

    validate_world_size(world_size)

    total_steps = get_total_steps(world_size)
    reduce_steps = get_reduce_steps(world_size)
    gather_steps = total_steps - reduce_steps

    print("=" * 80)
    print("IBing All-Reduce Schedule")
    print("=" * 80)
    print(f"world_size      : {world_size}")
    print(f"total_steps     : {total_steps}")
    print(f"reduce_steps    : {reduce_steps}")
    print(f"allgather_steps : {gather_steps}")

    if world_size == 2:
        print()
        print(
            "[Warning] world_size=2 is a degenerate ring: "
            "left and right neighbors are the same rank."
        )
        print(
            "          The schedule can still be printed, but real bidirectional "
            "communication should usually be tested with N >= 3."
        )

    print("=" * 80)

    schedules = build_all_schedules(
        world_size=world_size,
        target_rank=target_rank,
    )

    for rank, rank_schedule in schedules.items():
        print()
        print("-" * 80)
        print(f"Schedule for rank {rank}")
        print("-" * 80)

        for entry in rank_schedule:
            print(format_schedule_entry(entry, world_size))
            print()


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace，包含命令行输入参数。
    """

    parser = argparse.ArgumentParser(
        description="Print IBing All-Reduce communication schedule."
    )

    parser.add_argument(
        "--world_size",
        type=int,
        required=True,
        help="Number of workers participating in IBing All-Reduce.",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help=(
            "Optional target rank. "
            "If not set, schedules for all ranks will be printed."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    主函数。

    程序入口：
        1. 读取命令行参数；
        2. 检查 world_size 和 rank；
        3. 打印 IBing 调度表。
    """

    args = parse_args()

    print_schedule(
        world_size=args.world_size,
        target_rank=args.rank,
    )


if __name__ == "__main__":
    main()