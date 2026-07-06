"""
一键运行 Single、PS、DDP 实验矩阵。

默认运行：
1. Single Process
2. PS workers=2
3. DDP workers=2

如果加上 --include-4-workers，则额外运行：
4. PS workers=4
5. DDP workers=4

运行示例：
python -m scripts.run_all --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42 --device cpu
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import List


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """

    parser = argparse.ArgumentParser(
        description="Run all experiments: Single, PS, and DDP"
    )

    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results/raw")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--include-4-workers", action="store_true")
    parser.add_argument("--no-progress", action="store_true")

    return parser.parse_args()


def build_common_args(args: argparse.Namespace) -> List[str]:
    """
    构造各实验共用参数。
    """

    return [
        "--dataset",
        args.dataset,
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--test-batch-size",
        str(args.test_batch_size),
        "--lr",
        str(args.lr),
        "--momentum",
        str(args.momentum),
        "--weight-decay",
        str(args.weight_decay),
        "--hidden-dim",
        str(args.hidden_dim),
        "--dropout",
        str(args.dropout),
        "--seed",
        str(args.seed),
        "--data-dir",
        args.data_dir,
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
    ]


def run_command(name: str, command: List[str]) -> None:
    """
    运行一条实验命令。
    """

    print("=" * 80)
    print(f"开始实验: {name}")
    print("=" * 80)
    print(" ".join(command))
    print("=" * 80)

    start_time = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed_time = time.perf_counter() - start_time

    print("=" * 80)
    print(f"实验完成: {name}")
    print(f"耗时: {elapsed_time:.2f}s")
    print("=" * 80)


def build_single_command(args: argparse.Namespace) -> List[str]:
    """
    构造 Single 实验命令。
    """

    command = [
        sys.executable,
        "-m",
        "scripts.run_single",
    ]

    command.extend(build_common_args(args))

    if args.no_progress:
        command.append("--no-progress")

    return command


def build_ps_command(args: argparse.Namespace, num_workers: int) -> List[str]:
    """
    构造 PS 实验命令。
    """

    command = [
        sys.executable,
        "-m",
        "scripts.run_ps",
    ]

    command.extend(build_common_args(args))

    command.extend(
        [
            "--num-workers",
            str(num_workers),
            "--dataloader-num-workers",
            str(args.dataloader_num_workers),
        ]
    )

    return command


def build_ddp_command(args: argparse.Namespace, num_workers: int) -> List[str]:
    """
    构造 DDP 实验命令。
    """

    command = [
        sys.executable,
        "-m",
        "scripts.run_ddp",
    ]

    command.extend(build_common_args(args))

    command.extend(
        [
            "--num-workers",
            str(num_workers),
            "--dataloader-num-workers",
            str(args.dataloader_num_workers),
            "--backend",
            args.backend,
        ]
    )

    if args.no_progress:
        command.append("--no-progress")

    return command


def main() -> None:
    """
    主函数。
    """

    args = parse_args()

    experiments = [
        (
            "Single Process",
            build_single_command(args),
        ),
        (
            "Parameter Server workers=2",
            build_ps_command(args, num_workers=2),
        ),
        (
            "DDP / AllReduce workers=2",
            build_ddp_command(args, num_workers=2),
        ),
    ]

    if args.include_4_workers:
        experiments.extend(
            [
                (
                    "Parameter Server workers=4",
                    build_ps_command(args, num_workers=4),
                ),
                (
                    "DDP / AllReduce workers=4",
                    build_ddp_command(args, num_workers=4),
                ),
            ]
        )

    total_start_time = time.perf_counter()

    for name, command in experiments:
        run_command(name, command)

    total_elapsed_time = time.perf_counter() - total_start_time

    print("=" * 80)
    print("全部实验运行完成")
    print("=" * 80)
    print(f"实验数量: {len(experiments)}")
    print(f"总耗时: {total_elapsed_time:.2f}s")
    print("原始结果目录: results/raw/")
    print("=" * 80)


if __name__ == "__main__":
    main()