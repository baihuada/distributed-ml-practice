"""
运行 Single Process baseline。

运行示例：
python -m scripts.run_single --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42 --device cpu
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """

    parser = argparse.ArgumentParser(
        description="Run Single Process baseline"
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-progress", action="store_true")

    return parser.parse_args()


def build_command(args: argparse.Namespace) -> List[str]:
    """
    构造 Single baseline 运行命令。
    """

    command = [
        sys.executable,
        "-m",
        "single.train_single",
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
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]

    if args.no_progress:
        command.append("--no-progress")

    return command


def main() -> None:
    """
    主函数。
    """

    args = parse_args()
    command = build_command(args)

    print("=" * 80)
    print("运行 Single Process baseline")
    print("=" * 80)
    print(" ".join(command))
    print("=" * 80)

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()