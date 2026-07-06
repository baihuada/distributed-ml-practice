"""
run_all_experiments.py

功能：
1. 一键运行 DML-Bench Core 版的全部主要实验；
2. 包括 Centralized SGD、Sync-SGD、Local SGD、Async-SGD；
3. 包括 Local SGD 不同 local_steps 对比；
4. 包括 Sync-SGD / Async-SGD 在 straggler 场景下的对比；
5. 自动创建 results/raw、results/tables、results/figures、report 等目录；
6. 使用 Python subprocess 调用各个实验模块；
7. 若某个实验失败，脚本会立即停止；
8. 最后自动运行 straggler_analysis.py 和 summary_all_experiments.py。

运行方式：
python scripts/run_all_experiments.py

快速测试：
python scripts/run_all_experiments.py --epochs 1

完整运行：
python scripts/run_all_experiments.py --epochs 10
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def ensure_dirs() -> None:
    """
    创建实验输出目录。
    """
    dirs = [
        "results/raw",
        "results/tables",
        "results/figures",
        "scripts",
        "report",
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def check_required_files() -> None:
    """
    检查关键文件是否存在。
    """
    required_files = [
        "dmlbench/algorithms/centralized_sgd.py",
        "dmlbench/algorithms/sync_sgd.py",
        "dmlbench/algorithms/local_sgd.py",
        "dmlbench/algorithms/async_sgd.py",
        "scripts/straggler_analysis.py",
        "scripts/summary_all_experiments.py",
    ]

    missing_files = []

    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print("=" * 80)
        print("Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("=" * 80)
        raise FileNotFoundError("Some required files are missing.")

    print("[Check] All required files exist.")


def run_command(title: str, cmd: List[str]) -> None:
    """
    运行单个实验命令。

    参数：
        title:
            当前实验标题。
        cmd:
            subprocess 命令列表。
    """
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print("Command:")
    print(" ".join(cmd))
    print("-" * 80)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print()
        print("=" * 80)
        print(f"Experiment failed: {title}")
        print(f"Return code: {result.returncode}")
        print("=" * 80)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

    print("-" * 80)
    print(f"Finished: {title}")


def build_python_module_cmd(module: str, args: List[str]) -> List[str]:
    """
    构造 python -m xxx 命令。
    """
    return [sys.executable, "-m", module] + args


def build_python_script_cmd(script: str, args: List[str]) -> List[str]:
    """
    构造 python script.py 命令。
    """
    return [sys.executable, script] + args


def run_all_experiments(
    model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    seed: int,
    run_severe_straggler: bool = True,
) -> None:
    """
    运行全部实验。
    """
    ensure_dirs()
    check_required_files()

    # 减少 PyTorch CUDA deterministic warning。
    # 即使仍然出现 warning，也不影响实验运行。
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    common_basic_args = [
        "--model", model,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--seed", str(seed),
    ]

    common_worker_args = common_basic_args + [
        "--num-workers", str(num_workers),
    ]

    # 1. Centralized SGD
    run_command(
        title="[1/9] Centralized SGD",
        cmd=build_python_module_cmd(
            module="dmlbench.algorithms.centralized_sgd",
            args=common_basic_args,
        ),
    )

    # 2. Sync-SGD equal delay
    run_command(
        title="[2/9] Sync-SGD, equal delay = 1,1,1,1",
        cmd=build_python_module_cmd(
            module="dmlbench.algorithms.sync_sgd",
            args=common_worker_args + [
                "--worker-delays", "1,1,1,1",
            ],
        ),
    )

    # 3. Sync-SGD straggler delay
    run_command(
        title="[3/9] Sync-SGD, straggler delay = 1,1,1,5",
        cmd=build_python_module_cmd(
            module="dmlbench.algorithms.sync_sgd",
            args=common_worker_args + [
                "--worker-delays", "1,1,1,5",
            ],
        ),
    )

    # 4. Local SGD with different local steps
    local_steps_list = [1, 5, 10, 20]

    for local_steps in local_steps_list:
        run_command(
            title=f"[4/9] Local SGD, local_steps = {local_steps}",
            cmd=build_python_module_cmd(
                module="dmlbench.algorithms.local_sgd",
                args=common_worker_args + [
                    "--local-steps", str(local_steps),
                ],
            ),
        )

    # 5. Async-SGD equal delay
    run_command(
        title="[5/9] Async-SGD, equal delay = 1,1,1,1",
        cmd=build_python_module_cmd(
            module="dmlbench.algorithms.async_sgd",
            args=common_worker_args + [
                "--worker-delays", "1,1,1,1",
            ],
        ),
    )

    # 6. Async-SGD straggler delay
    run_command(
        title="[6/9] Async-SGD, straggler delay = 1,1,1,5",
        cmd=build_python_module_cmd(
            module="dmlbench.algorithms.async_sgd",
            args=common_worker_args + [
                "--worker-delays", "1,1,1,5",
            ],
        ),
    )

    # 7. Optional severe straggler
    if run_severe_straggler:
        run_command(
            title="[7/9] Async-SGD, severe straggler delay = 1,1,1,10",
            cmd=build_python_module_cmd(
                module="dmlbench.algorithms.async_sgd",
                args=common_worker_args + [
                    "--worker-delays", "1,1,1,10",
                ],
            ),
        )
    else:
        print()
        print("=" * 80)
        print("[7/9] Skip severe straggler experiment.")
        print("=" * 80)

    # 8. Straggler summary
    run_command(
        title="[8/9] Straggler summary",
        cmd=build_python_script_cmd(
            script="scripts/straggler_analysis.py",
            args=[
                "--model", model,
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--lr", str(lr),
                "--num-workers", str(num_workers),
                "--seed", str(seed),
            ],
        ),
    )

    # 9. Full summary
    run_command(
        title="[9/9] Full experiment summary",
        cmd=build_python_script_cmd(
            script="scripts/summary_all_experiments.py",
            args=[
                "--raw-dir", "results/raw",
                "--tables-dir", "results/tables",
                "--figures-dir", "results/figures",
            ],
        ),
    )

    print()
    print("=" * 80)
    print("All experiments finished successfully.")
    print("=" * 80)
    print("Summary table:")
    print("  results/tables/summary.csv")
    print("Straggler summary table:")
    print("  results/tables/straggler_summary.csv")
    print("Figures directory:")
    print("  results/figures")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="Run all DML-Bench experiments."
    )

    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "logistic"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--skip-severe-straggler",
        action="store_true",
        help="Skip Async-SGD delay=1,1,1,10 experiment.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_all_experiments(
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        run_severe_straggler=not args.skip_severe_straggler,
    )