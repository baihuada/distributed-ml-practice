"""
Parameter Server 训练入口。

运行示例：
python -m ps.train_ps --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42 --num-workers 2

本文件负责：
1. 解析命令行参数；
2. 创建多进程通信队列；
3. 启动 worker 进程；
4. 启动 server 进程；
5. 等待训练完成；
6. 输出日志路径和最终结果。

注意：
Windows 下必须使用 if __name__ == "__main__": 保护多进程启动逻辑。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import torch.multiprocessing as mp

from common.datasets import get_dataset
from common.seed import set_seed
from ps.server import ps_server_loop
from ps.worker import ps_worker_loop


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    返回
    ----
    argparse.Namespace
        命令行参数对象。
    """

    parser = argparse.ArgumentParser(
        description="Multiprocessing Parameter Server training for PS vs AllReduce project"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist", "fashion-mnist", "fashion"],
        help="数据集名称",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "logistic", "lr", "logreg"],
        help="模型名称",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练 epoch 数",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="每个 worker 的本地 batch size",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        help="测试 batch size",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="学习率",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="SGD 动量系数，第一版默认不使用动量",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="权重衰减系数",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="MLP 隐藏层维度",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="MLP dropout 概率，第一版默认不使用 dropout",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="数据集保存目录",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/raw",
        help="CSV 日志输出目录",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Parameter Server worker 数量",
    )

    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="每个 worker 内部 DataLoader 的数据加载进程数。Windows 下建议保持 0",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="server 和 worker 使用的设备。PS 第一版建议使用 cpu",
    )

    parser.add_argument(
        "--start-method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="多进程启动方式。Windows 只支持 spawn",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="主进程等待 server 返回结果的超时时间，单位秒",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """
    检查命令行参数是否合法。

    参数
    ----
    args : argparse.Namespace
        命令行参数对象。
    """

    if args.epochs <= 0:
        raise ValueError(f"epochs 必须为正整数，但得到 {args.epochs}")

    if args.batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，但得到 {args.batch_size}")

    if args.test_batch_size <= 0:
        raise ValueError(
            f"test_batch_size 必须为正整数，但得到 {args.test_batch_size}"
        )

    if args.lr <= 0:
        raise ValueError(f"lr 必须为正数，但得到 {args.lr}")

    if args.num_workers <= 0:
        raise ValueError(f"num_workers 必须为正整数，但得到 {args.num_workers}")

    if args.dataloader_num_workers < 0:
        raise ValueError(
            "dataloader_num_workers 不能为负数，"
            f"但得到 {args.dataloader_num_workers}"
        )

    if args.timeout <= 0:
        raise ValueError(f"timeout 必须为正数，但得到 {args.timeout}")


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    将命令行参数转换为配置字典。

    参数
    ----
    args : argparse.Namespace
        命令行参数对象。

    返回
    ----
    Dict[str, Any]
        可传递给 server 和 worker 的配置字典。
    """

    return {
        "dataset": args.dataset,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "seed": args.seed,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "dataloader_num_workers": args.dataloader_num_workers,
        "device": args.device,
    }


def prepare_dataset(dataset_name: str, data_dir: str) -> None:
    """
    在主进程中预下载数据集。

    参数
    ----
    dataset_name : str
        数据集名称。
    data_dir : str
        数据集保存目录。

    说明
    ----
    如果不预下载数据集，多个 worker 进程可能同时触发下载，
    在 Windows 下容易出现文件占用或下载冲突。
    """

    print("检查并准备数据集...")

    _ = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train=True,
        download=True,
    )

    _ = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train=False,
        download=True,
    )

    print("数据集准备完成")


def create_queues(ctx: Any, num_workers: int) -> Dict[str, Any]:
    """
    创建 server-worker 通信队列。

    参数
    ----
    ctx : multiprocessing context
        多进程上下文。
    num_workers : int
        worker 数量。

    返回
    ----
    Dict[str, Any]
        包含 task_queues、result_queue、log_queue 的字典。
    """

    task_queues = [ctx.Queue() for _ in range(num_workers)]

    result_queue = ctx.Queue()
    log_queue = ctx.Queue()

    return {
        "task_queues": task_queues,
        "result_queue": result_queue,
        "log_queue": log_queue,
    }


def start_worker_processes(
    ctx: Any,
    num_workers: int,
    task_queues: List[Any],
    result_queue: Any,
    config: Dict[str, Any],
) -> List[Any]:
    """
    启动所有 worker 进程。

    参数
    ----
    ctx : multiprocessing context
        多进程上下文。
    num_workers : int
        worker 数量。
    task_queues : List[Any]
        每个 worker 对应一个任务队列。
    result_queue : Any
        所有 worker 共用的结果队列。
    config : Dict[str, Any]
        训练配置字典。

    返回
    ----
    List[Any]
        worker 进程列表。
    """

    worker_processes = []

    for worker_id in range(num_workers):
        process = ctx.Process(
            target=ps_worker_loop,
            args=(
                worker_id,
                num_workers,
                task_queues[worker_id],
                result_queue,
                config,
            ),
            name=f"ps-worker-{worker_id}",
        )

        process.start()
        worker_processes.append(process)

    return worker_processes


def start_server_process(
    ctx: Any,
    num_workers: int,
    task_queues: List[Any],
    result_queue: Any,
    log_queue: Any,
    config: Dict[str, Any],
) -> Any:
    """
    启动 server 进程。

    参数
    ----
    ctx : multiprocessing context
        多进程上下文。
    num_workers : int
        worker 数量。
    task_queues : List[Any]
        每个 worker 对应一个任务队列。
    result_queue : Any
        worker 上传梯度结果的队列。
    log_queue : Any
        server 向主进程返回训练状态的队列。
    config : Dict[str, Any]
        训练配置字典。

    返回
    ----
    Any
        server 进程对象。
    """

    server_process = ctx.Process(
        target=ps_server_loop,
        args=(
            num_workers,
            task_queues,
            result_queue,
            log_queue,
            config,
        ),
        name="parameter-server",
    )

    server_process.start()

    return server_process


def terminate_processes(processes: List[Any]) -> None:
    """
    终止仍然存活的子进程。

    参数
    ----
    processes : List[Any]
        需要检查并终止的进程列表。
    """

    for process in processes:
        if process.is_alive():
            process.terminate()

    for process in processes:
        process.join(timeout=5)


def join_processes(processes: List[Any], timeout: float = 10.0) -> None:
    """
    等待子进程退出。

    参数
    ----
    processes : List[Any]
        子进程列表。
    timeout : float
        每个进程 join 的最长等待时间。
    """

    for process in processes:
        process.join(timeout=timeout)


def print_process_status(processes: List[Any]) -> None:
    """
    打印子进程退出状态。

    参数
    ----
    processes : List[Any]
        子进程列表。
    """

    print("子进程状态：")

    for process in processes:
        print(
            f"{process.name}: "
            f"pid={process.pid}, "
            f"exitcode={process.exitcode}, "
            f"alive={process.is_alive()}"
        )


def wait_for_server_result(
    log_queue: Any,
    timeout: float,
) -> Dict[str, Any]:
    """
    等待 server 返回最终训练状态。

    参数
    ----
    log_queue : Any
        server 向主进程发送结果的队列。
    timeout : float
        等待超时时间，单位秒。

    返回
    ----
    Dict[str, Any]
        server 返回的结果消息。
    """

    try:
        message = log_queue.get(timeout=timeout)
    except Exception as exc:
        raise RuntimeError(
            "主进程等待 server 结果超时或失败。"
            "请检查 worker / server 是否已经报错。"
        ) from exc

    return message


def run_parameter_server_training(args: argparse.Namespace) -> None:
    """
    运行完整 Parameter Server 训练。

    参数
    ----
    args : argparse.Namespace
        命令行参数对象。
    """

    validate_args(args)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepare_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
    )

    config = args_to_config(args)

    ctx = mp.get_context(args.start_method)

    queues = create_queues(
        ctx=ctx,
        num_workers=args.num_workers,
    )

    task_queues = queues["task_queues"]
    result_queue = queues["result_queue"]
    log_queue = queues["log_queue"]

    print("=" * 80)
    print("启动 Parameter Server 多进程训练")
    print("=" * 80)
    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model}")
    print(f"worker 数量: {args.num_workers}")
    print(f"训练轮数: {args.epochs}")
    print(f"每个 worker 的 batch size: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"随机种子: {args.seed}")
    print(f"设备: {args.device}")
    print(f"多进程启动方式: {args.start_method}")
    print("=" * 80)

    worker_processes: List[Any] = []
    server_process = None
    all_processes: List[Any] = []

    main_start_time = time.perf_counter()

    try:
        worker_processes = start_worker_processes(
            ctx=ctx,
            num_workers=args.num_workers,
            task_queues=task_queues,
            result_queue=result_queue,
            config=config,
        )

        server_process = start_server_process(
            ctx=ctx,
            num_workers=args.num_workers,
            task_queues=task_queues,
            result_queue=result_queue,
            log_queue=log_queue,
            config=config,
        )

        all_processes = [server_process] + worker_processes

        message = wait_for_server_result(
            log_queue=log_queue,
            timeout=args.timeout,
        )

        if message.get("type") == "error":
            error_text = message.get("traceback", "")
            raise RuntimeError(f"server 训练失败：\n{error_text}")

        if message.get("type") != "done":
            raise RuntimeError(f"主进程收到未知 server 消息: {message}")

        join_processes(all_processes, timeout=30.0)

        main_elapsed_time = time.perf_counter() - main_start_time

        print("=" * 80)
        print("Parameter Server 训练主进程结束")
        print("=" * 80)
        print(f"日志文件: {message.get('log_path')}")
        print(f"配置文件: {message.get('config_path')}")
        print(f"最终测试准确率: {float(message.get('final_test_acc', 0.0)) * 100:.2f}%")
        print(f"server 记录总耗时: {float(message.get('elapsed_time', 0.0)):.2f}s")
        print(f"主进程总耗时: {main_elapsed_time:.2f}s")
        print("=" * 80)

        print_process_status(all_processes)

    except KeyboardInterrupt:
        print("检测到用户中断，正在终止子进程...")
        terminate_processes(all_processes)

    except Exception:
        print("Parameter Server 训练失败，正在终止子进程...")
        terminate_processes(all_processes)
        raise

    finally:
        # 显式关闭队列，降低 Windows 下残留句柄的概率
        try:
            for q in task_queues:
                q.close()
            result_queue.close()
            log_queue.close()
        except Exception:
            pass


def main() -> None:
    """
    主函数。
    """

    args = parse_args()

    # Windows 下只支持 spawn。若用户误传 fork/forkserver，这里直接给出明确错误。
    if args.start_method != "spawn":
        import platform

        if platform.system().lower().startswith("windows"):
            raise RuntimeError("Windows 下 multiprocessing 只支持 --start-method spawn")

    run_parameter_server_training(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()