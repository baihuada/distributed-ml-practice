"""
Parameter Server Server 进程逻辑。

本文件实现 server 侧功能：
1. 保存全局模型；
2. 向 worker 广播模型参数；
3. 接收 worker 上传的梯度；
4. 按样本数加权聚合梯度；
5. 使用 SGD 更新全局模型；
6. 评估模型并保存 CSV 日志。

注意：
第一版实现同步 PS-SGD。
server 必须等待所有 worker 的梯度都到达后，才能执行一次全局更新。
"""

from __future__ import annotations

import math
import queue
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from common.datasets import get_dataloaders, get_dataset
from common.logger import CSVLogger, build_log_filename, print_epoch_log, save_config
from common.metrics import compute_accuracy, compute_samples_per_sec, evaluate
from common.models import build_model
from common.seed import set_seed
from utils.comm import count_parameters, estimate_comm_per_epoch, estimate_model_size


@dataclass
class ServerRuntime:
    """
    server 运行时状态。

    该对象只在 server 进程内部使用。

    字段说明：
    1. model：server 持有的全局模型；
    2. optimizer：server 端优化器；
    3. criterion：损失函数；
    4. test_loader：测试集 DataLoader；
    5. device：server 使用的设备；
    6. logger：CSV 日志记录器；
    7. steps_per_epoch：每个 epoch 的同步更新步数；
    8. model_params：模型参数量；
    9. model_size_mb：模型大小估算。
    """

    model: nn.Module
    optimizer: optim.Optimizer
    criterion: nn.Module
    test_loader: torch.utils.data.DataLoader
    device: torch.device
    logger: CSVLogger
    steps_per_epoch: int
    model_params: int
    model_size_mb: float
    log_path: Path
    config_path: Path


def resolve_server_device(device_name: str) -> torch.device:
    """
    解析 server 使用的设备。

    参数
    ----
    device_name : str
        设备名称，支持 "cpu"、"cuda"、"auto"。

    返回
    ----
    torch.device
        实际使用的设备。

    说明
    ----
    第一版 Parameter Server 建议使用 CPU。
    因为 Windows + multiprocessing + CUDA 组合更容易出现进程启动和显存复制问题。
    """

    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("server 指定了 cuda，但当前环境不可用 CUDA")
        return torch.device("cuda")

    if device_name == "auto":
        # PS 第一版默认使用 CPU，保证多进程版本稳定跑通
        return torch.device("cpu")

    raise ValueError(f"不支持的 server device: {device_name}")


def build_optimizer(
    model: nn.Module,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> optim.Optimizer:
    """
    构造 server 端优化器。

    参数
    ----
    model : nn.Module
        server 持有的全局模型。
    lr : float
        学习率。
    momentum : float
        SGD 动量系数。
    weight_decay : float
        权重衰减系数。

    返回
    ----
    optim.Optimizer
        PyTorch 优化器。
    """

    return optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def clone_state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    克隆全局模型参数，并移动到 CPU。

    参数
    ----
    model : nn.Module
        server 端全局模型。

    返回
    ----
    Dict[str, torch.Tensor]
        可发送给 worker 的模型参数字典。

    说明
    ----
    多进程 Queue 传递 CPU Tensor 更稳定。
    因此即使 server 模型在 GPU 上，也先转为 CPU 后再发送。
    """

    state_dict = model.state_dict()

    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state_dict.items()
    }


def compute_worker_subset_sizes(
    train_size: int,
    num_workers: int,
) -> List[int]:
    """
    计算每个 worker 的数据分片大小。

    参数
    ----
    train_size : int
        训练集总样本数。
    num_workers : int
        worker 数量。

    返回
    ----
    List[int]
        每个 worker 的样本数。
    """

    if num_workers <= 0:
        raise ValueError(f"num_workers 必须为正整数，但得到 {num_workers}")

    base_size = train_size // num_workers
    remainder = train_size % num_workers

    sizes = []

    for worker_id in range(num_workers):
        current_size = base_size + (1 if worker_id < remainder else 0)
        sizes.append(current_size)

    return sizes


def compute_steps_per_epoch(
    train_size: int,
    num_workers: int,
    batch_size: int,
) -> int:
    """
    计算同步 PS 每个 epoch 的 step 数。

    参数
    ----
    train_size : int
        训练集总样本数。
    num_workers : int
        worker 数量。
    batch_size : int
        每个 worker 的本地 batch size。

    返回
    ----
    int
        每个 epoch 的同步更新步数。

    说明
    ----
    每个 step 中，server 会让所有 worker 各计算一个本地 batch 的梯度。
    如果 worker 数据分片大小为 N_k，则该 worker 每个 epoch 约有 ceil(N_k / batch_size) 个 batch。
    同步版本中使用最大 worker batch 数作为 steps_per_epoch。
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，但得到 {batch_size}")

    subset_sizes = compute_worker_subset_sizes(
        train_size=train_size,
        num_workers=num_workers,
    )

    max_subset_size = max(subset_sizes)
    steps = math.ceil(max_subset_size / batch_size)

    return max(1, steps)


def build_server_runtime(
    num_workers: int,
    config: Dict[str, Any],
) -> ServerRuntime:
    """
    构造 server 运行时对象。

    参数
    ----
    num_workers : int
        worker 数量。
    config : Dict[str, Any]
        训练配置字典。

    返回
    ----
    ServerRuntime
        server 运行时对象。
    """

    seed = int(config.get("seed", 42))
    dataset_name = str(config.get("dataset", "mnist"))
    data_dir = str(config.get("data_dir", "./data"))
    model_name = str(config.get("model", "mlp"))
    batch_size = int(config.get("batch_size", 64))
    test_batch_size = int(config.get("test_batch_size", 1000))
    dataloader_num_workers = int(config.get("dataloader_num_workers", 0))
    hidden_dim = int(config.get("hidden_dim", 256))
    dropout = float(config.get("dropout", 0.0))
    lr = float(config.get("lr", 0.01))
    momentum = float(config.get("momentum", 0.0))
    weight_decay = float(config.get("weight_decay", 0.0))
    device_name = str(config.get("device", "cpu"))
    output_dir = str(config.get("output_dir", "./results/raw"))

    set_seed(seed)

    device = resolve_server_device(device_name)

    # server 需要测试集，用于每个 epoch 后评估全局模型
    _, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        num_workers=dataloader_num_workers,
        seed=seed,
        pin_memory=False,
        download=True,
    )

    # 只用于计算训练集大小和 steps_per_epoch
    train_dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train=True,
        download=True,
    )

    train_size = len(train_dataset)

    steps_per_epoch = compute_steps_per_epoch(
        train_size=train_size,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    model = build_model(
        model_name=model_name,
        dataset_name=dataset_name,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(
        model=model,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    model_params = count_parameters(model)
    model_size_mb = estimate_model_size(model, unit="MB")

    log_filename = build_log_filename(
        system="ps",
        dataset=dataset_name,
        model=model_name,
        seed=seed,
        num_workers=num_workers,
    )

    log_path = Path(output_dir) / log_filename
    config_path = Path(output_dir) / log_filename.replace(".csv", "_config.json")

    logger = CSVLogger(log_path=log_path)

    save_config(config, config_path)

    return ServerRuntime(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        test_loader=test_loader,
        device=device,
        logger=logger,
        steps_per_epoch=steps_per_epoch,
        model_params=model_params,
        model_size_mb=model_size_mb,
        log_path=log_path,
        config_path=config_path,
    )


def broadcast_compute_tasks(
    task_queues: List[Any],
    state_dict: Dict[str, torch.Tensor],
    epoch: int,
    step: int,
) -> None:
    """
    向所有 worker 广播计算梯度任务。

    参数
    ----
    task_queues : List[Any]
        每个 worker 对应一个任务队列。
    state_dict : Dict[str, torch.Tensor]
        当前全局模型参数。
    epoch : int
        当前 epoch。
    step : int
        当前 step。
    """

    for task_queue in task_queues:
        task_queue.put(
            {
                "type": "compute_grad",
                "epoch": epoch,
                "step": step,
                "state_dict": state_dict,
            }
        )


def stop_workers(task_queues: List[Any]) -> None:
    """
    向所有 worker 发送停止信号。

    参数
    ----
    task_queues : List[Any]
        每个 worker 对应一个任务队列。
    """

    for task_queue in task_queues:
        task_queue.put({"type": "stop"})


def collect_worker_results(
    result_queue: Any,
    num_workers: int,
    epoch: int,
    step: int,
    timeout: float = 300.0,
) -> List[Dict[str, Any]]:
    """
    收集所有 worker 的梯度结果。

    参数
    ----
    result_queue : multiprocessing.Queue
        worker 上传结果的队列。
    num_workers : int
        worker 数量。
    epoch : int
        当前 epoch。
    step : int
        当前 step。
    timeout : float
        等待单个结果的超时时间。

    返回
    ----
    List[Dict[str, Any]]
        所有 worker 的结果列表。

    说明
    ----
    同步 PS-SGD 必须等待所有 worker 的梯度。
    如果任何 worker 报错，server 会立即抛出异常。
    """

    results: List[Dict[str, Any]] = []
    received_workers = set()

    while len(results) < num_workers:
        try:
            message = result_queue.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(
                f"server 等待 worker 梯度超时：epoch={epoch}, step={step}"
            )

        message_type = message.get("type", "")

        if message_type == "error":
            worker_id = message.get("worker_id", "unknown")
            error_text = message.get("traceback", "")
            raise RuntimeError(
                f"Worker {worker_id} 报错，server 停止训练。\n{error_text}"
            )

        if message_type != "grad":
            raise RuntimeError(f"server 收到未知消息类型: {message_type}")

        worker_id = int(message["worker_id"])

        if worker_id in received_workers:
            raise RuntimeError(
                f"server 重复收到 worker {worker_id} 的结果：epoch={epoch}, step={step}"
            )

        msg_epoch = int(message["epoch"])
        msg_step = int(message["step"])

        if msg_epoch != epoch or msg_step != step:
            raise RuntimeError(
                "server 收到错位梯度："
                f"期望 epoch={epoch}, step={step}，"
                f"实际 epoch={msg_epoch}, step={msg_step}, worker={worker_id}"
            )

        received_workers.add(worker_id)
        results.append(message)

    return results


def aggregate_gradients(
    worker_results: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """
    按样本数加权聚合 worker 梯度。

    参数
    ----
    worker_results : List[Dict[str, Any]]
        所有 worker 上传的梯度结果。

    返回
    ----
    Dict[str, torch.Tensor]
        聚合后的全局梯度。

    公式
    ----
    g_t = sum_k (n_k / n) * g_k
    """

    if not worker_results:
        raise ValueError("worker_results 不能为空")

    total_samples = sum(int(result["num_samples"]) for result in worker_results)

    if total_samples <= 0:
        raise ValueError("worker 上传的总样本数必须为正数")

    aggregated: Dict[str, torch.Tensor] = {}

    for result in worker_results:
        weight = int(result["num_samples"]) / total_samples
        gradients = result["gradients"]

        for name, grad in gradients.items():
            grad_cpu = grad.detach().cpu()

            if name not in aggregated:
                aggregated[name] = grad_cpu.clone() * weight
            else:
                aggregated[name] += grad_cpu * weight

    return aggregated


def apply_gradients(
    model: nn.Module,
    optimizer: optim.Optimizer,
    aggregated_gradients: Dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    """
    将聚合梯度写入全局模型，并执行 optimizer.step()。

    参数
    ----
    model : nn.Module
        server 端全局模型。
    optimizer : optim.Optimizer
        server 端优化器。
    aggregated_gradients : Dict[str, torch.Tensor]
        聚合后的梯度。
    device : torch.device
        server 模型所在设备。
    """

    optimizer.zero_grad(set_to_none=True)

    for name, param in model.named_parameters():
        if name not in aggregated_gradients:
            raise KeyError(f"聚合梯度中缺少参数: {name}")

        param.grad = aggregated_gradients[name].to(device).clone()

    optimizer.step()


def summarize_worker_results(
    worker_results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    汇总当前 step 的训练统计信息。

    参数
    ----
    worker_results : List[Dict[str, Any]]
        所有 worker 的返回结果。

    返回
    ----
    Dict[str, float]
        当前 step 的 loss_sum、correct、num_samples。
    """

    loss_sum = sum(float(result["loss_sum"]) for result in worker_results)
    correct = sum(int(result["correct"]) for result in worker_results)
    num_samples = sum(int(result["num_samples"]) for result in worker_results)

    return {
        "loss_sum": float(loss_sum),
        "correct": float(correct),
        "num_samples": float(num_samples),
    }


def train_one_ps_step(
    runtime: ServerRuntime,
    task_queues: List[Any],
    result_queue: Any,
    num_workers: int,
    epoch: int,
    step: int,
) -> Dict[str, float]:
    """
    执行一次同步 PS-SGD 更新。

    参数
    ----
    runtime : ServerRuntime
        server 运行时状态。
    task_queues : List[Any]
        worker 任务队列列表。
    result_queue : multiprocessing.Queue
        worker 结果队列。
    num_workers : int
        worker 数量。
    epoch : int
        当前 epoch。
    step : int
        当前 step。

    返回
    ----
    Dict[str, float]
        当前 step 的训练统计信息。
    """

    # 1. 克隆当前全局模型参数，并下发给所有 worker
    state_dict = clone_state_dict_to_cpu(runtime.model)

    broadcast_compute_tasks(
        task_queues=task_queues,
        state_dict=state_dict,
        epoch=epoch,
        step=step,
    )

    # 2. 等待所有 worker 返回梯度
    worker_results = collect_worker_results(
        result_queue=result_queue,
        num_workers=num_workers,
        epoch=epoch,
        step=step,
    )

    # 3. 按样本数加权聚合梯度
    aggregated_gradients = aggregate_gradients(worker_results)

    # 4. 在 server 端更新全局模型
    apply_gradients(
        model=runtime.model,
        optimizer=runtime.optimizer,
        aggregated_gradients=aggregated_gradients,
        device=runtime.device,
    )

    # 5. 汇总当前 step 的训练指标
    step_summary = summarize_worker_results(worker_results)

    return step_summary


def train_one_ps_epoch(
    runtime: ServerRuntime,
    task_queues: List[Any],
    result_queue: Any,
    num_workers: int,
    epoch: int,
) -> Dict[str, float]:
    """
    训练一个 PS epoch。

    参数
    ----
    runtime : ServerRuntime
        server 运行时状态。
    task_queues : List[Any]
        worker 任务队列列表。
    result_queue : multiprocessing.Queue
        worker 结果队列。
    num_workers : int
        worker 数量。
    epoch : int
        当前 epoch。

    返回
    ----
    Dict[str, float]
        当前 epoch 的训练 loss、accuracy、样本数。
    """

    total_loss_sum = 0.0
    total_correct = 0
    total_samples = 0

    for step in range(1, runtime.steps_per_epoch + 1):
        step_summary = train_one_ps_step(
            runtime=runtime,
            task_queues=task_queues,
            result_queue=result_queue,
            num_workers=num_workers,
            epoch=epoch,
            step=step,
        )

        total_loss_sum += step_summary["loss_sum"]
        total_correct += int(step_summary["correct"])
        total_samples += int(step_summary["num_samples"])

    train_loss = total_loss_sum / total_samples
    train_acc = compute_accuracy(
        total_correct=total_correct,
        num_samples=total_samples,
    )

    return {
        "loss": float(train_loss),
        "accuracy": float(train_acc),
        "num_samples": float(total_samples),
    }


def ps_server_loop(
    num_workers: int,
    task_queues: List[Any],
    result_queue: Any,
    log_queue: Optional[Any],
    config: Dict[str, Any],
) -> None:
    """
    Parameter Server 主循环。

    参数
    ----
    num_workers : int
        worker 数量。
    task_queues : List[multiprocessing.Queue]
        server 发给每个 worker 的任务队列。
    result_queue : multiprocessing.Queue
        所有 worker 发回 server 的结果队列。
    log_queue : Optional[multiprocessing.Queue]
        server 发给主进程的状态队列。
        如果不需要向主进程返回状态，可以传 None。
    config : Dict[str, Any]
        训练配置字典。

    说明
    ----
    这个函数通常会在单独的 server 进程中运行。
    它负责完整训练过程和日志保存。
    """

    runtime: Optional[ServerRuntime] = None

    try:
        epochs = int(config.get("epochs", 10))
        dataset_name = str(config.get("dataset", "mnist"))
        model_name = str(config.get("model", "mlp"))
        batch_size = int(config.get("batch_size", 64))
        lr = float(config.get("lr", 0.01))
        seed = int(config.get("seed", 42))

        if epochs <= 0:
            raise ValueError(f"epochs 必须为正整数，但得到 {epochs}")

        runtime = build_server_runtime(
            num_workers=num_workers,
            config=config,
        )

        print("=" * 80)
        print("Parameter Server Training")
        print("=" * 80)
        print(f"数据集: {dataset_name}")
        print(f"模型: {model_name}")
        print(f"worker 数量: {num_workers}")
        print(f"训练轮数: {epochs}")
        print(f"batch size: {batch_size}")
        print(f"学习率: {lr}")
        print(f"随机种子: {seed}")
        print(f"server 设备: {runtime.device}")
        print(f"每个 epoch 的同步 step 数: {runtime.steps_per_epoch}")
        print(f"模型参数量: {runtime.model_params}")
        print(f"模型大小估算: {runtime.model_size_mb:.4f} MB")
        print("=" * 80)

        total_start_time = time.perf_counter()

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.perf_counter()

            train_metrics = train_one_ps_epoch(
                runtime=runtime,
                task_queues=task_queues,
                result_queue=result_queue,
                num_workers=num_workers,
                epoch=epoch,
            )

            test_metrics = evaluate(
                model=runtime.model,
                data_loader=runtime.test_loader,
                criterion=runtime.criterion,
                device=runtime.device,
            )

            epoch_time = time.perf_counter() - epoch_start_time
            elapsed_time = time.perf_counter() - total_start_time

            num_train_samples = int(train_metrics["num_samples"])
            num_test_samples = int(test_metrics["num_samples"])

            samples_per_sec = compute_samples_per_sec(
                num_samples=num_train_samples,
                elapsed_time=epoch_time,
            )

            comm_mb = estimate_comm_per_epoch(
                model=runtime.model,
                system="ps",
                num_workers=num_workers,
                num_steps=runtime.steps_per_epoch,
                unit="MB",
            )

            record = {
                "system": "ps",
                "dataset": dataset_name,
                "model": model_name,
                "num_workers": num_workers,
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_acc": test_metrics["accuracy"],
                "epoch_time": epoch_time,
                "elapsed_time": elapsed_time,
                "samples_per_sec": samples_per_sec,
                "num_train_samples": num_train_samples,
                "num_test_samples": num_test_samples,
                "model_params": runtime.model_params,
                "model_size_mb": runtime.model_size_mb,
                "comm_mb": comm_mb,
                "seed": seed,
                "lr": lr,
                "batch_size": batch_size,
            }

            runtime.logger.log(record)
            print_epoch_log(record)

        runtime.logger.save()

        final_record = runtime.logger.latest()

        print("=" * 80)
        print("PS 训练完成")
        print(f"CSV 日志保存位置: {runtime.log_path}")
        print(f"配置文件保存位置: {runtime.config_path}")
        print(f"最终测试准确率: {float(final_record['test_acc']) * 100:.2f}%")
        print(f"总耗时: {float(final_record['elapsed_time']):.2f}s")
        print("=" * 80)

        stop_workers(task_queues)

        if log_queue is not None:
            log_queue.put(
                {
                    "type": "done",
                    "system": "ps",
                    "log_path": str(runtime.log_path),
                    "config_path": str(runtime.config_path),
                    "final_test_acc": float(final_record["test_acc"]),
                    "elapsed_time": float(final_record["elapsed_time"]),
                }
            )

    except Exception:
        error_text = traceback.format_exc()

        print("[Server] 发生错误：")
        print(error_text)

        try:
            stop_workers(task_queues)
        except Exception:
            pass

        if log_queue is not None:
            try:
                log_queue.put(
                    {
                        "type": "error",
                        "traceback": error_text,
                    }
                )
            except Exception:
                pass

        raise

    finally:
        runtime = None