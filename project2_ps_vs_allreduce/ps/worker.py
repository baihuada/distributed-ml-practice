"""
Parameter Server Worker 进程逻辑。

本文件实现 worker 侧功能：
1. 加载本地数据分片；
2. 接收 server 下发的全局模型参数；
3. 基于本地 mini-batch 计算梯度；
4. 将梯度和本地统计信息上传给 server；
5. 接收停止信号后安全退出。

注意：
worker 不执行 optimizer.step()。
同步 PS-SGD 中，参数更新由 server 统一完成。
"""

from __future__ import annotations

import queue
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common.datasets import build_worker_dataloader, get_dataset, split_dataset_by_workers
from common.metrics import count_correct
from common.models import build_model
from common.seed import set_seed


@dataclass
class WorkerRuntime:
    """
    worker 运行时状态。

    这个对象只在 worker 进程内部使用，用来保存：
    1. worker 编号；
    2. 本地数据集；
    3. 本地 DataLoader；
    4. DataLoader 迭代器；
    5. 本地模型副本；
    6. 损失函数；
    7. 当前 epoch。
    """

    worker_id: int
    num_workers: int
    dataset: Dataset
    dataloader: DataLoader
    data_iter: Iterator
    model: nn.Module
    criterion: nn.Module
    device: torch.device
    current_epoch: int


def resolve_worker_device(device_name: str) -> torch.device:
    """
    解析 worker 使用的设备。

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
    第一版 PS 多进程建议使用 CPU。
    Windows + multiprocessing + CUDA 容易出现额外问题。
    """

    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("worker 指定了 cuda，但当前环境不可用 CUDA")
        return torch.device("cuda")

    if device_name == "auto":
        # PS 第一版默认仍然优先 CPU，避免多进程 CUDA 问题
        return torch.device("cpu")

    raise ValueError(f"不支持的 worker device: {device_name}")


def move_state_dict_to_device(
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    将 server 下发的模型参数移动到 worker 设备上。

    参数
    ----
    state_dict : Dict[str, torch.Tensor]
        server 下发的模型参数字典。
    device : torch.device
        worker 本地设备。

    返回
    ----
    Dict[str, torch.Tensor]
        移动到指定设备后的模型参数字典。
    """

    return {name: tensor.to(device) for name, tensor in state_dict.items()}


def gradients_to_cpu(
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    从模型中提取梯度，并转移到 CPU。

    参数
    ----
    model : nn.Module
        已经完成 backward() 的模型。

    返回
    ----
    Dict[str, torch.Tensor]
        参数名到梯度张量的映射。

    说明
    ----
    通过 Queue 在多进程之间传递张量时，统一使用 CPU 张量更稳定。
    """

    gradients: Dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            gradients[name] = torch.zeros_like(param.detach()).cpu()
        else:
            gradients[name] = param.grad.detach().cpu().clone()

    return gradients


def build_worker_runtime(
    worker_id: int,
    num_workers: int,
    config: Dict[str, Any],
) -> WorkerRuntime:
    """
    构造 worker 运行时对象。

    参数
    ----
    worker_id : int
        当前 worker 编号。
    num_workers : int
        worker 总数。
    config : Dict[str, Any]
        训练配置字典。

    返回
    ----
    WorkerRuntime
        worker 运行时状态。
    """

    seed = int(config.get("seed", 42))
    dataset_name = str(config.get("dataset", "mnist"))
    data_dir = str(config.get("data_dir", "./data"))
    batch_size = int(config.get("batch_size", 64))
    dataloader_num_workers = int(config.get("dataloader_num_workers", 0))
    model_name = str(config.get("model", "mlp"))
    hidden_dim = int(config.get("hidden_dim", 256))
    dropout = float(config.get("dropout", 0.0))
    device_name = str(config.get("device", "cpu"))

    # 每个 worker 使用不同但可复现的随机种子
    set_seed(seed + worker_id)

    device = resolve_worker_device(device_name)

    full_train_dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train=True,
        download=True,
    )

    worker_subsets = split_dataset_by_workers(
        dataset=full_train_dataset,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
    )

    worker_dataset = worker_subsets[worker_id]

    dataloader = build_worker_dataloader(
        dataset=worker_dataset,
        batch_size=batch_size,
        num_workers=dataloader_num_workers,
        seed=seed + worker_id,
        pin_memory=False,
        shuffle=True,
    )

    model = build_model(
        model_name=model_name,
        dataset_name=dataset_name,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    return WorkerRuntime(
        worker_id=worker_id,
        num_workers=num_workers,
        dataset=worker_dataset,
        dataloader=dataloader,
        data_iter=iter(dataloader),
        model=model,
        criterion=criterion,
        device=device,
        current_epoch=0,
    )


def reset_worker_epoch(
    runtime: WorkerRuntime,
    config: Dict[str, Any],
    epoch: int,
) -> None:
    """
    重置 worker 当前 epoch 的 DataLoader。

    参数
    ----
    runtime : WorkerRuntime
        worker 运行时状态。
    config : Dict[str, Any]
        训练配置字典。
    epoch : int
        当前 epoch 编号。

    说明
    ----
    每个 epoch 重新构造 DataLoader，可以让 shuffle 随 epoch 改变。
    """

    batch_size = int(config.get("batch_size", 64))
    dataloader_num_workers = int(config.get("dataloader_num_workers", 0))
    seed = int(config.get("seed", 42))

    runtime.dataloader = build_worker_dataloader(
        dataset=runtime.dataset,
        batch_size=batch_size,
        num_workers=dataloader_num_workers,
        seed=seed + epoch * 1000 + runtime.worker_id,
        pin_memory=False,
        shuffle=True,
    )

    runtime.data_iter = iter(runtime.dataloader)
    runtime.current_epoch = epoch


def get_next_batch(runtime: WorkerRuntime) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 worker 本地 DataLoader 中取下一个 batch。

    参数
    ----
    runtime : WorkerRuntime
        worker 运行时状态。

    返回
    ----
    Tuple[torch.Tensor, torch.Tensor]
        inputs 和 targets。

    说明
    ----
    如果当前 DataLoader 已经迭代完，则重新开始。
    第一版同步 PS 训练中，server 会控制每个 epoch 的 step 数，
    因此通常不会频繁触发 StopIteration。
    """

    try:
        inputs, targets = next(runtime.data_iter)
    except StopIteration:
        runtime.data_iter = iter(runtime.dataloader)
        inputs, targets = next(runtime.data_iter)

    return inputs, targets


def compute_local_gradient(
    runtime: WorkerRuntime,
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """
    基于 server 下发的全局参数计算一个本地 mini-batch 梯度。

    参数
    ----
    runtime : WorkerRuntime
        worker 运行时状态。
    state_dict : Dict[str, torch.Tensor]
        server 下发的全局模型参数。

    返回
    ----
    Dict[str, Any]
        worker 计算结果，包括梯度、loss、正确数和样本数。
    """

    model = runtime.model
    device = runtime.device

    # 加载 server 下发的最新全局参数
    state_dict_on_device = move_state_dict_to_device(state_dict, device)
    model.load_state_dict(state_dict_on_device)

    model.train()
    model.zero_grad(set_to_none=True)

    inputs, targets = get_next_batch(runtime)
    inputs = inputs.to(device)
    targets = targets.to(device)

    logits = model(inputs)
    loss = runtime.criterion(logits, targets)

    loss.backward()

    batch_size = targets.size(0)
    correct = count_correct(logits, targets)

    gradients = gradients_to_cpu(model)

    return {
        "gradients": gradients,
        "loss_sum": float(loss.item() * batch_size),
        "correct": int(correct),
        "num_samples": int(batch_size),
    }


def ps_worker_loop(
    worker_id: int,
    num_workers: int,
    task_queue: Any,
    result_queue: Any,
    config: Dict[str, Any],
) -> None:
    """
    Parameter Server worker 主循环。

    参数
    ----
    worker_id : int
        当前 worker 编号。
    num_workers : int
        worker 总数。
    task_queue : multiprocessing.Queue
        server 发给当前 worker 的任务队列。
    result_queue : multiprocessing.Queue
        worker 发回 server 的结果队列。
    config : Dict[str, Any]
        训练配置字典。

    消息协议
    --------
    server -> worker:
        {"type": "compute_grad", "epoch": int, "step": int, "state_dict": dict}
        {"type": "stop"}

    worker -> server:
        {
            "type": "grad",
            "worker_id": int,
            "epoch": int,
            "step": int,
            "gradients": dict,
            "loss_sum": float,
            "correct": int,
            "num_samples": int
        }

    如果 worker 内部报错，会发送：
        {"type": "error", "worker_id": int, "traceback": str}
    """

    runtime = None

    try:
        runtime = build_worker_runtime(
            worker_id=worker_id,
            num_workers=num_workers,
            config=config,
        )

        print(
            f"[Worker {worker_id}] 启动完成，"
            f"本地样本数={len(runtime.dataset)}，"
            f"设备={runtime.device}"
        )

        while True:
            try:
                message = task_queue.get(timeout=300)
            except queue.Empty:
                raise RuntimeError(f"[Worker {worker_id}] 等待 server 任务超时")

            message_type = message.get("type", "")

            if message_type == "stop":
                print(f"[Worker {worker_id}] 收到 stop 信号，准备退出")
                break

            if message_type != "compute_grad":
                raise ValueError(
                    f"[Worker {worker_id}] 收到未知消息类型: {message_type}"
                )

            epoch = int(message["epoch"])
            step = int(message["step"])
            state_dict = message["state_dict"]

            # 如果进入新的 epoch，则重置本地 DataLoader
            if runtime.current_epoch != epoch:
                reset_worker_epoch(runtime, config=config, epoch=epoch)

            local_result = compute_local_gradient(
                runtime=runtime,
                state_dict=state_dict,
            )

            result_queue.put(
                {
                    "type": "grad",
                    "worker_id": worker_id,
                    "epoch": epoch,
                    "step": step,
                    "gradients": local_result["gradients"],
                    "loss_sum": local_result["loss_sum"],
                    "correct": local_result["correct"],
                    "num_samples": local_result["num_samples"],
                }
            )

    except Exception:
        error_text = traceback.format_exc()

        try:
            result_queue.put(
                {
                    "type": "error",
                    "worker_id": worker_id,
                    "traceback": error_text,
                }
            )
        except Exception:
            pass

        print(f"[Worker {worker_id}] 发生错误：")
        print(error_text)

    finally:
        # 显式释放引用，便于多进程退出时回收资源
        runtime = None