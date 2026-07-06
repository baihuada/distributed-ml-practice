"""
通信量估算模块。

本文件负责：
1. 统计模型参数总量；
2. 估算模型大小；
3. 估算 Parameter Server 通信量；
4. 估算 Ring AllReduce 通信量；
5. 为 Single、PS、DDP 的日志提供统一通信量字段。

说明：
这里的通信量是理论估算值，不是真实网络抓包结果。
本项目第一版是单机多进程实验，因此通信量主要用于系统结构对比。
"""

from __future__ import annotations

from typing import Dict, Literal

import torch.nn as nn


CommUnit = Literal["B", "KB", "MB", "GB"]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    统计模型参数量。

    参数
    ----
    model : nn.Module
        PyTorch 模型。
    trainable_only : bool
        如果为 True，只统计 requires_grad=True 的参数。
        如果为 False，统计全部参数。

    返回
    ----
    int
        参数总量。
    """

    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())


def bytes_to_unit(num_bytes: float, unit: CommUnit = "MB") -> float:
    """
    将字节数转换为指定单位。

    参数
    ----
    num_bytes : float
        字节数。
    unit : str
        目标单位，支持 B、KB、MB、GB。

    返回
    ----
    float
        转换后的数值。
    """

    unit = unit.upper()

    if unit == "B":
        return float(num_bytes)

    if unit == "KB":
        return float(num_bytes) / 1024

    if unit == "MB":
        return float(num_bytes) / (1024**2)

    if unit == "GB":
        return float(num_bytes) / (1024**3)

    raise ValueError(f"不支持的单位: {unit}. 只支持 B、KB、MB、GB。")


def estimate_model_size_bytes(
    model: nn.Module,
    bytes_per_param: int = 4,
    trainable_only: bool = False,
) -> int:
    """
    估算模型参数大小，单位为 bytes。

    参数
    ----
    model : nn.Module
        PyTorch 模型。
    bytes_per_param : int
        每个参数占用的字节数。
        float32 通常为 4 字节。
    trainable_only : bool
        是否只统计可训练参数。

    返回
    ----
    int
        模型大小，单位 bytes。
    """

    if bytes_per_param <= 0:
        raise ValueError(f"bytes_per_param 必须为正数，但得到 {bytes_per_param}")

    num_params = count_parameters(model, trainable_only=trainable_only)
    return int(num_params * bytes_per_param)


def estimate_model_size(
    model: nn.Module,
    unit: CommUnit = "MB",
    bytes_per_param: int = 4,
    trainable_only: bool = False,
) -> float:
    """
    估算模型参数大小，并转换为指定单位。

    参数
    ----
    model : nn.Module
        PyTorch 模型。
    unit : str
        输出单位，支持 B、KB、MB、GB。
    bytes_per_param : int
        每个参数占用的字节数。
    trainable_only : bool
        是否只统计可训练参数。

    返回
    ----
    float
        模型大小。
    """

    size_bytes = estimate_model_size_bytes(
        model=model,
        bytes_per_param=bytes_per_param,
        trainable_only=trainable_only,
    )

    return bytes_to_unit(size_bytes, unit=unit)


def estimate_ps_comm_per_round_bytes(
    model: nn.Module,
    num_workers: int,
    bytes_per_param: int = 4,
) -> float:
    """
    估算同步 Parameter Server 每轮总通信量，单位为 bytes。

    估算逻辑：
    每个 worker 每轮通常需要：
    1. 从 server 拉取一份全局模型参数；
    2. 向 server 上传一份梯度。

    因此，每个 worker 约通信 2 * S_model。
    K 个 worker 总通信量约为 2 * K * S_model。

    参数
    ----
    model : nn.Module
        PyTorch 模型。
    num_workers : int
        worker 数量。
    bytes_per_param : int
        每个参数占用字节数。

    返回
    ----
    float
        每轮总通信量，单位 bytes。
    """

    if num_workers <= 0:
        raise ValueError(f"num_workers 必须为正整数，但得到 {num_workers}")

    model_size_bytes = estimate_model_size_bytes(
        model=model,
        bytes_per_param=bytes_per_param,
    )

    comm_bytes = 2.0 * num_workers * model_size_bytes
    return comm_bytes


def estimate_ps_comm_per_round(
    model: nn.Module,
    num_workers: int,
    unit: CommUnit = "MB",
    bytes_per_param: int = 4,
) -> float:
    """
    估算同步 Parameter Server 每轮总通信量，并转换为指定单位。

    参数
    ----
    model : nn.Module
        PyTorch 模型。
    num_workers : int
        worker 数量。
    unit : str
        输出单位，支持 B、KB、MB、GB。
    bytes_per_param : int
        每个参数占用字节数。

    返回
    ----
    float
        每轮总通信量。
    """

    comm_bytes = estimate_ps_comm_per_round_bytes(
        model=model,
        num_workers=num_workers,
        bytes_per_param=bytes_per_param,
    )

    return bytes_to_unit(comm_bytes, unit=unit)


def estimate_ring_allreduce_comm_per_process_bytes(
    model: nn.Module,
    num_workers: int,
    bytes_per_param: int = 4,
) -> float:
    """
    估算 Ring AllReduce 中每个进程每轮通信量，单位为 bytes。

    经典 Ring AllReduce 中，每个进程的通信量近似为：

        2 * (K - 1) / K * S_model

    其中：
    K 表示进程数量；
    S_model 表示模型梯度大小，近似等于模型参数大小。

    注意：
    这里估算的是每个进程承担的通信量，不是所有进程加总后的通信量。

    参数
    ----
    model : nn.Module
        PyTorch 模型。
    num_workers : int
        worker / rank 数量。
    bytes_per_param : int
        每个参数占用字节数。

    返回
    ----
    float
        每个进程每轮通信量，单位 bytes。
    """

    if num_workers <= 0:
        raise ValueError(f"num_workers 必须为正整数，但得到 {num_workers}")

    if num_workers == 1:
        return 0.0

    model_size_bytes = estimate_model_size_bytes(
        model=model,
        bytes_per_param=bytes_per_param,
    )

    comm_bytes = 2.0 * (num_workers - 1) / num_workers * model_size_bytes
    return comm_bytes


def estimate_ring_allreduce_comm_per_process(
    model: nn.Module,
    num_workers: int,
    unit: CommUnit = "MB",
    bytes_per_param: int = 4,
) -> float:
    """
    估算 Ring AllReduce 中每个进程每轮通信量，并转换为指定单位。

    参数
    ----
    model : nn.Module
        PyTorch 模型。
    num_workers : int
        worker / rank 数量。
    unit : str
        输出单位，支持 B、KB、MB、GB。
    bytes_per_param : int
        每个参数占用字节数。

    返回
    ----
    float
        每个进程每轮通信量。
    """

    comm_bytes = estimate_ring_allreduce_comm_per_process_bytes(
        model=model,
        num_workers=num_workers,
        bytes_per_param=bytes_per_param,
    )

    return bytes_to_unit(comm_bytes, unit=unit)


def estimate_ring_allreduce_total_comm_bytes(
    model: nn.Module,
    num_workers: int,
    bytes_per_param: int = 4,
) -> float:
    """
    估算 Ring AllReduce 所有进程每轮总通信量，单位为 bytes。

    总通信量 = 每个进程通信量 * 进程数

    注意：
    论文和系统分析中经常更关注“每个进程通信量”，
    因为它更能体现单个 rank 的通信压力。
    这里提供总通信量只是为了日志和图表分析更灵活。

    参数
    ----
    model : nn.Module
        PyTorch 模型。
    num_workers : int
        worker / rank 数量。
    bytes_per_param : int
        每个参数占用字节数。

    返回
    ----
    float
        所有进程每轮总通信量，单位 bytes。
    """

    per_process_bytes = estimate_ring_allreduce_comm_per_process_bytes(
        model=model,
        num_workers=num_workers,
        bytes_per_param=bytes_per_param,
    )

    return per_process_bytes * num_workers


def estimate_ring_allreduce_total_comm(
    model: nn.Module,
    num_workers: int,
    unit: CommUnit = "MB",
    bytes_per_param: int = 4,
) -> float:
    """
    估算 Ring AllReduce 所有进程每轮总通信量，并转换为指定单位。

    参数
    ----
    model : nn.Module
        PyTorch 模型。
    num_workers : int
        worker / rank 数量。
    unit : str
        输出单位，支持 B、KB、MB、GB。
    bytes_per_param : int
        每个参数占用字节数。

    返回
    ----
    float
        所有进程每轮总通信量。
    """

    total_bytes = estimate_ring_allreduce_total_comm_bytes(
        model=model,
        num_workers=num_workers,
        bytes_per_param=bytes_per_param,
    )

    return bytes_to_unit(total_bytes, unit=unit)


def estimate_comm_per_epoch(
    model: nn.Module,
    system: str,
    num_workers: int = 1,
    num_steps: int = 1,
    unit: CommUnit = "MB",
    bytes_per_param: int = 4,
) -> float:
    """
    估算一个 epoch 的通信量。

    参数
    ----
    model : nn.Module
        PyTorch 模型。
    system : str
        系统类型，支持 single、ps、ddp、allreduce。
    num_workers : int
        worker / rank 数量。
    num_steps : int
        每个 epoch 中发生同步通信的 step 数。
        对于 mini-batch SGD，一般每个 batch 后同步一次梯度。
    unit : str
        输出单位，支持 B、KB、MB、GB。
    bytes_per_param : int
        每个参数占用字节数。

    返回
    ----
    float
        一个 epoch 的通信量。
    """

    if num_steps < 0:
        raise ValueError(f"num_steps 不能为负数，但得到 {num_steps}")

    system_name = system.lower()

    if system_name in ["single", "single_process", "baseline"]:
        return 0.0

    if system_name in ["ps", "parameter_server", "parameter-server"]:
        per_round = estimate_ps_comm_per_round(
            model=model,
            num_workers=num_workers,
            unit=unit,
            bytes_per_param=bytes_per_param,
        )
        return per_round * num_steps

    if system_name in ["ddp", "allreduce", "all_reduce", "ring"]:
        per_round = estimate_ring_allreduce_comm_per_process(
            model=model,
            num_workers=num_workers,
            unit=unit,
            bytes_per_param=bytes_per_param,
        )
        return per_round * num_steps

    raise ValueError(
        f"不支持的系统类型: {system}. 当前支持 single、ps、ddp、allreduce。"
    )


def get_model_comm_summary(
    model: nn.Module,
    num_workers: int = 1,
    bytes_per_param: int = 4,
) -> Dict[str, float]:
    """
    获取模型大小与通信量摘要。

    参数
    ----
    model : nn.Module
        PyTorch 模型。
    num_workers : int
        worker / rank 数量。
    bytes_per_param : int
        每个参数占用字节数。

    返回
    ----
    Dict[str, float]
        包含模型参数量、模型大小、PS 通信量、AllReduce 通信量的字典。
    """

    num_params = count_parameters(model)
    model_size_mb = estimate_model_size(
        model=model,
        unit="MB",
        bytes_per_param=bytes_per_param,
    )

    ps_comm_mb = estimate_ps_comm_per_round(
        model=model,
        num_workers=num_workers,
        unit="MB",
        bytes_per_param=bytes_per_param,
    )

    ring_per_process_mb = estimate_ring_allreduce_comm_per_process(
        model=model,
        num_workers=num_workers,
        unit="MB",
        bytes_per_param=bytes_per_param,
    )

    ring_total_mb = estimate_ring_allreduce_total_comm(
        model=model,
        num_workers=num_workers,
        unit="MB",
        bytes_per_param=bytes_per_param,
    )

    return {
        "num_params": float(num_params),
        "model_size_mb": float(model_size_mb),
        "ps_comm_per_round_mb": float(ps_comm_mb),
        "ring_comm_per_process_per_round_mb": float(ring_per_process_mb),
        "ring_total_comm_per_round_mb": float(ring_total_mb),
    }


def print_comm_summary(
    model: nn.Module,
    num_workers: int = 1,
    bytes_per_param: int = 4,
) -> None:
    """
    打印模型与通信量摘要。

    参数
    ----
    model : nn.Module
        PyTorch 模型。
    num_workers : int
        worker / rank 数量。
    bytes_per_param : int
        每个参数占用字节数。
    """

    summary = get_model_comm_summary(
        model=model,
        num_workers=num_workers,
        bytes_per_param=bytes_per_param,
    )

    print("模型与通信量估算：")
    print(f"参数总量: {int(summary['num_params'])}")
    print(f"模型大小: {summary['model_size_mb']:.4f} MB")
    print(f"PS 每轮总通信量: {summary['ps_comm_per_round_mb']:.4f} MB")
    print(
        "Ring AllReduce 每进程每轮通信量: "
        f"{summary['ring_comm_per_process_per_round_mb']:.4f} MB"
    )
    print(
        "Ring AllReduce 所有进程每轮总通信量: "
        f"{summary['ring_total_comm_per_round_mb']:.4f} MB"
    )