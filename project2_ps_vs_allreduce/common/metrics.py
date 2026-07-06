"""
评估指标模块。

本文件负责：
1. 计算分类准确率；
2. 计算平均损失；
3. 在验证集或测试集上评估模型；
4. 计算系统吞吐量 samples/s。

Single、PS、DDP 都应该复用这里的函数，避免不同训练方式使用不同评估逻辑。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    根据模型输出 logits 和真实标签计算分类准确率。

    参数
    ----
    logits : torch.Tensor
        模型输出，形状为 [batch_size, num_classes]。
    targets : torch.Tensor
        真实标签，形状为 [batch_size]。

    返回
    ----
    float
        当前 batch 的准确率，范围为 [0, 1]。
    """

    if logits.ndim != 2:
        raise ValueError(f"logits 应该是二维张量，但实际维度为 {logits.ndim}")

    if targets.ndim != 1:
        raise ValueError(f"targets 应该是一维张量，但实际维度为 {targets.ndim}")

    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)

    return correct / total


def count_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    """
    统计预测正确的样本数量。

    参数
    ----
    logits : torch.Tensor
        模型输出，形状为 [batch_size, num_classes]。
    targets : torch.Tensor
        真实标签，形状为 [batch_size]。

    返回
    ----
    int
        当前 batch 中预测正确的样本数。
    """

    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return int(correct)


def compute_average_loss(total_loss: float, num_samples: int) -> float:
    """
    根据累计 loss 和样本数计算平均 loss。

    参数
    ----
    total_loss : float
        按样本数累计后的 loss。
    num_samples : int
        样本数量。

    返回
    ----
    float
        平均 loss。
    """

    if num_samples <= 0:
        return 0.0

    return total_loss / num_samples


def compute_accuracy(total_correct: int, num_samples: int) -> float:
    """
    根据预测正确数量和总样本数计算准确率。

    参数
    ----
    total_correct : int
        预测正确的样本数量。
    num_samples : int
        总样本数量。

    返回
    ----
    float
        准确率，范围为 [0, 1]。
    """

    if num_samples <= 0:
        return 0.0

    return total_correct / num_samples


def compute_samples_per_sec(num_samples: int, elapsed_time: float) -> float:
    """
    计算系统吞吐量 samples/s。

    参数
    ----
    num_samples : int
        处理的样本数量。
    elapsed_time : float
        耗时，单位为秒。

    返回
    ----
    float
        每秒处理的样本数量。
    """

    if elapsed_time <= 0:
        return 0.0

    return num_samples / elapsed_time


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str = "cpu",
) -> Dict[str, float]:
    """
    在给定数据集上评估模型。

    参数
    ----
    model : nn.Module
        待评估模型。
    data_loader : DataLoader
        测试集或验证集 DataLoader。
    criterion : nn.Module
        损失函数，例如 nn.CrossEntropyLoss()。
    device : torch.device | str
        运行设备。

    返回
    ----
    Dict[str, float]
        包含 loss、accuracy、num_samples 的字典。
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)

        # CrossEntropyLoss 默认返回当前 batch 的平均 loss，因此需要乘以 batch_size 累加
        total_loss += loss.item() * batch_size
        total_correct += count_correct(logits, targets)
        total_samples += batch_size

    avg_loss = compute_average_loss(total_loss, total_samples)
    avg_acc = compute_accuracy(total_correct, total_samples)

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "num_samples": float(total_samples),
    }


def format_metrics(metrics: Dict[str, float], percent: bool = True) -> str:
    """
    将指标字典格式化为字符串，便于打印日志。

    参数
    ----
    metrics : Dict[str, float]
        指标字典。
    percent : bool
        是否将 accuracy 显示为百分数。

    返回
    ----
    str
        格式化后的字符串。
    """

    loss = metrics.get("loss", 0.0)
    acc = metrics.get("accuracy", 0.0)

    if percent:
        return f"loss={loss:.4f}, acc={acc * 100:.2f}%"

    return f"loss={loss:.4f}, acc={acc:.4f}"