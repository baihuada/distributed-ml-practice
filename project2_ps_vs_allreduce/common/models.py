"""
模型定义模块。

本文件负责：
1. 定义 Logistic Regression；
2. 定义简单 MLP；
3. 提供统一的 build_model() 接口；
4. 提供模型参数量统计函数。

第一阶段默认使用 MNIST + MLP。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """
    Logistic Regression 模型。

    对 MNIST 来说，输入图像大小为 [1, 28, 28]。
    展平后输入维度为 784。
    输出维度为 10，对应 10 个数字类别。
    """

    def __init__(self, input_dim: int = 784, num_classes: int = 10) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数
        ----
        x : torch.Tensor
            输入图像，形状通常为 [batch_size, 1, 28, 28]。

        返回
        ----
        torch.Tensor
            分类 logits，形状为 [batch_size, 10]。
        """

        x = self.flatten(x)
        logits = self.linear(x)
        return logits


class MLP(nn.Module):
    """
    简单多层感知机模型。

    结构：
    输入层 784
    隐藏层 hidden_dim
    ReLU
    隐藏层 hidden_dim
    ReLU
    输出层 10

    这个模型足够用于 MNIST 第一版系统机制验证。
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]

        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        layers.extend(
            [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ]
        )

        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(hidden_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数
        ----
        x : torch.Tensor
            输入图像，形状通常为 [batch_size, 1, 28, 28]。

        返回
        ----
        torch.Tensor
            分类 logits，形状为 [batch_size, 10]。
        """

        logits = self.net(x)
        return logits


def build_model(
    model_name: str = "mlp",
    dataset_name: str = "mnist",
    hidden_dim: int = 256,
    dropout: float = 0.0,
) -> nn.Module:
    """
    根据名称构造模型。

    参数
    ----
    model_name : str
        模型名称，支持 "logistic" 和 "mlp"。
    dataset_name : str
        数据集名称。当前 MNIST / Fashion-MNIST 输入维度相同。
    hidden_dim : int
        MLP 隐藏层维度。
    dropout : float
        MLP dropout 概率，第一版默认 0.0。

    返回
    ----
    nn.Module
        PyTorch 模型。
    """

    model = model_name.lower()
    dataset = dataset_name.lower()

    input_dim, num_classes = get_model_dims(dataset)

    if model in ["logistic", "lr", "logreg"]:
        return LogisticRegression(
            input_dim=input_dim,
            num_classes=num_classes,
        )

    if model == "mlp":
        return MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    raise ValueError(
        f"不支持的模型: {model_name}. 当前只支持 logistic 和 mlp。"
    )


def get_model_dims(dataset_name: str) -> Tuple[int, int]:
    """
    根据数据集返回输入维度和类别数。

    参数
    ----
    dataset_name : str
        数据集名称。

    返回
    ----
    Tuple[int, int]
        input_dim 和 num_classes。
    """

    dataset = dataset_name.lower()

    if dataset in ["mnist", "fashion_mnist", "fashion-mnist", "fashion"]:
        input_dim = 28 * 28
        num_classes = 10
        return input_dim, num_classes

    raise ValueError(
        f"不支持的数据集: {dataset_name}. 当前只支持 mnist 和 fashion_mnist。"
    )


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    统计模型参数量。

    参数
    ----
    model : nn.Module
        待统计的模型。
    trainable_only : bool
        如果为 True，只统计 requires_grad=True 的参数。

    返回
    ----
    int
        参数总量。
    """

    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())


def estimate_model_size_mb(model: nn.Module, bytes_per_param: int = 4) -> float:
    """
    估算模型参数大小，单位为 MB。

    参数
    ----
    model : nn.Module
        待估算的模型。
    bytes_per_param : int
        每个参数占用的字节数。float32 默认为 4 字节。

    返回
    ----
    float
        模型参数大小，单位 MB。
    """

    num_params = count_parameters(model)
    size_bytes = num_params * bytes_per_param
    size_mb = size_bytes / (1024**2)
    return size_mb


def print_model_info(model: nn.Module) -> None:
    """
    打印模型基本信息。

    参数
    ----
    model : nn.Module
        待打印信息的模型。
    """

    num_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    size_mb = estimate_model_size_mb(model)

    print(model)
    print(f"模型参数量: {num_params}")
    print(f"可训练参数量: {trainable_params}")
    print(f"模型大小估算: {size_mb:.4f} MB")


def check_forward_shape(
    model: nn.Module,
    batch_size: int = 64,
    device: str = "cpu",
) -> None:
    """
    检查模型 forward 输出维度是否正确。

    参数
    ----
    model : nn.Module
        待检查的模型。
    batch_size : int
        测试 batch size。
    device : str
        运行设备。
    """

    model = model.to(device)
    model.eval()

    dummy_x = torch.randn(batch_size, 1, 28, 28, device=device)

    with torch.no_grad():
        logits = model(dummy_x)

    print(f"输入形状: {tuple(dummy_x.shape)}")
    print(f"输出形状: {tuple(logits.shape)}")

    expected_shape = (batch_size, 10)
    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            f"模型输出形状错误，期望 {expected_shape}，实际 {tuple(logits.shape)}"
        )