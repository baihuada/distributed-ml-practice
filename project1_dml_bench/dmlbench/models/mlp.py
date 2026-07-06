"""
mlp.py

功能：
1. 定义 MNIST 上使用的 Logistic Regression 和 MLP；
2. 提供 build_model() 工厂函数，方便后续根据配置创建模型；
3. 为 Centralized SGD、Sync-SGD、Local SGD、Async-SGD 提供统一模型接口。
"""

from typing import Literal

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """
    MNIST Logistic Regression 模型。

    输入：
        x: [batch_size, 1, 28, 28]

    输出：
        logits: [batch_size, 10]

    说明：
        这里的 Logistic Regression 实际上是多分类 softmax 回归。
        模型结构为：
            Flatten -> Linear(784, 10)
    """

    def __init__(self, input_dim: int = 784, num_classes: int = 10) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


class MLP(nn.Module):
    """
    MNIST 简单 MLP 模型。

    输入：
        x: [batch_size, 1, 28, 28]

    输出：
        logits: [batch_size, 10]

    默认结构：
        Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 10)

    说明：
        这是 DML-Bench 第一版推荐使用的主模型。
        它比 Logistic Regression 稍强，但仍然足够简单，方便观察分布式优化机制。
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
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

        layers.append(nn.Linear(hidden_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits


def build_model(
    model_name: Literal["logistic", "mlp"] = "mlp",
    input_dim: int = 784,
    hidden_dim: int = 128,
    num_classes: int = 10,
    dropout: float = 0.0,
) -> nn.Module:
    """
    根据模型名称创建模型。

    参数：
        model_name:
            模型名称，可选：
            - "logistic"
            - "mlp"
        input_dim:
            输入维度，MNIST 默认为 784。
        hidden_dim:
            MLP 隐藏层维度。
        num_classes:
            分类类别数，MNIST 为 10。
        dropout:
            dropout 比例，默认 0.0。

    返回：
        PyTorch 模型。
    """

    model_name = model_name.lower()

    if model_name == "logistic":
        return LogisticRegression(
            input_dim=input_dim,
            num_classes=num_classes,
        )

    if model_name == "mlp":
        return MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


if __name__ == "__main__":
    # 简单测试：检查模型输入输出维度是否正确。
    batch_size = 64
    x = torch.randn(batch_size, 1, 28, 28)

    for name in ["logistic", "mlp"]:
        model = build_model(model_name=name)
        logits = model(x)
        print(f"{name} output shape: {logits.shape}")