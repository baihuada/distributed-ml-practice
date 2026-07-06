"""
server.py

功能：
1. 定义 DML-Bench 中的 Server 抽象；
2. 支持广播模型；
3. 支持梯度聚合与全局更新；
4. 支持模型平均；
5. 支持通信量估计；
6. 为 Sync-SGD、Local SGD、Async-SGD 提供统一 server 接口。
"""

from typing import Dict, List, Optional, Sequence, Any

import torch
import torch.nn as nn


TensorDict = Dict[str, torch.Tensor]


class Server:
    """
    分布式训练中的中心 server。
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.01,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        参数：
            model:
                全局模型。
            lr:
                server 端学习率，用于梯度更新。
            device:
                设备。若为 None，则自动选择 cuda 或 cpu。
        """

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(device)
        self.lr = lr
        self.device = device
        self.version = 0

    def broadcast_model(self) -> TensorDict:
        """
        广播全局模型参数。

        返回 CPU 上的模型参数副本。
        """

        return {
            key: value.detach().clone().cpu()
            for key, value in self.model.state_dict().items()
        }

    def set_model_state(self, state_dict: TensorDict) -> None:
        """
        设置 server 全局模型参数。
        """

        state_dict = {
            key: value.detach().clone().to(self.device)
            for key, value in state_dict.items()
        }

        self.model.load_state_dict(state_dict)

    def get_model_state(self) -> TensorDict:
        """
        获取 server 当前全局模型参数。
        """

        return self.broadcast_model()

    def aggregate_gradients(
        self,
        gradient_packages: Sequence[Dict[str, Any]],
        weights: Optional[Sequence[float]] = None,
    ) -> TensorDict:
        """
        聚合多个 worker 上传的梯度。

        如果 weights 为 None，则按每个 worker 当前 batch 样本数加权。
        """

        if len(gradient_packages) == 0:
            raise ValueError("gradient_packages cannot be empty.")

        if weights is None:
            sample_counts = [pkg["num_samples"] for pkg in gradient_packages]
            total_samples = sum(sample_counts)
            weights = [count / total_samples for count in sample_counts]
        else:
            if len(weights) != len(gradient_packages):
                raise ValueError("weights and gradient_packages must have the same length.")
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]

        first_grads = gradient_packages[0]["gradients"]

        avg_gradients: TensorDict = {
            name: torch.zeros_like(grad)
            for name, grad in first_grads.items()
        }

        for pkg, weight in zip(gradient_packages, weights):
            gradients = pkg["gradients"]
            for name in avg_gradients:
                avg_gradients[name] += gradients[name] * weight

        return avg_gradients

    def apply_gradients(self, gradients: TensorDict) -> None:
        """
        使用聚合梯度更新全局模型。

        更新规则：
            w_{t+1} = w_t - lr * grad
        """

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in gradients:
                    raise KeyError(f"Gradient for parameter '{name}' not found.")

                grad = gradients[name].to(self.device)
                param.data -= self.lr * grad

        self.version += 1

    def aggregate_model_states(
        self,
        model_packages: Sequence[Dict[str, Any]],
        weights: Optional[Sequence[float]] = None,
    ) -> TensorDict:
        """
        聚合多个 worker 上传的本地模型参数。

        用于 Local SGD / Model Averaging。
        """

        if len(model_packages) == 0:
            raise ValueError("model_packages cannot be empty.")

        if weights is None:
            sample_counts = [pkg["num_samples"] for pkg in model_packages]
            total_samples = sum(sample_counts)
            weights = [count / total_samples for count in sample_counts]
        else:
            if len(weights) != len(model_packages):
                raise ValueError("weights and model_packages must have the same length.")
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]

        first_state = model_packages[0]["model_state"]

        avg_state: TensorDict = {
            name: torch.zeros_like(value)
            for name, value in first_state.items()
        }

        for pkg, weight in zip(model_packages, weights):
            state = pkg["model_state"]
            for name in avg_state:
                avg_state[name] += state[name] * weight

        return avg_state

    def update_model_by_state(self, state_dict: TensorDict) -> None:
        """
        用给定 state_dict 更新 server 全局模型。

        用于 Local SGD 中模型平均后的更新。
        """

        self.set_model_state(state_dict)
        self.version += 1

    def count_parameters(self) -> int:
        """
        统计模型可训练参数数量。
        """

        return sum(param.numel() for param in self.model.parameters() if param.requires_grad)

    def model_size_bytes(self, bytes_per_param: int = 4) -> int:
        """
        估算模型参数大小。

        默认使用 float32，即每个参数 4 bytes。
        """

        return self.count_parameters() * bytes_per_param

    def estimate_broadcast_bytes(self, num_workers: int, bytes_per_param: int = 4) -> int:
        """
        估算 server 向所有 worker 广播模型的通信量。
        """

        return self.model_size_bytes(bytes_per_param) * num_workers

    def estimate_upload_gradient_bytes(self, num_workers: int, bytes_per_param: int = 4) -> int:
        """
        估算所有 worker 上传梯度的通信量。
        """

        return self.model_size_bytes(bytes_per_param) * num_workers

    def estimate_upload_model_bytes(self, num_workers: int, bytes_per_param: int = 4) -> int:
        """
        估算所有 worker 上传模型参数的通信量。
        """

        return self.model_size_bytes(bytes_per_param) * num_workers

    def estimate_sync_sgd_round_bytes(self, num_workers: int, bytes_per_param: int = 4) -> int:
        """
        估算一轮 Sync-SGD 通信量。

        包括：
            1. server 广播模型；
            2. worker 上传梯度。
        """

        return (
            self.estimate_broadcast_bytes(num_workers, bytes_per_param)
            + self.estimate_upload_gradient_bytes(num_workers, bytes_per_param)
        )

    def estimate_local_sgd_round_bytes(self, num_workers: int, bytes_per_param: int = 4) -> int:
        """
        估算一轮 Local SGD 通信量。

        包括：
            1. server 广播模型；
            2. worker 上传本地模型。
        """

        return (
            self.estimate_broadcast_bytes(num_workers, bytes_per_param)
            + self.estimate_upload_model_bytes(num_workers, bytes_per_param)
        )

    def __repr__(self) -> str:
        return (
            f"Server(version={self.version}, "
            f"lr={self.lr}, "
            f"num_params={self.count_parameters()}, "
            f"model_size={self.model_size_bytes() / 1024:.2f} KB, "
            f"device={self.device})"
        )


if __name__ == "__main__":
    from dmlbench.models.mlp import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model("mlp")
    server = Server(model=model, lr=0.01, device=device)

    print(server)

    state = server.broadcast_model()
    print(f"Number of tensors in state_dict: {len(state)}")
    print(f"Model parameters: {server.count_parameters()}")
    print(f"Model size: {server.model_size_bytes() / 1024:.2f} KB")

    fake_gradients = {}
    for name, param in server.model.named_parameters():
        fake_gradients[name] = torch.ones_like(param.detach()).cpu() * 0.01

    print(f"Version before update: {server.version}")
    server.apply_gradients(fake_gradients)
    print(f"Version after update: {server.version}")

    print(f"One Sync-SGD round bytes with 4 workers: {server.estimate_sync_sgd_round_bytes(4)}")