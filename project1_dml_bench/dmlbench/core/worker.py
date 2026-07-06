"""
worker.py

功能：
1. 定义 DML-Bench 中的 Worker 抽象；
2. 支持计算单个 mini-batch 梯度；
3. 支持本地训练若干 step；
4. 为 Sync-SGD、Local SGD、Async-SGD 提供统一 worker 接口。
"""

import copy
from typing import Dict, Iterator, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


TensorDict = Dict[str, torch.Tensor]


class Worker:
    """
    分布式训练中的 worker 节点。
    """

    def __init__(
        self,
        worker_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
    ) -> None:
        """
        参数：
            worker_id:
                worker 编号。
            model:
                基础模型。初始化时会 deepcopy，避免多个 worker 共享同一个模型对象。
            train_loader:
                当前 worker 的本地数据。
            device:
                训练设备。
        """

        self.worker_id = worker_id
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.device = device

        self._iterator: Optional[Iterator] = None
        self.num_samples = len(train_loader.dataset)

    def reset_iterator(self) -> None:
        """
        重置本地数据迭代器。
        """

        self._iterator = iter(self.train_loader)

    def get_next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取下一个本地 batch。

        如果当前数据已经遍历完，则自动重新开始。
        """

        if self._iterator is None:
            self.reset_iterator()

        try:
            images, labels = next(self._iterator)
        except StopIteration:
            self.reset_iterator()
            images, labels = next(self._iterator)

        images = images.to(self.device)
        labels = labels.to(self.device)

        return images, labels

    def set_model_state(self, state_dict: TensorDict) -> None:
        """
        将 server 广播的全局模型参数加载到本地模型。
        """

        state_dict = {
            key: value.detach().clone().to(self.device)
            for key, value in state_dict.items()
        }

        self.model.load_state_dict(state_dict)

    def get_model_state(self) -> TensorDict:
        """
        返回当前 worker 本地模型参数。
        """

        return {
            key: value.detach().clone().cpu()
            for key, value in self.model.state_dict().items()
        }

    def compute_gradient(
        self,
        global_state: TensorDict,
        criterion: nn.Module,
    ) -> Dict[str, Any]:
        """
        在一个本地 mini-batch 上计算梯度。

        用于 Sync-SGD / Async-SGD。

        返回：
            {
                "worker_id": int,
                "num_samples": int,
                "loss": float,
                "correct": int,
                "total": int,
                "gradients": Dict[str, Tensor]
            }
        """

        self.set_model_state(global_state)
        self.model.train()

        images, labels = self.get_next_batch()

        self.model.zero_grad(set_to_none=True)

        logits = self.model(images)
        loss = criterion(logits, labels)
        loss.backward()

        gradients: TensorDict = {}

        for name, param in self.model.named_parameters():
            if param.grad is None:
                gradients[name] = torch.zeros_like(param.detach()).cpu()
            else:
                gradients[name] = param.grad.detach().clone().cpu()

        preds = logits.argmax(dim=1)
        correct = int((preds == labels).sum().item())
        total = int(labels.size(0))

        return {
            "worker_id": self.worker_id,
            "num_samples": total,
            "loss": float(loss.item()),
            "correct": correct,
            "total": total,
            "gradients": gradients,
        }

    def train_local_steps(
        self,
        global_state: TensorDict,
        criterion: nn.Module,
        lr: float,
        local_steps: int = 1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> Dict[str, Any]:
        """
        从全局模型出发，在本地训练若干 step。

        用于 Local SGD / Model Averaging。
        """

        if local_steps <= 0:
            raise ValueError("local_steps must be positive.")

        self.set_model_state(global_state)
        self.model.train()

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for _ in range(local_steps):
            images, labels = self.get_next_batch()

            optimizer.zero_grad()
            logits = self.model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            preds = logits.argmax(dim=1)
            correct = int((preds == labels).sum().item())

            total_loss += float(loss.item()) * batch_size
            total_correct += correct
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples

        return {
            "worker_id": self.worker_id,
            "num_samples": total_samples,
            "local_steps": local_steps,
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
            "model_state": self.get_model_state(),
        }

    def __repr__(self) -> str:
        return (
            f"Worker(worker_id={self.worker_id}, "
            f"num_samples={self.num_samples}, "
            f"device={self.device})"
        )


if __name__ == "__main__":
    from dmlbench.data.datasets import get_mnist_datasets
    from dmlbench.data.partition import iid_partition, build_worker_dataloaders
    from dmlbench.models.mlp import build_model
    from dmlbench.utils.seed import set_seed

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, _ = get_mnist_datasets(data_dir="./data", download=True)

    partitions = iid_partition(
        dataset=train_dataset,
        num_workers=4,
        seed=42,
    )

    worker_loaders = build_worker_dataloaders(
        dataset=train_dataset,
        partitions=partitions,
        batch_size=64,
        num_workers=0,
        seed=42,
    )

    base_model = build_model("mlp")
    global_state = base_model.state_dict()
    criterion = nn.CrossEntropyLoss()

    worker = Worker(
        worker_id=0,
        model=base_model,
        train_loader=worker_loaders[0],
        device=device,
    )

    print(worker)

    grad_package = worker.compute_gradient(
        global_state=global_state,
        criterion=criterion,
    )

    print(f"Gradient package keys: {grad_package.keys()}")
    print(f"Loss: {grad_package['loss']:.4f}")
    print(f"Batch total: {grad_package['total']}")
    print(f"Number of gradient tensors: {len(grad_package['gradients'])}")

    local_package = worker.train_local_steps(
        global_state=global_state,
        criterion=criterion,
        lr=0.01,
        local_steps=5,
    )

    print(f"Local avg loss: {local_package['avg_loss']:.4f}")
    print(f"Local avg acc: {local_package['avg_acc']:.2f}%")
    print(f"Number of model tensors: {len(local_package['model_state'])}")