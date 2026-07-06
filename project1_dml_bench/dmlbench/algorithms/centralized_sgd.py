"""
centralized_sgd.py

功能：
1. 实现单机 Centralized SGD；
2. 作为 Sync-SGD、Local SGD、Async-SGD 的 baseline；
3. 保存每个 epoch 的 train/test loss 和 accuracy；
4. 支持命令行直接运行。

运行示例：
python -m dmlbench.algorithms.centralized_sgd \
  --model mlp \
  --epochs 10 \
  --batch-size 64 \
  --lr 0.01 \
  --seed 42
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dmlbench.data.datasets import get_mnist_dataloaders
from dmlbench.models.mlp import build_model
from dmlbench.utils.seed import set_seed
from dmlbench.utils.logger import ExperimentLogger


def get_device(device: str = "auto") -> torch.device:
    """
    获取训练设备。

    参数：
        device:
            - "auto": 有 GPU 则使用 cuda，否则使用 cpu
            - "cpu": 强制使用 CPU
            - "cuda": 强制使用 GPU

    返回：
        torch.device
    """

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but device='cuda' was requested.")

    return torch.device(device)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    """
    计算一个 batch 中预测正确的样本数。

    参数：
        logits:
            模型输出，形状为 [batch_size, num_classes]。
        labels:
            真实标签，形状为 [batch_size]。

    返回：
        correct:
            预测正确数量。
        total:
            样本总数。
    """

    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    训练一个 epoch。

    参数：
        model:
            待训练模型。
        train_loader:
            训练集 DataLoader。
        criterion:
            损失函数。
        optimizer:
            优化器。
        device:
            训练设备。
        epoch:
            当前 epoch 编号。
        show_progress:
            是否显示 tqdm 进度条。

    返回：
        包含 train_loss、train_acc 的字典。
    """

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    iterator = tqdm(
        train_loader,
        desc=f"Train Epoch {epoch}",
        leave=False,
        disable=not show_progress,
    )

    for images, labels in iterator:
        images = images.to(device)
        labels = labels.to(device)

        # 1. 清空上一轮梯度。
        optimizer.zero_grad()

        # 2. 前向传播。
        logits = model(images)

        # 3. 计算损失。
        loss = criterion(logits, labels)

        # 4. 反向传播，计算梯度。
        loss.backward()

        # 5. SGD 更新参数。
        optimizer.step()

        # 6. 统计 loss 和 accuracy。
        batch_size = labels.size(0)
        correct, total = compute_accuracy(logits, labels)

        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += total

        iterator.set_postfix({
            "loss": total_loss / total_samples,
            "acc": 100.0 * total_correct / total_samples,
        })

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples

    return {
        "train_loss": avg_loss,
        "train_acc": avg_acc,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    在测试集上评估模型。

    参数：
        model:
            待评估模型。
        test_loader:
            测试集 DataLoader。
        criterion:
            损失函数。
        device:
            评估设备。

    返回：
        包含 test_loss、test_acc 的字典。
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        correct, total = compute_accuracy(logits, labels)

        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += total

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples

    return {
        "test_loss": avg_loss,
        "test_acc": avg_acc,
    }


def run_centralized_sgd(
    model_name: str = "mlp",
    data_dir: str = "./data",
    output_dir: str = "./results/raw",
    epochs: int = 10,
    batch_size: int = 64,
    test_batch_size: int = 256,
    lr: float = 0.01,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    seed: int = 42,
    device: str = "auto",
    num_workers: int = 0,
    show_progress: bool = True,
) -> List[Dict[str, float]]:
    """
    运行 Centralized SGD 实验。

    参数：
        model_name:
            模型名称，"logistic" 或 "mlp"。
        data_dir:
            MNIST 数据目录。
        output_dir:
            日志保存目录。
        epochs:
            训练 epoch 数。
        batch_size:
            训练 batch size。
        test_batch_size:
            测试 batch size。
        lr:
            学习率。
        momentum:
            SGD momentum。
        weight_decay:
            权重衰减。
        seed:
            随机种子。
        device:
            训练设备。
        num_workers:
            DataLoader 子进程数。
        show_progress:
            是否显示训练进度条。

    返回：
        每个 epoch 的实验记录列表。
    """

    # 1. 固定随机种子。
    set_seed(seed)

    # 2. 设置训练设备。
    torch_device = get_device(device)
    print(f"Using device: {torch_device}")

    # 3. 加载数据。
    train_loader, test_loader = get_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        seed=seed,
        download=True,
        normalize=True,
        pin_memory=(torch_device.type == "cuda"),
    )

    # 4. 创建模型。
    model = build_model(model_name=model_name)
    model = model.to(torch_device)

    # 5. 定义损失函数和优化器。
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # 6. 创建日志器。
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_name = (
        f"centralized_{model_name}"
        f"_epochs{epochs}"
        f"_bs{batch_size}"
        f"_lr{lr}"
        f"_seed{seed}.csv"
    )

    logger = ExperimentLogger(
        save_path=str(output_path / csv_name),
        auto_save=True,
    )

    # 7. 训练循环。
    records: List[Dict[str, float]] = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=torch_device,
            epoch=epoch,
            show_progress=show_progress,
        )

        test_metrics = evaluate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=torch_device,
        )

        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time

        record = {
            "algorithm": "centralized_sgd",
            "model": model_name,
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "train_acc": train_metrics["train_acc"],
            "test_loss": test_metrics["test_loss"],
            "test_acc": test_metrics["test_acc"],
            "epoch_time": epoch_time,
            "elapsed_time": elapsed_time,

            # Centralized SGD 是单机训练，当前阶段没有分布式通信。
            # 后续 Sync-SGD / Local SGD / Async-SGD 会真实填写通信量。
            "comm_round": 0,
            "comm_bytes": 0,
            "virtual_time": elapsed_time,
        }

        logger.log(record)
        records.append(record)

        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"train_loss={record['train_loss']:.4f} "
            f"train_acc={record['train_acc']:.2f}% "
            f"test_loss={record['test_loss']:.4f} "
            f"test_acc={record['test_acc']:.2f}% "
            f"time={record['epoch_time']:.2f}s"
        )

    final_record = records[-1]
    print("=" * 80)
    print("Centralized SGD Finished")
    print(f"Final Test Accuracy: {final_record['test_acc']:.2f}%")
    print(f"Log saved to: {output_path / csv_name}")
    print("=" * 80)

    return records


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """

    parser = argparse.ArgumentParser(description="Run Centralized SGD on MNIST.")

    parser.add_argument("--model", type=str, default="mlp", choices=["logistic", "mlp"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results/raw")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_centralized_sgd(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        show_progress=not args.no_progress,
    )