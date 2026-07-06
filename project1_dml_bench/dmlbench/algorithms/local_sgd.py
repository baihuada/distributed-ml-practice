"""
local_sgd.py

功能：
1. 实现 Local SGD / Model Averaging；
2. 使用单进程模拟多个 worker；
3. 每个 worker 持有一份本地数据；
4. server 广播全局模型；
5. 每个 worker 本地训练 local_steps 步；
6. worker 上传本地模型参数；
7. server 对本地模型做加权平均；
8. 记录 loss、accuracy、通信轮数、通信量等指标。

运行示例：
python -m dmlbench.algorithms.local_sgd --model mlp --epochs 10 --batch-size 64 --lr 0.01 --num-workers 4 --local-steps 5 --seed 42
"""

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dmlbench.data.datasets import get_mnist_datasets, get_mnist_dataloaders
from dmlbench.data.partition import iid_partition, build_worker_dataloaders
from dmlbench.models.mlp import build_model
from dmlbench.core.worker import Worker
from dmlbench.core.server import Server
from dmlbench.utils.seed import set_seed
from dmlbench.utils.logger import ExperimentLogger


def get_device(device: str = "auto") -> torch.device:
    """
    获取训练设备。

    参数：
        device:
            "auto"：有 GPU 就用 GPU，否则用 CPU；
            "cpu" ：强制使用 CPU；
            "cuda"：强制使用 GPU。
    """

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but device='cuda' was requested.")

    return torch.device(device)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    """
    计算一个 batch 的预测正确数和总样本数。
    """

    preds = logits.argmax(dim=1)
    correct = int((preds == labels).sum().item())
    total = int(labels.size(0))

    return correct, total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    在测试集上评估 server 的全局模型。

    Local SGD 中，server.model 是聚合后的全局模型。
    因此测试时评估 server.model，而不是某个 worker.model。
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

        total_loss += float(loss.item()) * batch_size
        total_correct += correct
        total_samples += total

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples

    return {
        "test_loss": avg_loss,
        "test_acc": avg_acc,
    }


def build_local_workers(
    base_model: nn.Module,
    train_dataset,
    num_workers: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    dataloader_workers: int = 0,
    pin_memory: bool = False,
) -> List[Worker]:
    """
    构造 Local SGD 使用的多个 worker。

    步骤：
        1. 对训练集做 IID 划分；
        2. 为每个 worker 构造本地 DataLoader；
        3. 为每个 worker 创建 Worker 对象。
    """

    partitions = iid_partition(
        dataset=train_dataset,
        num_workers=num_workers,
        seed=seed,
        shuffle=True,
        drop_last=False,
    )

    worker_loaders = build_worker_dataloaders(
        dataset=train_dataset,
        partitions=partitions,
        batch_size=batch_size,
        num_workers=dataloader_workers,
        seed=seed,
        shuffle=True,
        pin_memory=pin_memory,
    )

    workers: List[Worker] = []

    for worker_id, loader in enumerate(worker_loaders):
        worker = Worker(
            worker_id=worker_id,
            model=base_model,
            train_loader=loader,
            device=device,
        )
        workers.append(worker)

    return workers


def train_one_epoch_local_sgd(
    server: Server,
    workers: List[Worker],
    criterion: nn.Module,
    rounds_per_epoch: int,
    local_steps: int,
    lr: float,
    epoch: int,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    训练一个 epoch 的 Local SGD。

    每一个 communication round 的流程：

        1. server.broadcast_model()
           server 广播当前全局模型。

        2. worker.train_local_steps()
           每个 worker 从全局模型出发，本地训练 local_steps 步。

        3. server.aggregate_model_states()
           server 对所有 worker 上传的本地模型参数做平均。

        4. server.update_model_by_state()
           server 用平均模型作为新的全局模型。
    """

    if local_steps <= 0:
        raise ValueError("local_steps must be positive.")

    total_loss = 0.0
    total_correct_estimate = 0.0
    total_samples = 0

    num_workers = len(workers)
    comm_rounds = 0
    comm_bytes = 0

    iterator = tqdm(
        range(rounds_per_epoch),
        desc=f"Local-SGD Epoch {epoch}",
        leave=False,
        disable=not show_progress,
    )

    for _ in iterator:
        # 1. server 广播当前全局模型
        global_state = server.broadcast_model()

        # 2. 每个 worker 本地训练 local_steps 步
        model_packages = []

        for worker in workers:
            local_pkg = worker.train_local_steps(
                global_state=global_state,
                criterion=criterion,
                lr=lr,
                local_steps=local_steps,
                momentum=momentum,
                weight_decay=weight_decay,
            )
            model_packages.append(local_pkg)

        # 3. server 对本地模型做平均
        avg_state = server.aggregate_model_states(model_packages)

        # 4. server 更新全局模型
        server.update_model_by_state(avg_state)

        # 5. 统计训练指标
        round_loss_sum = 0.0
        round_correct_estimate = 0.0
        round_samples = 0

        for pkg in model_packages:
            worker_samples = int(pkg["num_samples"])
            worker_loss = float(pkg["avg_loss"])
            worker_acc = float(pkg["avg_acc"])

            round_loss_sum += worker_loss * worker_samples
            round_correct_estimate += (worker_acc / 100.0) * worker_samples
            round_samples += worker_samples

        total_loss += round_loss_sum
        total_correct_estimate += round_correct_estimate
        total_samples += round_samples

        # 6. 统计通信量
        # 一轮 Local SGD 通信包括：
        # server -> workers：广播模型
        # workers -> server：上传本地模型
        comm_rounds += 1
        comm_bytes += server.estimate_local_sgd_round_bytes(num_workers=num_workers)

        avg_loss_so_far = total_loss / total_samples
        avg_acc_so_far = 100.0 * total_correct_estimate / total_samples

        iterator.set_postfix(
            {
                "loss": avg_loss_so_far,
                "acc": avg_acc_so_far,
                "round": comm_rounds,
            }
        )

    train_loss = total_loss / total_samples
    train_acc = 100.0 * total_correct_estimate / total_samples

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "comm_rounds": comm_rounds,
        "comm_bytes": comm_bytes,
    }


def run_local_sgd(
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
    num_workers: int = 4,
    local_steps: int = 5,
    device: str = "auto",
    dataloader_workers: int = 0,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    运行 Local SGD 实验。
    """

    if num_workers <= 0:
        raise ValueError("num_workers must be positive.")
    if local_steps <= 0:
        raise ValueError("local_steps must be positive.")

    # 1. 固定随机种子
    set_seed(seed)

    # 2. 设置设备
    torch_device = get_device(device)
    print(f"Using device: {torch_device}")

    # 3. 加载 MNIST
    train_dataset, _ = get_mnist_datasets(
        data_dir=data_dir,
        download=True,
        normalize=True,
    )

    _, test_loader = get_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        num_workers=dataloader_workers,
        seed=seed,
        download=True,
        normalize=True,
        pin_memory=(torch_device.type == "cuda"),
    )

    # 4. 创建基础模型
    base_model = build_model(model_name=model_name)

    # 5. 创建 server
    server = Server(
        model=base_model,
        lr=lr,
        device=torch_device,
    )

    # 6. 创建 workers
    workers = build_local_workers(
        base_model=base_model,
        train_dataset=train_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        seed=seed,
        device=torch_device,
        dataloader_workers=dataloader_workers,
        pin_memory=(torch_device.type == "cuda"),
    )

    print(server)
    for worker in workers:
        print(worker)

    # 7. 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 8. 计算每个 epoch 的通信轮数
    num_batches_per_worker = min(len(worker.train_loader) for worker in workers)
    rounds_per_epoch = math.ceil(num_batches_per_worker / local_steps)

    print(f"Batches per worker: {num_batches_per_worker}")
    print(f"Local steps per communication round: {local_steps}")
    print(f"Communication rounds per epoch: {rounds_per_epoch}")

    # 9. 创建日志器
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_name = (
        f"local_sgd_{model_name}"
        f"_workers{num_workers}"
        f"_local{local_steps}"
        f"_epochs{epochs}"
        f"_bs{batch_size}"
        f"_lr{lr}"
        f"_seed{seed}.csv"
    )

    logger = ExperimentLogger(
        save_path=str(output_path / csv_name),
        auto_save=True,
    )

    # 10. 主训练循环
    records: List[Dict[str, Any]] = []

    total_comm_rounds = 0
    total_comm_bytes = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch_local_sgd(
            server=server,
            workers=workers,
            criterion=criterion,
            rounds_per_epoch=rounds_per_epoch,
            local_steps=local_steps,
            lr=lr,
            epoch=epoch,
            momentum=momentum,
            weight_decay=weight_decay,
            show_progress=show_progress,
        )

        test_metrics = evaluate(
            model=server.model,
            test_loader=test_loader,
            criterion=criterion,
            device=torch_device,
        )

        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time

        total_comm_rounds += int(train_metrics["comm_rounds"])
        total_comm_bytes += int(train_metrics["comm_bytes"])

        processed_samples_per_comm_round = batch_size * num_workers * local_steps

        record = {
            "algorithm": "local_sgd",
            "model": model_name,
            "epoch": epoch,
            "num_workers": num_workers,
            "batch_size_per_worker": batch_size,
            "local_steps": local_steps,
            "processed_samples_per_comm_round": processed_samples_per_comm_round,
            "batches_per_worker": num_batches_per_worker,
            "rounds_per_epoch": rounds_per_epoch,
            "train_loss": train_metrics["train_loss"],
            "train_acc": train_metrics["train_acc"],
            "test_loss": test_metrics["test_loss"],
            "test_acc": test_metrics["test_acc"],
            "epoch_time": epoch_time,
            "elapsed_time": elapsed_time,
            "comm_round": total_comm_rounds,
            "comm_bytes": total_comm_bytes,
            "comm_mb": total_comm_bytes / (1024 ** 2),
            "server_version": server.version,
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
            f"local_steps={local_steps} "
            f"comm_round={record['comm_round']} "
            f"comm_mb={record['comm_mb']:.2f} "
            f"time={record['epoch_time']:.2f}s"
        )

    final_record = records[-1]

    print("=" * 80)
    print("Local SGD Finished")
    print(f"Final Test Accuracy: {final_record['test_acc']:.2f}%")
    print(f"Local Steps: {local_steps}")
    print(f"Total Communication Rounds: {final_record['comm_round']}")
    print(f"Total Communication: {final_record['comm_mb']:.2f} MB")
    print(f"Server Version: {server.version}")
    print(f"Log saved to: {output_path / csv_name}")
    print("=" * 80)

    return records


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """

    parser = argparse.ArgumentParser(description="Run Local SGD on MNIST.")

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
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_local_sgd(
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
        num_workers=args.num_workers,
        local_steps=args.local_steps,
        device=args.device,
        dataloader_workers=args.dataloader_workers,
        show_progress=not args.no_progress,
    )