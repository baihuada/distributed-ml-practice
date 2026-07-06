# dmlbench/algorithms/async_sgd.py
"""
async_sgd.py

功能：
1. 实现 Async-SGD / Asynchronous Stochastic Gradient Descent；
2. 使用单进程模拟多个异步 worker；
3. 每个 worker 持有一份本地数据，并从 server 拉取当前全局模型；
4. worker 基于拉取到的模型计算本地 mini-batch 梯度；
5. 使用事件队列 event_queue 和虚拟时间 virtual_time 模拟不同 worker 的完成顺序；
6. worker 完成计算后立即上传梯度，server 收到一个梯度就立即更新一次，不等待其他 worker；
7. 记录 worker 拉取模型时的版本号 pull_version，并计算 stale gradient 的 staleness；
8. 支持通过 worker_delays 模拟 straggler，例如 1,1,1,5 表示第 4 个 worker 是慢节点；
9. 记录 loss、accuracy、通信轮数、通信量、虚拟时间、平均 staleness、最大 staleness 等指标。

运行示例：
python -m dmlbench.algorithms.async_sgd --model mlp --epochs 10 --batch-size 64 --lr 0.01 --num-workers 4 --worker-delays 1,1,1,1 --seed 42

慢节点示例：
python -m dmlbench.algorithms.async_sgd --model mlp --epochs 10 --batch-size 64 --lr 0.01 --num-workers 4 --worker-delays 1,1,1,5 --seed 42
"""


import argparse
import heapq
import itertools
import os
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dmlbench.utils.seed import set_seed
from dmlbench.data.datasets import get_mnist_datasets
from dmlbench.data.partition import iid_partition, build_worker_dataloaders
from dmlbench.models.mlp import build_model
from dmlbench.core.worker import Worker
from dmlbench.core.server import Server


def get_device(device: str = "auto") -> torch.device:
    """
    Select training device.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy for a batch.
    """
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate global model on test set.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)

    return avg_loss, avg_acc


def parse_worker_delays(worker_delays: str, num_workers: int) -> List[float]:
    """
    Parse worker delay string.

    Example:
        "1,1,1,5" -> [1.0, 1.0, 1.0, 5.0]
    """
    if worker_delays is None or worker_delays.strip() == "":
        return [1.0 for _ in range(num_workers)]

    delays = [float(x.strip()) for x in worker_delays.split(",") if x.strip() != ""]

    if len(delays) != num_workers:
        raise ValueError(
            f"worker_delays length must equal num_workers. "
            f"Got {len(delays)} delays, but num_workers={num_workers}."
        )

    if any(d <= 0 for d in delays):
        raise ValueError("All worker delays must be positive.")

    return delays


def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Clone a model state_dict to avoid accidental reference sharing.
    """
    return {k: v.detach().clone() for k, v in state_dict.items()}


def extract_gradients(gradient_package: Any):
    """
    Extract gradients from worker output.

    This function is written defensively because different versions of Worker
    may use slightly different key names.
    """
    if isinstance(gradient_package, dict):
        if "gradients" in gradient_package:
            return gradient_package["gradients"]
        if "grads" in gradient_package:
            return gradient_package["grads"]
        if "gradient" in gradient_package:
            return gradient_package["gradient"]

    return gradient_package


def extract_batch_size(gradient_package: Any) -> int:
    """
    Extract batch size from worker output.
    """
    if isinstance(gradient_package, dict):
        for key in ["batch_size", "num_samples", "samples"]:
            if key in gradient_package:
                return int(gradient_package[key])
    return 1


def extract_loss(gradient_package: Any) -> float:
    """
    Extract batch training loss from worker output.
    """
    if isinstance(gradient_package, dict):
        for key in ["loss", "train_loss", "batch_loss"]:
            if key in gradient_package:
                return float(gradient_package[key])
    return 0.0


def extract_acc(gradient_package: Any) -> float:
    """
    Extract batch training accuracy from worker output.

    Your current Worker.compute_gradient() returns correct/total rather than acc.
    Therefore this function first uses correct/total, and then falls back to
    possible accuracy keys for compatibility.
    """
    if isinstance(gradient_package, dict):
        if "correct" in gradient_package and "total" in gradient_package:
            total = int(gradient_package["total"])
            if total > 0:
                return float(gradient_package["correct"]) / float(total)

        for key in ["acc", "accuracy", "train_acc", "batch_acc"]:
            if key in gradient_package:
                acc = float(gradient_package[key])
                if acc > 1.0:
                    acc = acc / 100.0
                return acc

    return 0.0


def build_async_workers(
    base_model: torch.nn.Module,
    train_dataset,
    num_workers: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    dataloader_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[List[Worker], List[DataLoader]]:
    """
    Build IID workers for Async-SGD.
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

    workers = []

    for worker_id, loader in enumerate(worker_loaders):
        worker = Worker(
            worker_id=worker_id,
            model=base_model,
            train_loader=loader,
            device=device,
        )
        workers.append(worker)

    return workers, worker_loaders


def schedule_worker_task(
    event_queue: List[Tuple[float, int, Dict[str, Any]]],
    counter,
    current_time: float,
    worker_id: int,
    worker_delay: float,
    worker: Worker,
    server: Server,
    criterion: torch.nn.Module,
):
    """
    Schedule one asynchronous worker task.

    The worker pulls the current server model immediately, computes a gradient
    based on that pulled model, and then its result will be delivered at
    current_time + worker_delay.

    In this simulation, the gradient is computed immediately, but it is applied
    only when the virtual finish time is reached.
    """
    pull_version = int(server.version)
    pulled_state = clone_state_dict(server.broadcast_model())

    gradient_package = worker.compute_gradient(
        global_state=pulled_state,
        criterion=criterion,
    )

    finish_time = current_time + worker_delay

    task = {
        "worker_id": worker_id,
        "pull_version": pull_version,
        "finish_time": finish_time,
        "gradient_package": gradient_package,
    }

    heapq.heappush(event_queue, (finish_time, next(counter), task))


def train_one_epoch_async_sgd(
    server: Server,
    workers: List[Worker],
    criterion: torch.nn.Module,
    updates_per_epoch: int,
    worker_delays: List[float],
    start_virtual_time: float,
    epoch: int,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Train one epoch with Async-SGD.

    Difference from Sync-SGD:
        Sync-SGD:
            server waits for all workers, averages gradients, then updates once.

        Async-SGD:
            server updates immediately after receiving one worker gradient.
    """
    event_queue = []
    counter = itertools.count()

    current_time = start_virtual_time
    scheduled_updates = 0
    completed_updates = 0

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    staleness_values = []

    # Initially schedule one task for each worker.
    initial_tasks = min(len(workers), updates_per_epoch)
    for worker_id in range(initial_tasks):
        schedule_worker_task(
            event_queue=event_queue,
            counter=counter,
            current_time=current_time,
            worker_id=worker_id,
            worker_delay=worker_delays[worker_id],
            worker=workers[worker_id],
            server=server,
            criterion=criterion,
        )
        scheduled_updates += 1

    progress = tqdm(
        total=updates_per_epoch,
        desc=f"Async-SGD Epoch {epoch}",
        disable=not show_progress,
    )

    while completed_updates < updates_per_epoch:
        finish_time, _, task = heapq.heappop(event_queue)

        current_time = finish_time
        worker_id = task["worker_id"]
        pull_version = task["pull_version"]
        gradient_package = task["gradient_package"]

        current_version = int(server.version)
        staleness = current_version - pull_version
        staleness_values.append(staleness)

        gradients = extract_gradients(gradient_package)
        server.apply_gradients(gradients)

        batch_size = extract_batch_size(gradient_package)
        batch_loss = extract_loss(gradient_package)
        batch_acc = extract_acc(gradient_package)

        total_loss += batch_loss * batch_size
        total_acc += batch_acc * batch_size
        total_samples += batch_size

        completed_updates += 1
        progress.update(1)

        # The same worker immediately pulls the latest global model
        # and starts its next local computation.
        if scheduled_updates < updates_per_epoch:
            schedule_worker_task(
                event_queue=event_queue,
                counter=counter,
                current_time=current_time,
                worker_id=worker_id,
                worker_delay=worker_delays[worker_id],
                worker=workers[worker_id],
                server=server,
                criterion=criterion,
            )
            scheduled_updates += 1

    progress.close()

    avg_train_loss = total_loss / max(total_samples, 1)
    avg_train_acc = total_acc / max(total_samples, 1)

    avg_staleness = float(np.mean(staleness_values)) if staleness_values else 0.0
    max_staleness = int(np.max(staleness_values)) if staleness_values else 0
    min_staleness = int(np.min(staleness_values)) if staleness_values else 0

    return {
        "train_loss": avg_train_loss,
        "train_acc": avg_train_acc,
        "virtual_time": current_time,
        "epoch_virtual_time": current_time - start_virtual_time,
        "async_updates": completed_updates,
        "avg_staleness": avg_staleness,
        "max_staleness": max_staleness,
        "min_staleness": min_staleness,
    }


def run_async_sgd(
    model_name: str = "mlp",
    data_dir: str = "./data",
    output_dir: str = "./results/raw",
    epochs: int = 10,
    batch_size: int = 64,
    test_batch_size: int = 256,
    lr: float = 0.01,
    num_workers: int = 4,
    worker_delays: str = "1,1,1,1",
    seed: int = 42,
    device: str = "auto",
    dataloader_workers: int = 0,
    pin_memory: bool = False,
    no_progress: bool = False,
) -> str:
    """
    Run Async-SGD experiment.
    """
    set_seed(seed)

    device = get_device(device)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Async-SGD Experiment")
    print("=" * 80)
    print(f"model              : {model_name}")
    print(f"epochs             : {epochs}")
    print(f"batch_size/worker  : {batch_size}")
    print(f"lr                 : {lr}")
    print(f"num_workers        : {num_workers}")
    print(f"worker_delays      : {worker_delays}")
    print(f"seed               : {seed}")
    print(f"device             : {device}")
    print("=" * 80)

    train_dataset, test_dataset = get_mnist_datasets(
        data_dir=data_dir,
        download=True,
        normalize=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
        pin_memory=pin_memory,
    )

    global_model = build_model(model_name=model_name).to(device)

    # Store model name for build_async_workers.
    global_model.model_name = model_name

    server = Server(
        model=global_model,
        lr=lr,
        device=device,
    )

    workers, worker_loaders = build_async_workers(
        base_model=global_model,
        train_dataset=train_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        seed=seed,
        device=device,
        dataloader_workers=dataloader_workers,
        pin_memory=pin_memory,
    )

    delays = parse_worker_delays(worker_delays, num_workers)

    criterion = nn.CrossEntropyLoss()

    batches_per_worker = max(len(loader) for loader in worker_loaders)

    # For Async-SGD, one epoch is defined as approximately the same amount
    # of local mini-batch computations as all workers together.
    #
    # Example:
    #   MNIST train = 60000
    #   num_workers = 4
    #   batch_size = 64
    #   batches_per_worker = ceil(15000 / 64) = 235
    #   updates_per_epoch = 235 * 4 = 940
    updates_per_epoch = batches_per_worker * num_workers

    model_size_bytes = server.model_size_bytes(bytes_per_param=4)
    comm_bytes_per_update = 2 * model_size_bytes

    print(f"train samples       : {len(train_dataset)}")
    print(f"test samples        : {len(test_dataset)}")
    print(f"batches/worker      : {batches_per_worker}")
    print(f"updates/epoch       : {updates_per_epoch}")
    print(f"model size          : {model_size_bytes / 1024:.2f} KB")
    print(f"comm/update         : {comm_bytes_per_update / 1024 / 1024:.4f} MB")
    print("=" * 80)

    records = []
    total_comm_bytes = 0
    virtual_time = 0.0
    wall_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        train_stats = train_one_epoch_async_sgd(
            server=server,
            workers=workers,
            criterion=criterion,
            updates_per_epoch=updates_per_epoch,
            worker_delays=delays,
            start_virtual_time=virtual_time,
            epoch=epoch,
            show_progress=not no_progress,
        )

        virtual_time = train_stats["virtual_time"]

        test_loss, test_acc = evaluate(
            model=server.model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        epoch_comm_bytes = updates_per_epoch * comm_bytes_per_update
        total_comm_bytes += epoch_comm_bytes

        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - wall_start_time

        record = {
            "algorithm": "async_sgd",
            "model": model_name,
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "train_acc": train_stats["train_acc"] * 100.0,
            "test_loss": test_loss,
            "test_acc": test_acc * 100.0,
            "epoch_time": epoch_time,
            "elapsed_time": elapsed_time,
            "virtual_time": virtual_time,
            "epoch_virtual_time": train_stats["epoch_virtual_time"],
            "comm_round": int(server.version),
            "async_updates": train_stats["async_updates"],
            "comm_bytes": total_comm_bytes,
            "comm_mb": total_comm_bytes / 1024 / 1024,
            "epoch_comm_bytes": epoch_comm_bytes,
            "epoch_comm_mb": epoch_comm_bytes / 1024 / 1024,
            "avg_staleness": train_stats["avg_staleness"],
            "max_staleness": train_stats["max_staleness"],
            "min_staleness": train_stats["min_staleness"],
            "num_workers": num_workers,
            "batch_size": batch_size,
            "lr": lr,
            "worker_delays": "-".join(str(int(d)) if float(d).is_integer() else str(d) for d in delays),
            "server_version": int(server.version),
        }

        records.append(record)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={record['train_loss']:.4f} | "
            f"train_acc={record['train_acc']:.2f}% | "
            f"test_loss={record['test_loss']:.4f} | "
            f"test_acc={record['test_acc']:.2f}% | "
            f"comm_round={record['comm_round']} | "
            f"comm_mb={record['comm_mb']:.2f} | "
            f"virtual_time={record['virtual_time']:.2f} | "
            f"avg_staleness={record['avg_staleness']:.2f} | "
            f"max_staleness={record['max_staleness']}"
        )

    delay_tag = "-".join(str(int(d)) if float(d).is_integer() else str(d) for d in delays)
    output_name = (
        f"async_sgd_{model_name}_workers{num_workers}_delay{delay_tag}_"
        f"epochs{epochs}_bs{batch_size}_lr{lr}_seed{seed}.csv"
    )

    output_path = os.path.join(output_dir, output_name)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Async-SGD finished.")
    print(f"final_test_acc      : {records[-1]['test_acc']:.2f}%")
    print(f"total_comm_rounds   : {records[-1]['comm_round']}")
    print(f"total_comm_mb       : {records[-1]['comm_mb']:.2f}")
    print(f"final_virtual_time  : {records[-1]['virtual_time']:.2f}")
    print(f"final_avg_staleness : {records[-1]['avg_staleness']:.2f}")
    print(f"log saved to        : {output_path}")
    print("=" * 80)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run Async-SGD on MNIST.")

    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "logistic"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results/raw")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--worker-delays",
        type=str,
        default="1,1,1,1",
        help="Comma-separated virtual delays for workers, e.g., 1,1,1,5.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-progress", action="store_true")

    args = parser.parse_args()

    run_async_sgd(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        worker_delays=args.worker_delays,
        seed=args.seed,
        device=args.device,
        dataloader_workers=args.dataloader_workers,
        pin_memory=args.pin_memory,
        no_progress=args.no_progress,
    )


if __name__ == "__main__":
    main()