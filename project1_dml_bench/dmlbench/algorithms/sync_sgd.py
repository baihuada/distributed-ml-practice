"""
sync_sgd.py

功能：
1. 实现同步数据并行 SGD，即 Sync-SGD；
2. 使用单进程模拟多个 worker；
3. 每个 worker 持有一份本地数据；
4. server 广播全局模型，worker 计算梯度，server 聚合梯度并更新；
5. 记录 loss、accuracy、通信轮数、通信量等指标。

运行示例：
python -m dmlbench.algorithms.sync_sgd \
  --model mlp \
  --epochs 10 \
  --batch-size 64 \
  --lr 0.01 \
  --num-workers 4 \
  --seed 42
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
    计算一个 batch 的预测正确数和总样本数。

    参数：
        logits:
            模型输出，形状为 [batch_size, num_classes]。
        labels:
            真实标签，形状为 [batch_size]。

    返回：
        correct:
            预测正确数量。
        total:
            batch 样本总数。
    """

    preds = logits.argmax(dim=1)
    correct = int((preds == labels).sum().item())
    total = int(labels.size(0))

    return correct, total


def parse_worker_delays(worker_delays: str, num_workers: int) -> List[float]:
    """
    解析 worker 虚拟计算延迟。

    示例：
        "1,1,1,1" -> [1.0, 1.0, 1.0, 1.0]
        "1,1,1,5" -> [1.0, 1.0, 1.0, 5.0]

    这些 delay 不会让程序真实 sleep，只用于计算虚拟时间 virtual_time。
    """

    if worker_delays is None or worker_delays.strip() == "":
        delays = [1.0 for _ in range(num_workers)]
    else:
        delays = [float(x.strip()) for x in worker_delays.split(",") if x.strip() != ""]

    if len(delays) != num_workers:
        raise ValueError(
            f"worker_delays length must equal num_workers. "
            f"Got {len(delays)} delays, but num_workers={num_workers}."
        )

    if any(delay <= 0 for delay in delays):
        raise ValueError("All worker delays must be positive.")

    return delays


def format_delay_tag(delays: List[float]) -> str:
    """
    把 delay 列表转成文件名中使用的标签。
    """

    return "-".join(
        str(int(delay)) if float(delay).is_integer() else str(delay)
        for delay in delays
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    在测试集上评估 server 的全局模型。

    注意：
        Sync-SGD 训练过程中只有 server 的模型是全局模型；
        因此测试时应该评估 server.model，而不是某个 worker.model。

    参数：
        model:
            server 当前全局模型。
        test_loader:
            测试集 DataLoader。
        criterion:
            损失函数。
        device:
            评估设备。

    返回：
        {
            "test_loss": float,
            "test_acc": float
        }
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


def build_sync_workers(
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
    构造 Sync-SGD 使用的多个 worker。

    步骤：
        1. 对训练集做 IID 划分；
        2. 为每个 worker 构造本地 DataLoader；
        3. 为每个 worker 创建 Worker 对象。

    参数：
        base_model:
            基础模型。每个 worker 会 deepcopy 一份。
        train_dataset:
            完整 MNIST 训练集。
        num_workers:
            分布式模拟中的 worker 数量。
        batch_size:
            每个 worker 的本地 batch size。
        seed:
            随机种子。
        device:
            训练设备。
        dataloader_workers:
            PyTorch DataLoader 的 num_workers。
            注意：它和分布式 worker 数量不是一回事。
        pin_memory:
            GPU 训练时可设为 True。

    返回：
        workers:
            Worker 对象列表。
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


def train_one_epoch_sync_sgd(
    server: Server,
    workers: List[Worker],
    criterion: nn.Module,
    steps_per_epoch: int,
    epoch: int,
    worker_delays: List[float],
    start_virtual_time: float = 0.0,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    训练一个 epoch 的 Sync-SGD。

    每一个 step 的完整同步流程：

        1. server.broadcast_model()
           server 广播当前全局模型 w_t。

        2. worker.compute_gradient()
           每个 worker 基于同一个 w_t 计算本地梯度 g_k(w_t)。

        3. server.aggregate_gradients()
           server 对所有 worker 梯度进行加权平均。

        4. server.apply_gradients()
           server 更新全局模型：
               w_{t+1} = w_t - lr * avg_grad

    参数：
        server:
            中心 server。
        workers:
            worker 列表。
        criterion:
            损失函数。
        steps_per_epoch:
            一个 epoch 内执行多少个同步 step。
        epoch:
            当前 epoch 编号。
        show_progress:
            是否显示进度条。

    返回：
        {
            "train_loss": float,
            "train_acc": float,
            "comm_rounds": int,
            "comm_bytes": int
        }
    """

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    num_workers = len(workers)
    if len(worker_delays) != num_workers:
        raise ValueError("worker_delays length must equal number of workers.")

    comm_rounds = 0
    comm_bytes = 0

    # Sync-SGD 有同步屏障，每一轮必须等待最慢 worker 完成。
    # 因此一轮虚拟耗时为 max(worker_delays)。
    max_worker_delay = max(worker_delays)
    virtual_time = start_virtual_time

    iterator = tqdm(
        range(steps_per_epoch),
        desc=f"Sync-SGD Epoch {epoch}",
        leave=False,
        disable=not show_progress,
    )

    for _ in iterator:
        # ------------------------------------------------------------
        # Step 1: server 广播当前全局模型。
        # ------------------------------------------------------------
        global_state = server.broadcast_model()

        # ------------------------------------------------------------
        # Step 2: 所有 worker 基于同一个 global_state 计算梯度。
        # 这是 Sync-SGD 的核心：所有梯度都来自同一个 w_t。
        # ------------------------------------------------------------
        gradient_packages = []

        for worker in workers:
            grad_pkg = worker.compute_gradient(
                global_state=global_state,
                criterion=criterion,
            )
            gradient_packages.append(grad_pkg)

        # ------------------------------------------------------------
        # Step 3: server 聚合所有 worker 的梯度。
        # 默认按照每个 worker 当前 batch 样本数加权。
        # ------------------------------------------------------------
        avg_gradients = server.aggregate_gradients(gradient_packages)

        # ------------------------------------------------------------
        # Step 4: server 用平均梯度更新全局模型。
        # ------------------------------------------------------------
        server.apply_gradients(avg_gradients)

        # ------------------------------------------------------------
        # Step 5: 统计当前同步 step 的训练指标。
        # ------------------------------------------------------------
        step_loss_sum = 0.0
        step_correct = 0
        step_samples = 0

        for pkg in gradient_packages:
            batch_samples = pkg["total"]
            step_loss_sum += pkg["loss"] * batch_samples
            step_correct += pkg["correct"]
            step_samples += batch_samples

        total_loss += step_loss_sum
        total_correct += step_correct
        total_samples += step_samples

        # ------------------------------------------------------------
        # Step 6: 统计通信量。
        # 一轮 Sync-SGD 通信包括：
        #   server -> workers: 广播模型
        #   workers -> server: 上传梯度
        # ------------------------------------------------------------
        comm_rounds += 1
        comm_bytes += server.estimate_sync_sgd_round_bytes(num_workers=num_workers)

        # 同步训练每一轮都被最慢 worker 决定。
        virtual_time += max_worker_delay

        avg_loss_so_far = total_loss / total_samples
        avg_acc_so_far = 100.0 * total_correct / total_samples

        iterator.set_postfix({
            "loss": avg_loss_so_far,
            "acc": avg_acc_so_far,
            "round": comm_rounds,
        })

    train_loss = total_loss / total_samples
    train_acc = 100.0 * total_correct / total_samples

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "comm_rounds": comm_rounds,
        "comm_bytes": comm_bytes,
        "virtual_time": virtual_time,
        "epoch_virtual_time": virtual_time - start_virtual_time,
        "max_worker_delay": max_worker_delay,
    }


def run_sync_sgd(
    model_name: str = "mlp",
    data_dir: str = "./data",
    output_dir: str = "./results/raw",
    epochs: int = 10,
    batch_size: int = 64,
    test_batch_size: int = 256,
    lr: float = 0.01,
    seed: int = 42,
    num_workers: int = 4,
    worker_delays: str = "1,1,1,1",
    device: str = "auto",
    dataloader_workers: int = 0,
    show_progress: bool = True,
) -> List[Dict[str, float]]:
    """
    运行 Sync-SGD 实验。

    参数：
        model_name:
            "logistic" 或 "mlp"。
        data_dir:
            MNIST 数据目录。
        output_dir:
            CSV 日志保存目录。
        epochs:
            训练 epoch 数。
        batch_size:
            每个 worker 的本地 batch size。
            注意：总 batch size = batch_size * num_workers。
        test_batch_size:
            测试 batch size。
        lr:
            server 更新学习率。
        seed:
            随机种子。
        num_workers:
            模拟的分布式 worker 数量。
        device:
            "auto"、"cpu" 或 "cuda"。
        dataloader_workers:
            DataLoader 的 num_workers。
            注意不是分布式 worker 数量。
        show_progress:
            是否显示 tqdm 进度条。

    返回：
        每个 epoch 的日志记录列表。
    """

    if num_workers <= 0:
        raise ValueError("num_workers must be positive.")

    delays = parse_worker_delays(worker_delays, num_workers)
    delay_tag = format_delay_tag(delays)

    # ------------------------------------------------------------
    # 1. 固定随机种子。
    # ------------------------------------------------------------
    set_seed(seed)

    # ------------------------------------------------------------
    # 2. 设置设备。
    # ------------------------------------------------------------
    torch_device = get_device(device)
    print(f"Using device: {torch_device}")

    # ------------------------------------------------------------
    # 3. 加载完整 MNIST 数据集。
    # 训练集用于切分给 worker，测试集用于评估全局模型。
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 4. 创建基础模型。
    # base_model 用于初始化 server 和所有 worker。
    # ------------------------------------------------------------
    base_model = build_model(model_name=model_name)

    # ------------------------------------------------------------
    # 5. 创建 server。
    # server 持有唯一的全局模型。
    # ------------------------------------------------------------
    server = Server(
        model=base_model,
        lr=lr,
        device=torch_device,
    )

    # ------------------------------------------------------------
    # 6. 创建 workers。
    # 每个 worker 有自己的数据分片和模型副本。
    # ------------------------------------------------------------
    workers = build_sync_workers(
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

    # ------------------------------------------------------------
    # 7. 定义损失函数。
    # Sync-SGD 里 worker 只计算梯度，server 用聚合梯度更新。
    # 所以不需要给 server 定义 optimizer。
    # ------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------
    # 8. 确定每个 epoch 有多少个同步 step。
    #
    # 每个 worker 数据量基本一致。
    # 这里取最短 worker_loader 的 batch 数，避免某些 worker 额外循环。
    # ------------------------------------------------------------
    steps_per_epoch = min(len(worker.train_loader) for worker in workers)
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Worker delays: {delays}")
    print(f"Sync virtual time per step: {max(delays)}")

    # ------------------------------------------------------------
    # 9. 创建日志器。
    # ------------------------------------------------------------
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_name = (
        f"sync_sgd_{model_name}"
        f"_workers{num_workers}"
        f"_delay{delay_tag}"
        f"_epochs{epochs}"
        f"_bs{batch_size}"
        f"_lr{lr}"
        f"_seed{seed}.csv"
    )

    logger = ExperimentLogger(
        save_path=str(output_path / csv_name),
        auto_save=True,
    )

    # ------------------------------------------------------------
    # 10. 主训练循环。
    # ------------------------------------------------------------
    records: List[Dict[str, float]] = []

    total_comm_rounds = 0
    total_comm_bytes = 0
    virtual_time = 0.0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch_sync_sgd(
            server=server,
            workers=workers,
            criterion=criterion,
            steps_per_epoch=steps_per_epoch,
            epoch=epoch,
            worker_delays=delays,
            start_virtual_time=virtual_time,
            show_progress=show_progress,
        )

        virtual_time = float(train_metrics["virtual_time"])

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

        record = {
            "algorithm": "sync_sgd",
            "model": model_name,
            "epoch": epoch,
            "num_workers": num_workers,
            "batch_size_per_worker": batch_size,
            "effective_batch_size": batch_size * num_workers,
            "steps_per_epoch": steps_per_epoch,
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
            "worker_delays": delay_tag,
            "max_worker_delay": train_metrics["max_worker_delay"],
            "epoch_virtual_time": train_metrics["epoch_virtual_time"],
            "virtual_time": virtual_time,
        }

        logger.log(record)
        records.append(record)

        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"train_loss={record['train_loss']:.4f} "
            f"train_acc={record['train_acc']:.2f}% "
            f"test_loss={record['test_loss']:.4f} "
            f"test_acc={record['test_acc']:.2f}% "
            f"comm_round={record['comm_round']} "
            f"comm_mb={record['comm_mb']:.2f} "
            f"virtual_time={record['virtual_time']:.2f} "
            f"time={record['epoch_time']:.2f}s"
        )

    final_record = records[-1]

    print("=" * 80)
    print("Sync-SGD Finished")
    print(f"Final Test Accuracy: {final_record['test_acc']:.2f}%")
    print(f"Total Communication Rounds: {final_record['comm_round']}")
    print(f"Total Communication: {final_record['comm_mb']:.2f} MB")
    print(f"Final Virtual Time: {final_record['virtual_time']:.2f}")
    print(f"Server Version: {server.version}")
    print(f"Log saved to: {output_path / csv_name}")
    print("=" * 80)

    return records


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """

    parser = argparse.ArgumentParser(description="Run Sync-SGD on MNIST.")

    parser.add_argument("--model", type=str, default="mlp", choices=["logistic", "mlp"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results/raw")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--worker-delays",
        type=str,
        default="1,1,1,1",
        help="Comma-separated virtual delays for workers, e.g., 1,1,1,5.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_sync_sgd(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        worker_delays=args.worker_delays,
        device=args.device,
        dataloader_workers=args.dataloader_workers,
        show_progress=not args.no_progress,
    )