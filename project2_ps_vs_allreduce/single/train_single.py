"""
单进程训练入口。

运行示例：
python -m single.train_single --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42

本文件实现 Single Process baseline。
后续 PS 和 DDP 的实验结果都需要和这个 baseline 对比。

1. 解析命令行参数；
2. 固定随机种子；
3. 加载 MNIST / Fashion-MNIST；
4. 构造 MLP / Logistic Regression；
5. 执行普通单进程训练；
6. 每个 epoch 测试一次；
7. 统计 epoch_time、elapsed_time、samples/s；
8. 统计模型参数量和模型大小；
9. 保存 CSV 日志到 results/raw/。

"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from common.datasets import get_dataloaders, print_dataset_info
from common.logger import CSVLogger, build_log_filename, print_epoch_log, save_config
from common.metrics import compute_accuracy, compute_samples_per_sec, count_correct, evaluate
from common.models import build_model
from common.seed import set_seed
from utils.comm import count_parameters, estimate_comm_per_epoch, estimate_model_size


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    返回
    ----
    argparse.Namespace
        命令行参数对象。
    """

    parser = argparse.ArgumentParser(
        description="Single Process baseline for PS vs AllReduce project"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist", "fashion-mnist", "fashion"],
        help="数据集名称",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "logistic", "lr", "logreg"],
        help="模型名称",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练 epoch 数",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="训练 batch size",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        help="测试 batch size",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="学习率",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="SGD 动量系数，第一版默认不使用动量",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="权重衰减系数",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="MLP 隐藏层维度",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="MLP dropout 概率，第一版默认不使用 dropout",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="数据集保存目录",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/raw",
        help="CSV 日志输出目录",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader 数据加载进程数。Windows 下建议第一版设为 0",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="训练设备",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="是否启用更严格的确定性设置",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="是否关闭 tqdm 进度条",
    )

    parser.add_argument(
        "--print-data-info",
        action="store_true",
        help="是否打印数据集基本信息",
    )

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """
    根据命令行参数选择训练设备。

    参数
    ----
    device_arg : str
        auto、cpu 或 cuda。

    返回
    ----
    torch.device
        实际使用的设备。
    """

    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境不可用 CUDA")
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    训练一个 epoch。

    参数
    ----
    model : nn.Module
        待训练模型。
    train_loader : DataLoader
        训练集 DataLoader。
    criterion : nn.Module
        损失函数。
    optimizer : optim.Optimizer
        优化器。
    device : torch.device
        训练设备。
    epoch : int
        当前 epoch 编号。
    total_epochs : int
        总 epoch 数。
    show_progress : bool
        是否显示 tqdm 进度条。

    返回
    ----
    Dict[str, float]
        当前 epoch 的训练 loss、accuracy 和样本数。
    """

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    iterator = train_loader
    if show_progress:
        iterator = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch}/{total_epochs}",
            leave=False,
        )

    for inputs, targets in iterator:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)

        # CrossEntropyLoss 默认是 batch 平均 loss，因此乘以 batch_size 后累计
        total_loss += loss.item() * batch_size
        total_correct += count_correct(logits, targets)
        total_samples += batch_size

    train_loss = total_loss / total_samples
    train_acc = compute_accuracy(total_correct, total_samples)

    return {
        "loss": train_loss,
        "accuracy": train_acc,
        "num_samples": float(total_samples),
    }


def build_optimizer(
    model: nn.Module,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> optim.Optimizer:
    """
    构造优化器。

    第一版使用 SGD，便于和后续 PS-SGD / DDP-SGD 对齐。

    参数
    ----
    model : nn.Module
        待训练模型。
    lr : float
        学习率。
    momentum : float
        动量系数。
    weight_decay : float
        权重衰减系数。

    返回
    ----
    optim.Optimizer
        PyTorch 优化器。
    """

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    return optimizer


def main() -> None:
    """
    单进程训练主函数。
    """

    args = parse_args()

    if args.epochs <= 0:
        raise ValueError(f"epochs 必须为正整数，但得到 {args.epochs}")

    if args.batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，但得到 {args.batch_size}")

    if args.lr <= 0:
        raise ValueError(f"lr 必须为正数，但得到 {args.lr}")

    set_seed(args.seed, deterministic=args.deterministic)

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"

    print("=" * 80)
    print("Single Process Baseline")
    print("=" * 80)
    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model}")
    print(f"训练轮数: {args.epochs}")
    print(f"batch size: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"随机种子: {args.seed}")
    print(f"训练设备: {device}")
    print("=" * 80)

    train_loader, test_loader = get_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
        download=True,
    )

    if args.print_data_info:
        print_dataset_info(train_loader, test_loader)

    model = build_model(
        model_name=args.model,
        dataset_name=args.dataset,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model=model,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    model_params = count_parameters(model)
    model_size_mb = estimate_model_size(model, unit="MB")

    print(f"模型参数量: {model_params}")
    print(f"模型大小估算: {model_size_mb:.4f} MB")
    print("=" * 80)

    log_filename = build_log_filename(
        system="single",
        dataset=args.dataset,
        model=args.model,
        seed=args.seed,
        num_workers=1,
    )

    log_path = Path(args.output_dir) / log_filename
    logger = CSVLogger(log_path=log_path)

    config_path = Path(args.output_dir) / log_filename.replace(".csv", "_config.json")
    save_config(vars(args), config_path)

    total_start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            show_progress=not args.no_progress,
        )

        test_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time = time.perf_counter() - epoch_start_time
        elapsed_time = time.perf_counter() - total_start_time

        num_train_samples = int(train_metrics["num_samples"])
        num_test_samples = int(test_metrics["num_samples"])

        samples_per_sec = compute_samples_per_sec(
            num_samples=num_train_samples,
            elapsed_time=epoch_time,
        )

        # Single Process 没有跨进程通信，因此通信量估算为 0
        comm_mb = estimate_comm_per_epoch(
            model=model,
            system="single",
            num_workers=1,
            num_steps=len(train_loader),
            unit="MB",
        )

        record = {
            "system": "single",
            "dataset": args.dataset,
            "model": args.model,
            "num_workers": 1,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["accuracy"],
            "epoch_time": epoch_time,
            "elapsed_time": elapsed_time,
            "samples_per_sec": samples_per_sec,
            "num_train_samples": num_train_samples,
            "num_test_samples": num_test_samples,
            "model_params": model_params,
            "model_size_mb": model_size_mb,
            "comm_mb": comm_mb,
            "seed": args.seed,
            "lr": args.lr,
            "batch_size": args.batch_size,
        }

        logger.log(record)
        print_epoch_log(record)

    logger.save()

    final_record = logger.latest()

    print("=" * 80)
    print("训练完成")
    print(f"CSV 日志保存位置: {log_path}")
    print(f"配置文件保存位置: {config_path}")
    print(f"最终测试准确率: {float(final_record['test_acc']) * 100:.2f}%")
    print(f"总耗时: {float(final_record['elapsed_time']):.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()