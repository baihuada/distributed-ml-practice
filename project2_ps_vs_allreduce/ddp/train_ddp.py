"""
PyTorch DDP / AllReduce 训练入口。

运行示例：
torchrun --standalone --nproc_per_node=2 ddp/train_ddp.py --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42

Windows CPU 第一版推荐：
torchrun --standalone --nproc_per_node=2 ddp/train_ddp.py --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42 --device cpu --backend gloo

本文件实现 DDP / AllReduce 训练：
1. 每个进程称为一个 rank；
2. 每个 rank 持有完整模型副本；
3. 每个 rank 读取不同训练数据分片；
4. DDP 在 loss.backward() 后自动执行梯度 AllReduce；
5. rank 0 负责保存日志和输出最终结果。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from common.datasets import get_dataset, get_dataloaders
from common.logger import CSVLogger, build_log_filename, print_epoch_log, save_config
from common.metrics import compute_accuracy, compute_samples_per_sec, count_correct, evaluate
from common.models import build_model
from common.seed import seed_worker, set_seed
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
        description="PyTorch DDP / AllReduce training for PS vs AllReduce project"
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
        help="每个 rank 的本地 batch size",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        help="测试 batch size。测试只在 rank 0 执行",
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
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="每个 rank 内部 DataLoader 的数据加载进程数。Windows 下建议保持 0",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="训练设备。Windows 第一版建议使用 cpu",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        choices=["gloo", "nccl"],
        help="分布式后端。Windows / CPU 使用 gloo；Linux GPU 可使用 nccl",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="是否关闭 tqdm 进度条",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="是否启用更严格的确定性设置",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """
    检查命令行参数是否合法。

    参数
    ----
    args : argparse.Namespace
        命令行参数对象。
    """

    if args.epochs <= 0:
        raise ValueError(f"epochs 必须为正整数，但得到 {args.epochs}")

    if args.batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，但得到 {args.batch_size}")

    if args.test_batch_size <= 0:
        raise ValueError(
            f"test_batch_size 必须为正整数，但得到 {args.test_batch_size}"
        )

    if args.lr <= 0:
        raise ValueError(f"lr 必须为正数，但得到 {args.lr}")

    if args.dataloader_num_workers < 0:
        raise ValueError(
            "dataloader_num_workers 不能为负数，"
            f"但得到 {args.dataloader_num_workers}"
        )

    if args.backend == "nccl" and args.device == "cpu":
        raise ValueError("backend=nccl 不能和 device=cpu 同时使用")


def init_distributed(backend: str) -> Tuple[int, int, int]:
    """
    初始化 PyTorch 分布式进程组。

    参数
    ----
    backend : str
        分布式通信后端，例如 gloo 或 nccl。

    返回
    ----
    Tuple[int, int, int]
        rank、local_rank、world_size。

    说明
    ----
    使用 torchrun 启动时，环境变量中会自动包含：
    RANK
    LOCAL_RANK
    WORLD_SIZE
    MASTER_ADDR
    MASTER_PORT
    """

    if "RANK" not in os.environ:
        raise RuntimeError(
            "没有检测到 RANK 环境变量。请使用 torchrun 启动 DDP，例如："
            "torchrun --standalone --nproc_per_node=2 ddp/train_ddp.py ..."
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend=backend,
        init_method="env://",
    )

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """
    销毁分布式进程组。
    """

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    """
    判断当前进程是否为 rank 0。

    参数
    ----
    rank : int
        当前进程编号。

    返回
    ----
    bool
        是否为 rank 0。
    """

    return rank == 0


def resolve_device(
    device_arg: str,
    local_rank: int,
) -> torch.device:
    """
    根据参数和 local_rank 选择训练设备。

    参数
    ----
    device_arg : str
        cpu、cuda 或 auto。
    local_rank : int
        当前节点内的 rank 编号。

    返回
    ----
    torch.device
        当前 rank 使用的设备。

    说明
    ----
    第一版建议 CPU + gloo。
    后续如果在 Linux + GPU 环境下，可以使用 cuda + nccl。
    """

    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境不可用 CUDA")
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")

    if device_arg == "auto":
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")

    raise ValueError(f"不支持的 device: {device_arg}")


def create_train_loader(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    dataloader_num_workers: int,
    seed: int,
    rank: int,
    world_size: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DistributedSampler]:
    """
    构造 DDP 训练集 DataLoader。

    参数
    ----
    dataset_name : str
        数据集名称。
    data_dir : str
        数据集目录。
    batch_size : int
        每个 rank 的本地 batch size。
    dataloader_num_workers : int
        DataLoader 数据加载进程数。
    seed : int
        随机种子。
    rank : int
        当前进程编号。
    world_size : int
        总进程数。
    pin_memory : bool
        是否启用 pin_memory。

    返回
    ----
    Tuple[DataLoader, DistributedSampler]
        训练集 DataLoader 和 DistributedSampler。

    关键点
    ----
    DistributedSampler 会保证不同 rank 读取不同数据分片。
    """

    train_dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train=True,
        download=True,
    )

    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )

    generator = torch.Generator()
    generator.manual_seed(seed + rank)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=dataloader_num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if dataloader_num_workers > 0 else None,
        generator=generator,
    )

    return train_loader, train_sampler


def create_test_loader_rank0(
    dataset_name: str,
    data_dir: str,
    test_batch_size: int,
    dataloader_num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    """
    只为 rank 0 构造测试集 DataLoader。

    参数
    ----
    dataset_name : str
        数据集名称。
    data_dir : str
        数据集目录。
    test_batch_size : int
        测试 batch size。
    dataloader_num_workers : int
        DataLoader 数据加载进程数。
    seed : int
        随机种子。
    pin_memory : bool
        是否启用 pin_memory。

    返回
    ----
    DataLoader
        测试集 DataLoader。
    """

    _, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=64,
        test_batch_size=test_batch_size,
        num_workers=dataloader_num_workers,
        seed=seed,
        pin_memory=pin_memory,
        download=True,
    )

    return test_loader


def build_optimizer(
    model: nn.Module,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> optim.Optimizer:
    """
    构造优化器。

    参数
    ----
    model : nn.Module
        DDP 包装后的模型。
    lr : float
        学习率。
    momentum : float
        SGD 动量系数。
    weight_decay : float
        权重衰减系数。

    返回
    ----
    optim.Optimizer
        PyTorch 优化器。
    """

    return optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def reduce_sum_scalar(value: float, device: torch.device) -> float:
    """
    对所有 rank 的标量值做求和 all_reduce。

    参数
    ----
    value : float
        当前 rank 的标量值。
    device : torch.device
        当前 rank 的设备。

    返回
    ----
    float
        所有 rank 求和后的结果。
    """

    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def train_one_ddp_epoch(
    model: DDP,
    train_loader: DataLoader,
    train_sampler: DistributedSampler,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    rank: int,
    show_progress: bool,
) -> Dict[str, float]:
    """
    训练一个 DDP epoch。

    参数
    ----
    model : DDP
        DDP 包装后的模型。
    train_loader : DataLoader
        当前 rank 的训练 DataLoader。
    train_sampler : DistributedSampler
        当前 rank 的分布式采样器。
    criterion : nn.Module
        损失函数。
    optimizer : optim.Optimizer
        优化器。
    device : torch.device
        当前 rank 使用的设备。
    epoch : int
        当前 epoch。
    total_epochs : int
        总 epoch 数。
    rank : int
        当前进程编号。
    show_progress : bool
        是否显示 tqdm 进度条。

    返回
    ----
    Dict[str, float]
        聚合后的全局 train_loss、train_acc、num_samples。
    """

    model.train()

    # DistributedSampler 每个 epoch 需要设置 epoch，否则每轮 shuffle 顺序可能相同
    train_sampler.set_epoch(epoch)

    local_loss_sum = 0.0
    local_correct = 0
    local_samples = 0

    iterator = train_loader

    if show_progress and rank == 0:
        iterator = tqdm(
            train_loader,
            desc=f"DDP Train Epoch {epoch}/{total_epochs}",
            leave=False,
        )

    for inputs, targets in iterator:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        # 关键点：
        # loss.backward() 后，DDP 会自动触发梯度 AllReduce。
        # 用户不需要手写梯度平均。
        loss.backward()

        optimizer.step()

        batch_size = targets.size(0)

        local_loss_sum += loss.item() * batch_size
        local_correct += count_correct(logits, targets)
        local_samples += batch_size

    # 汇总所有 rank 的训练统计量
    global_loss_sum = reduce_sum_scalar(local_loss_sum, device=device)
    global_correct = reduce_sum_scalar(local_correct, device=device)
    global_samples = reduce_sum_scalar(local_samples, device=device)

    train_loss = global_loss_sum / global_samples
    train_acc = compute_accuracy(
        total_correct=int(global_correct),
        num_samples=int(global_samples),
    )

    return {
        "loss": float(train_loss),
        "accuracy": float(train_acc),
        "num_samples": float(global_samples),
    }


def build_ddp_model(
    model: nn.Module,
    device: torch.device,
) -> DDP:
    """
    将普通模型包装为 DistributedDataParallel。

    参数
    ----
    model : nn.Module
        普通 PyTorch 模型。
    device : torch.device
        当前 rank 的设备。

    返回
    ----
    DDP
        DDP 包装后的模型。
    """

    model = model.to(device)

    if device.type == "cuda":
        ddp_model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
        )
    else:
        ddp_model = DDP(model)

    return ddp_model


def main() -> None:
    """
    DDP 训练主函数。
    """

    args = parse_args()
    validate_args(args)

    rank, local_rank, world_size = init_distributed(args.backend)

    try:
        set_seed(args.seed + rank, deterministic=args.deterministic)

        device = resolve_device(
            device_arg=args.device,
            local_rank=local_rank,
        )

        pin_memory = device.type == "cuda"

        output_dir = Path(args.output_dir)
        if is_rank0(rank):
            output_dir.mkdir(parents=True, exist_ok=True)

        # 等待 rank 0 创建目录
        dist.barrier()

        train_loader, train_sampler = create_train_loader(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            dataloader_num_workers=args.dataloader_num_workers,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            pin_memory=pin_memory,
        )

        test_loader = None
        if is_rank0(rank):
            test_loader = create_test_loader_rank0(
                dataset_name=args.dataset,
                data_dir=args.data_dir,
                test_batch_size=args.test_batch_size,
                dataloader_num_workers=args.dataloader_num_workers,
                seed=args.seed,
                pin_memory=pin_memory,
            )

        base_model = build_model(
            model_name=args.model,
            dataset_name=args.dataset,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )

        ddp_model = build_ddp_model(
            model=base_model,
            device=device,
        )

        criterion = nn.CrossEntropyLoss()

        optimizer = build_optimizer(
            model=ddp_model,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        # DDP 包装后，真实模型在 ddp_model.module 中
        model_params = count_parameters(ddp_model.module)
        model_size_mb = estimate_model_size(ddp_model.module, unit="MB")

        logger = None
        log_path = None
        config_path = None

        if is_rank0(rank):
            print("=" * 80)
            print("DDP / AllReduce Training")
            print("=" * 80)
            print(f"数据集: {args.dataset}")
            print(f"模型: {args.model}")
            print(f"world_size: {world_size}")
            print(f"rank 0 local_rank: {local_rank}")
            print(f"训练轮数: {args.epochs}")
            print(f"每个 rank 的 batch size: {args.batch_size}")
            print(f"全局有效 batch size: {args.batch_size * world_size}")
            print(f"学习率: {args.lr}")
            print(f"随机种子: {args.seed}")
            print(f"后端: {args.backend}")
            print(f"设备: {device}")
            print(f"每个 rank 的 step 数: {len(train_loader)}")
            print(f"模型参数量: {model_params}")
            print(f"模型大小估算: {model_size_mb:.4f} MB")
            print("=" * 80)

            log_filename = build_log_filename(
                system="ddp",
                dataset=args.dataset,
                model=args.model,
                seed=args.seed,
                num_workers=world_size,
            )

            log_path = output_dir / log_filename
            config_path = output_dir / log_filename.replace(".csv", "_config.json")

            logger = CSVLogger(log_path=log_path)

            save_config(
                {
                    "dataset": args.dataset,
                    "model": args.model,
                    "epochs": args.epochs,
                    "batch_size_per_rank": args.batch_size,
                    "global_batch_size": args.batch_size * world_size,
                    "test_batch_size": args.test_batch_size,
                    "lr": args.lr,
                    "momentum": args.momentum,
                    "weight_decay": args.weight_decay,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "seed": args.seed,
                    "data_dir": args.data_dir,
                    "output_dir": args.output_dir,
                    "dataloader_num_workers": args.dataloader_num_workers,
                    "device": args.device,
                    "backend": args.backend,
                    "world_size": world_size,
                },
                config_path,
            )

        dist.barrier()

        total_start_time = time.perf_counter()

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.perf_counter()

            train_metrics = train_one_ddp_epoch(
                model=ddp_model,
                train_loader=train_loader,
                train_sampler=train_sampler,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                total_epochs=args.epochs,
                rank=rank,
                show_progress=not args.no_progress,
            )

            # 等待所有 rank 完成本 epoch 训练
            dist.barrier()

            test_metrics = {
                "loss": 0.0,
                "accuracy": 0.0,
                "num_samples": 0.0,
            }

            # 测试只在 rank 0 上做，避免重复保存和重复打印
            if is_rank0(rank):
                assert test_loader is not None

                test_metrics = evaluate(
                    model=ddp_model.module,
                    data_loader=test_loader,
                    criterion=criterion,
                    device=device,
                )

            # 等待 rank 0 完成测试
            dist.barrier()

            epoch_time = time.perf_counter() - epoch_start_time
            elapsed_time = time.perf_counter() - total_start_time

            num_train_samples = int(train_metrics["num_samples"])

            # 所有 rank 都计算 epoch_time，但只由 rank 0 保存
            if is_rank0(rank):
                num_test_samples = int(test_metrics["num_samples"])

                samples_per_sec = compute_samples_per_sec(
                    num_samples=num_train_samples,
                    elapsed_time=epoch_time,
                )

                comm_mb = estimate_comm_per_epoch(
                    model=ddp_model.module,
                    system="ddp",
                    num_workers=world_size,
                    num_steps=len(train_loader),
                    unit="MB",
                )

                record = {
                    "system": "ddp",
                    "dataset": args.dataset,
                    "model": args.model,
                    "num_workers": world_size,
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

                assert logger is not None
                logger.log(record)
                print_epoch_log(record)

            dist.barrier()

        if is_rank0(rank):
            assert logger is not None
            assert log_path is not None
            assert config_path is not None

            logger.save()
            final_record = logger.latest()

            print("=" * 80)
            print("DDP 训练完成")
            print(f"CSV 日志保存位置: {log_path}")
            print(f"配置文件保存位置: {config_path}")
            print(f"最终测试准确率: {float(final_record['test_acc']) * 100:.2f}%")
            print(f"总耗时: {float(final_record['elapsed_time']):.2f}s")
            print("=" * 80)

        dist.barrier()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()