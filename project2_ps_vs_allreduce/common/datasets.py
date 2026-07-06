"""
数据集加载模块。

本文件负责：
1. 加载 MNIST / Fashion-MNIST；
2. 构造训练集和测试集 DataLoader；
3. 为后续 Parameter Server 提供 worker 数据划分接口。

第一阶段只需要使用 get_dataloaders() 跑通单进程训练。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from common.seed import seed_worker


def get_dataset(
    dataset_name: str,
    data_dir: str = "./data",
    train: bool = True,
    download: bool = True,
) -> Dataset:
    """
    加载指定数据集。

    参数
    ----
    dataset_name : str
        数据集名称，支持 "mnist" 和 "fashion_mnist"。
    data_dir : str
        数据集保存目录。
    train : bool
        True 表示加载训练集，False 表示加载测试集。
    download : bool
        若本地不存在数据集，是否自动下载。

    返回
    ----
    Dataset
        PyTorch Dataset 对象。
    """

    name = dataset_name.lower()

    if name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        return datasets.MNIST(
            root=data_dir,
            train=train,
            transform=transform,
            download=download,
        )

    if name in ["fashion_mnist", "fashion-mnist", "fashion"]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )

        return datasets.FashionMNIST(
            root=data_dir,
            train=train,
            transform=transform,
            download=download,
        )

    raise ValueError(
        f"不支持的数据集: {dataset_name}. 当前只支持 mnist 和 fashion_mnist。"
    )


def get_dataloaders(
    dataset_name: str = "mnist",
    data_dir: str = "./data",
    batch_size: int = 64,
    test_batch_size: int = 1000,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    构造训练集和测试集 DataLoader。

    参数
    ----
    dataset_name : str
        数据集名称。
    data_dir : str
        数据集保存目录。
    batch_size : int
        训练集 batch size。
    test_batch_size : int
        测试集 batch size。
    num_workers : int
        DataLoader 使用的数据加载进程数。
        Windows 下第一版建议设为 0，稳定性更好。
    seed : int
        用于控制 shuffle 的随机种子。
    pin_memory : bool
        如果使用 GPU，可以设为 True。
    download : bool
        是否自动下载数据集。

    返回
    ----
    train_loader : DataLoader
        训练集 DataLoader。
    test_loader : DataLoader
        测试集 DataLoader。
    """

    train_dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train=True,
        download=download,
    )

    test_dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train=False,
        download=download,
    )

    # 使用独立 generator 控制 DataLoader shuffle 的随机性
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    return train_loader, test_loader


def split_dataset_by_workers(
    dataset: Dataset,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 42,
) -> List[Subset]:
    """
    将训练集切分为多个 worker 子集。

    这个函数主要为后续 Parameter Server 阶段准备。
    第一阶段单进程训练暂时不会直接使用它。

    参数
    ----
    dataset : Dataset
        原始训练集。
    num_workers : int
        worker 数量。
    shuffle : bool
        是否在切分前随机打乱样本索引。
    seed : int
        随机种子。

    返回
    ----
    List[Subset]
        每个 worker 对应一个 Dataset 子集。
    """

    if num_workers <= 0:
        raise ValueError(f"num_workers 必须为正整数，但得到 {num_workers}")

    dataset_size = len(dataset)
    indices = torch.arange(dataset_size)

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        indices = indices[torch.randperm(dataset_size, generator=generator)]

    # 尽量平均切分数据。如果不能整除，前几个 worker 会多 1 个样本。
    base_size = dataset_size // num_workers
    remainder = dataset_size % num_workers

    subsets = []
    start = 0

    for worker_id in range(num_workers):
        current_size = base_size + (1 if worker_id < remainder else 0)
        end = start + current_size
        worker_indices = indices[start:end].tolist()
        subsets.append(Subset(dataset, worker_indices))
        start = end

    return subsets


def build_worker_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
    shuffle: bool = True,
) -> DataLoader:
    """
    为单个 worker 的 Dataset 子集构造 DataLoader。

    后续 Parameter Server 阶段中，每个 worker 会有自己的数据子集。
    这个函数用于把某个子集封装成 DataLoader。

    参数
    ----
    dataset : Dataset
        某个 worker 对应的数据子集。
    batch_size : int
        batch size。
    num_workers : int
        DataLoader 的数据加载进程数。
    seed : int
        随机种子。
    pin_memory : bool
        是否使用 pin_memory。
    shuffle : bool
        是否打乱当前 worker 的数据。

    返回
    ----
    DataLoader
        worker 使用的 DataLoader。
    """

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def print_dataset_info(train_loader: DataLoader, test_loader: DataLoader) -> None:
    """
    打印数据集基本信息，用于快速检查数据加载是否正常。

    参数
    ----
    train_loader : DataLoader
        训练集 DataLoader。
    test_loader : DataLoader
        测试集 DataLoader。
    """

    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)

    sample_x, sample_y = next(iter(train_loader))

    print(f"训练集样本数: {train_size}")
    print(f"测试集样本数: {test_size}")
    print(f"输入 batch 形状: {tuple(sample_x.shape)}")
    print(f"标签 batch 形状: {tuple(sample_y.shape)}")