# PS vs AllReduce：两类分布式训练系统实现与对比

## 1. 项目简介

本项目是一个单机多进程分布式训练系统实践项目，目标是对比三类训练方式：

1. Single Process baseline；
2. Multiprocessing Parameter Server；
3. PyTorch DDP / AllReduce。

项目第一版使用：

```text
数据集：MNIST / Fashion-MNIST
模型：Logistic Regression / MLP
运行环境：单机多进程，不做真实多机集群
```

本项目的重点不是追求最高精度，而是理解分布式训练系统中的核心机制：

```text
1. 数据并行；
2. 参数服务器；
3. AllReduce 梯度同步；
4. 多进程训练；
5. 通信量估算；
6. 训练吞吐量；
7. worker 数扩展性；
8. 系统瓶颈分析。
```

本项目可以看作从单进程算法模拟进入真实多进程分布式训练系统的过渡项目。

------

## 2. 项目目标

本项目需要完成以下目标：

```text
1. 实现单进程 PyTorch 训练 baseline；
2. 实现 multiprocessing 版本的同步 Parameter Server；
3. 实现 PyTorch DDP / AllReduce 训练；
4. 统一记录 train_loss、train_acc、test_loss、test_acc；
5. 统一记录 epoch_time、elapsed_time、samples_per_sec；
6. 估算模型参数量、模型大小和通信量；
7. 对比 Single、PS、DDP 的精度、时间、吞吐量和通信成本；
8. 生成 summary.csv 和实验图表；
9. 形成可复现的小型实验报告。
```

------

## 3. 项目结构

```text
ps-vs-allreduce/
├── README.md
├── requirements.txt
├── configs/
├── scripts/
│   ├── __init__.py
│   ├── run_single.py
│   ├── run_ps.py
│   ├── run_ddp.py
│   └── run_all.py
├── common/
│   ├── __init__.py
│   ├── seed.py
│   ├── datasets.py
│   ├── models.py
│   ├── metrics.py
│   └── logger.py
├── single/
│   ├── __init__.py
│   └── train_single.py
├── ps/
│   ├── __init__.py
│   ├── worker.py
│   ├── server.py
│   └── train_ps.py
├── ddp/
│   ├── __init__.py
│   └── train_ddp.py
├── utils/
│   ├── __init__.py
│   ├── comm.py
│   ├── summarize.py
│   └── plot.py
├── results/
│   ├── raw/
│   ├── tables/
│   └── figures/
└── report/
    └── ps_vs_allreduce_report.md
```

------

## 4. 环境安装

### 4.1 创建环境

可以使用已有 Python 环境，也可以单独创建虚拟环境。

建议 Python 版本：

```text
Python >= 3.9
```

### 4.2 安装依赖

在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
```

### 4.3 验证环境

```powershell
python -c "import torch, torchvision, numpy, pandas, matplotlib; print('env ok'); print(torch.__version__)"
```

如果输出 `env ok`，说明基础环境可用。

------

## 5. 依赖列表

`requirements.txt` 中主要依赖如下：

```text
torch
torchvision
numpy
pandas
matplotlib
tqdm
pyyaml
Pillow
```

各依赖作用：

```text
torch          PyTorch 训练框架
torchvision    MNIST / Fashion-MNIST 数据集
numpy          数值计算
pandas         CSV 日志读取与汇总
matplotlib     绘图
tqdm           训练进度条
pyyaml         后续配置文件扩展
Pillow         图像数据处理依赖
```

------

## 6. 运行 Single Process baseline

### 6.1 直接运行

```powershell
python -m single.train_single --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42 --device cpu
```

### 6.2 使用统一脚本运行

```powershell
python -m scripts.run_single --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42 --device cpu
```

### 6.3 输出文件

```text
results/raw/single_mnist_mlp_workers1_seed42.csv
```

Single baseline 的 `comm_mb` 应为：

```text
0.0
```

因为单进程训练没有跨进程通信。

------

## 7. 运行 Parameter Server

### 7.1 运行 2 workers

```powershell
python -m ps.train_ps --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42 --num-workers 2 --device cpu
```

或者：

```powershell
python -m scripts.run_ps --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42 --num-workers 2 --device cpu
```

### 7.2 运行 4 workers

```powershell
python -m scripts.run_ps --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42 --num-workers 4 --device cpu
```

### 7.3 输出文件

```text
results/raw/ps_mnist_mlp_workers2_seed42.csv
results/raw/ps_mnist_mlp_workers4_seed42.csv
```

### 7.4 Parameter Server 结构

```text
main process
├── parameter server process
├── worker process 0
├── worker process 1
├── worker process 2
└── worker process 3
```

server 负责保存全局模型，worker 负责计算本地梯度。

------

## 8. 运行 DDP / AllReduce

### 8.1 使用 torchrun 运行 2 workers

```powershell
torchrun --standalone --nproc_per_node=2 ddp/train_ddp.py --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42 --device cpu --backend gloo
```

### 8.2 使用 Python 模块方式运行 2 workers

如果 PowerShell 找不到 `torchrun`，使用：

```powershell
python -m torch.distributed.run --standalone --nproc_per_node=2 ddp/train_ddp.py --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42 --device cpu --backend gloo
```

### 8.3 使用统一脚本运行

```powershell
python -m scripts.run_ddp --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42 --num-workers 2 --device cpu --backend gloo
```

### 8.4 运行 4 workers

```powershell
python -m scripts.run_ddp --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42 --num-workers 4 --device cpu --backend gloo
```

### 8.5 输出文件

```text
results/raw/ddp_mnist_mlp_workers2_seed42.csv
results/raw/ddp_mnist_mlp_workers4_seed42.csv
```

------

## 9. 一键运行全部实验

### 9.1 最小实验矩阵

运行：

```powershell
python -m scripts.run_all --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42 --device cpu
```

该命令会运行：

```text
Single Process
PS workers=2
DDP workers=2
```

### 9.2 完整实验矩阵

运行：

```powershell
python -m scripts.run_all --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42 --device cpu --include-4-workers
```

该命令会运行：

```text
Single Process
PS workers=2
DDP workers=2
PS workers=4
DDP workers=4
```

------

## 10. 结果汇总

运行：

```powershell
python -m utils.summarize --raw-dir results/raw --output results/tables/summary.csv
```

输出文件：

```text
results/tables/summary.csv
```

汇总表包含：

```text
system
dataset
model
num_workers
epochs
final_train_loss
final_train_acc
final_test_loss
final_test_acc
mean_epoch_time
total_elapsed_time
mean_samples_per_sec
final_comm_mb
model_params
model_size_mb
seed
log_file
system_label
```

------

## 11. 绘制图表

运行：

```powershell
python -m utils.plot --raw-dir results/raw --summary results/tables/summary.csv --fig-dir results/figures
```

输出图表：

```text
results/figures/system_accuracy.png
results/figures/epoch_time_compare.png
results/figures/samples_per_sec_compare.png
results/figures/comm_mb_compare.png
results/figures/scalability_workers.png
results/figures/train_loss_curves.png
results/figures/test_loss_curves.png
```

------

## 12. 日志字段说明

每个训练 CSV 都包含以下核心字段：

| 字段              | 含义                      |
| ----------------- | ------------------------- |
| `system`          | 系统类型：single、ps、ddp |
| `dataset`         | 数据集名称                |
| `model`           | 模型名称                  |
| `num_workers`     | worker / rank 数量        |
| `epoch`           | 当前训练轮数              |
| `train_loss`      | 当前 epoch 平均训练损失   |
| `train_acc`       | 当前 epoch 训练准确率     |
| `test_loss`       | 当前 epoch 后测试集损失   |
| `test_acc`        | 当前 epoch 后测试集准确率 |
| `epoch_time`      | 当前 epoch 耗时           |
| `elapsed_time`    | 累计耗时                  |
| `samples_per_sec` | 训练吞吐量                |
| `model_params`    | 模型参数量                |
| `model_size_mb`   | 模型大小估算              |
| `comm_mb`         | 每个 epoch 理论通信量估算 |
| `seed`            | 随机种子                  |
| `lr`              | 学习率                    |
| `batch_size`      | 本地 batch size           |

------

## 13. 核心公式

### 13.1 模型参数量

$$
P=\sum_{i=1}^{L}|\theta_i|
$$

其中，$P$ 是模型参数总量，$\theta_i$ 是第 $i$ 个参数张量。

### 13.2 模型大小

若使用 float32，则每个参数占 4 bytes：

$$
S_{\mathrm{model}}=4P \ \mathrm{bytes}
$$

### 13.3 Parameter Server 通信量估算

Parameter Server 中，每个 worker 每轮通常需要：

```text
1. 从 server 下载模型参数；
2. 向 server 上传梯度。
```

因此每个 step 的总通信量近似为：

$$
C_{\mathrm{PS}}\approx 2KS_{\mathrm{model}}
$$

其中 $K$ 是 worker 数量。

### 13.4 Ring AllReduce 通信量估算

Ring AllReduce 中，每个进程每轮通信量近似为：

$$
C_{\mathrm{Ring}}\approx 2\frac{K-1}{K}S_{\mathrm{model}}
$$

其中 $K$ 是进程数量。

### 13.5 吞吐量

$$
\mathrm{samples/s}=\frac{N}{T}
$$

其中，$N$ 是当前 epoch 处理的训练样本数，$T$ 是当前 epoch 耗时。

------

## 14. Single、PS、DDP 对比

| 维度     | Single           | Parameter Server         | DDP / AllReduce            |
| -------- | ---------------- | ------------------------ | -------------------------- |
| 进程数量 | 1                | 1 server + K workers     | K ranks                    |
| 数据划分 | 不划分           | worker 数据分片          | DistributedSampler         |
| 模型位置 | 单进程           | server 持有全局模型      | 每个 rank 持有完整模型副本 |
| 梯度同步 | 无               | worker 上传，server 聚合 | DDP 自动 AllReduce         |
| 参数更新 | 单进程 optimizer | server optimizer         | 每个 rank optimizer        |
| 通信量   | 0                | worker-server 通信       | rank-rank AllReduce        |
| 中心瓶颈 | 无               | server 可能成为瓶颈      | 无显式中心 server          |
| 日志保存 | 单进程保存       | server 保存              | rank 0 保存                |

------

## 15. 常见问题

### 15.1 DDP 中谁负责梯度平均？

DDP 在 `loss.backward()` 后自动触发梯度 AllReduce。用户不需要手写梯度聚合。

### 15.2 为什么 DDP 需要 DistributedSampler？

因为 DDP 中每个 rank 都是独立进程。如果不使用 `DistributedSampler`，每个 rank 可能读取相同数据，导致数据重复。`DistributedSampler` 可以保证不同 rank 读取不同训练数据分片。

### 15.3 rank 和 world_size 是什么？

```text
rank       当前进程编号
world_size 总进程数量
```

例如：

```powershell
torchrun --standalone --nproc_per_node=2 ...
```

会启动两个进程：

```text
rank 0
rank 1
world_size = 2
```

### 15.4 为什么 PS 可能成为瓶颈？

因为所有 worker 都需要和 server 通信：

```text
1. worker 从 server 拉取参数；
2. worker 向 server 上传梯度；
3. server 聚合所有梯度；
4. server 更新全局模型。
```

当 worker 数增加时，server 的通信和聚合压力会增加。

### 15.5 为什么 worker 数增加不一定更快？

原因包括：

```text
1. 多进程启动开销；
2. Python Queue 序列化开销；
3. CPU 资源竞争；
4. 同步等待慢 worker；
5. MNIST + MLP 任务太小；
6. 通信开销抵消并行收益。
```

### 15.6 PS 和 DML-Bench 中 Sync-SGD 的关系是什么？

两者数学更新类似，都是同步聚合多个 worker 的梯度。

区别是：

```text
DML-Bench：单进程模拟多 worker 的算法机制；
本项目 PS：真实 multiprocessing 多进程通信实现。
```

------

## 16. 当前项目完成标准

本项目第一版完成后，应满足：

```text
1. Single 能运行并保存 CSV；
2. PS workers=2 能运行并保存 CSV；
3. PS workers=4 能运行并保存 CSV；
4. DDP workers=2 能运行并保存 CSV；
5. DDP workers=4 能运行并保存 CSV；
6. 能生成 results/tables/summary.csv；
7. 能生成 results/figures/*.png；
8. 能解释 PS 与 AllReduce 的系统结构差异；
9. 能解释 DDP 中梯度同步由谁完成；
10. 能解释为什么分布式训练不仅看 accuracy，还要看 time、samples/s 和 communication cost。
```

------

## 17. 后续扩展方向

后续可以继续扩展：

```text
1. 异步 Parameter Server；
2. stale gradient / staleness 统计；
3. Local SGD 与 DDP 对比；
4. CIFAR-10 + CNN；
5. 多 seed 实验；
6. GPU + NCCL DDP；
7. AllReduce 算法模拟器；
8. EASGD / Elastic Averaging SGD；
9. IBing / AllReduce 相关论文复现。
```
