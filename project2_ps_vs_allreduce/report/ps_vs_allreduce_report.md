# PS vs AllReduce：两类分布式训练系统实现与对比实验报告

## 1. 项目背景

分布式机器学习训练通常需要在多个计算进程或计算节点之间协同完成模型训练。对于数据并行训练，每个 worker 或 rank 持有一部分训练数据，并基于本地 mini-batch 计算梯度。系统随后需要将多个计算单元产生的梯度进行同步或聚合，从而更新模型参数。

本项目实现并对比两类典型分布式训练系统结构：

```text
1. Parameter Server；
2. AllReduce / PyTorch DDP。
```

同时保留一个 Single Process baseline 作为对照。Single baseline 用于衡量非分布式训练下的基础精度、训练时间和吞吐量。Parameter Server 用于理解中心化参数管理和 worker-server 通信模式。DDP / AllReduce 用于理解去中心化梯度同步和工程化数据并行训练机制。

本项目第一版只在单机上使用多进程实现，不涉及真实多机集群。数据集使用 MNIST，模型使用简单 MLP。该设置虽然规模较小，但足以验证分布式训练系统中的核心机制。

------

## 2. 项目目标

本项目目标包括：

```text
1. 实现 Single Process baseline；
2. 实现 multiprocessing 版本的同步 Parameter Server；
3. 实现 PyTorch DDP / AllReduce 训练；
4. 记录训练精度、测试精度、训练时间和系统吞吐量；
5. 估算模型大小和通信量；
6. 比较 Single、PS、DDP 的系统行为；
7. 理解中心化 PS 与去中心化 AllReduce 的结构差异；
8. 为后续 AllReduce、EASGD、异步 PS 等论文复现打基础。
```

------

## 3. 实验环境

实验环境如下：

| 项目         | 设置                  |
| ------------ | --------------------- |
| 操作系统     | Windows               |
| 编程语言     | Python                |
| 深度学习框架 | PyTorch               |
| 数据集       | MNIST                 |
| 模型         | MLP                   |
| 分布式方式   | 单机多进程            |
| PS 通信方式  | multiprocessing Queue |
| DDP 后端     | gloo                  |
| GPU          | 第一版默认 CPU        |

依赖库包括：

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

------

## 4. 项目结构

```text
ps-vs-allreduce/
├── common/
│   ├── seed.py
│   ├── datasets.py
│   ├── models.py
│   ├── metrics.py
│   └── logger.py
├── single/
│   └── train_single.py
├── ps/
│   ├── worker.py
│   ├── server.py
│   └── train_ps.py
├── ddp/
│   └── train_ddp.py
├── utils/
│   ├── comm.py
│   ├── summarize.py
│   └── plot.py
├── scripts/
│   ├── run_single.py
│   ├── run_ps.py
│   ├── run_ddp.py
│   └── run_all.py
├── results/
│   ├── raw/
│   ├── tables/
│   └── figures/
└── report/
    └── ps_vs_allreduce_report.md
```

------

## 5. 数据集与模型

### 5.1 数据集

实验使用 MNIST 手写数字分类数据集。

```text
训练集样本数：60000
测试集样本数：10000
输入形状：[1, 28, 28]
类别数：10
```

数据预处理包括：

```text
1. 转换为 Tensor；
2. 使用 MNIST 均值和标准差进行 Normalize。
```

### 5.2 模型

实验使用简单 MLP：

```text
Flatten
Linear(784, 256)
ReLU
Linear(256, 256)
ReLU
Linear(256, 10)
```

模型输出维度为：

```text
[batch_size, 10]
```

对应 MNIST 的 10 个类别。

------

## 6. 方法设计

### 6.1 Single Process baseline

Single baseline 是普通单进程 PyTorch 训练流程：

```text
1. 加载完整训练集；
2. 构造模型；
3. 使用 SGD 优化器；
4. 每个 batch 执行 forward；
5. 计算 CrossEntropyLoss；
6. 执行 backward；
7. 执行 optimizer.step；
8. 每个 epoch 后在测试集评估。
```

训练更新形式为：

$$
w_{t+1}=w_t-\eta g_t
$$

其中，$w_t$ 是当前模型参数，$\eta$ 是学习率，$g_t$ 是当前 mini-batch 梯度。

Single baseline 没有跨进程通信，因此通信量为：

$$
C_{\mathrm{single}}=0
$$

------

### 6.2 Parameter Server

Parameter Server 使用一个 server 进程和多个 worker 进程。

结构如下：

```text
main process
├── parameter server process
├── worker process 0
├── worker process 1
├── ...
└── worker process K-1
```

server 负责：

```text
1. 保存全局模型；
2. 向 worker 下发模型参数；
3. 接收 worker 上传的梯度；
4. 聚合梯度；
5. 更新全局模型；
6. 评估测试集；
7. 保存日志。
```

worker 负责：

```text
1. 读取自己的数据分片；
2. 接收 server 下发的全局模型参数；
3. 加载参数到本地模型副本；
4. 计算本地 mini-batch 梯度；
5. 上传梯度、loss、accuracy 和样本数。
```

同步 Parameter Server 的梯度聚合公式为：

$$
g_t=\sum_{k=1}^{K}\frac{n_k}{n}g_k$w_t$
$$

其中：

```text
K      worker 数量
n_k    第 k 个 worker 当前 batch 的样本数
n      所有 worker 当前 step 的样本总数
g_k    第 k 个 worker 计算得到的本地梯度
g_t    server 聚合后的全局梯度
```

server 更新参数：

$$
w_{t+1}=w_t-\eta g_t
$$

本项目实现的是同步 PS-SGD，即 server 必须等待所有 worker 上传梯度之后才更新模型。

------

### 6.3 DDP / AllReduce

PyTorch DDP 是工程中常用的数据并行训练方式。它与 Parameter Server 的主要区别是：DDP 没有中心 server 保存唯一全局模型，而是每个 rank 持有完整模型副本。

DDP 训练结构：

```text
rank 0: model replica + data shard 0
rank 1: model replica + data shard 1
...
rank K-1: model replica + data shard K-1
```

每个 rank 执行：

```text
1. 读取自己的数据分片；
2. 本地 forward；
3. 本地 loss 计算；
4. loss.backward；
5. DDP 自动执行梯度 AllReduce；
6. optimizer.step。
```

DDP 梯度同步可表示为：

$$
g_t=\frac{1}{K}\sum_{k=1}^{K}g_k$w_t$
$$

但这个平均过程不需要用户手写。DDP 会在 `loss.backward()` 后自动触发 AllReduce。

DDP 中需要使用 `DistributedSampler`，其作用是保证不同 rank 读取不同训练数据分片。如果不使用 `DistributedSampler`，多个 rank 可能会重复读取相同样本，导致训练数据重复和统计口径错误。

------

## 7. 通信量估算

### 7.1 模型参数量

模型参数总量为：

$$
P=\sum_{i=1}^{L}|\theta_i|
$$

其中，$\theta_i$ 是第 $i$ 个参数张量。

### 7.2 模型大小

若使用 float32，则每个参数占 4 bytes：

$$
S_{\mathrm{model}}=4P \ \mathrm{bytes}
$$

### 7.3 Parameter Server 通信量

Parameter Server 中，每个 worker 每轮通常需要：

```text
1. 从 server 下载一份全局模型参数；
2. 向 server 上传一份梯度。
```

因此每个 step 的总通信量估算为：

$$
C_{\mathrm{PS}}\approx 2KS_{\mathrm{model}}
$$

其中 $K$ 是 worker 数量。

一个 epoch 的通信量近似为：

$$
C_{\mathrm{PS,epoch}}
\approx
2KS_{\mathrm{model}}\times \mathrm{steps\_per\_epoch}
$$

### 7.4 Ring AllReduce 通信量

Ring AllReduce 中，每个进程每轮通信量近似为：

$$
C_{\mathrm{Ring}}\approx 2\frac{K-1}{K}S_{\mathrm{model}}
$$

一个 epoch 的每进程通信量近似为：

$$
C_{\mathrm{Ring,epoch}}
\approx
2\frac{K-1}{K}S_{\mathrm{model}}\times \mathrm{steps\_per\_epoch}
$$

需要注意：本项目记录的 DDP 通信量是每个 rank 的近似通信量，而 PS 通信量是 worker-server 总通信量。因此二者数值口径不同，报告分析时不能简单等同。

------

## 8. 实验设置

主要实验参数如下：

| 参数        | 设置  |
| ----------- | ----- |
| 数据集      | MNIST |
| 模型        | MLP   |
| 优化器      | SGD   |
| 学习率      | 0.01  |
| batch size  | 64    |
| epochs      | 10    |
| seed        | 42    |
| PS workers  | 2 / 4 |
| DDP workers | 2 / 4 |
| DDP backend | gloo  |
| device      | CPU   |

运行完整实验矩阵命令：

```powershell
python -m scripts.run_all --dataset mnist --model mlp --epochs 10 --batch-size 64 --lr 0.01 --seed 42 --device cpu --include-4-workers
```

汇总结果命令：

```powershell
python -m utils.summarize --raw-dir results/raw --output results/tables/summary.csv
```

绘制图表命令：

```powershell
python -m utils.plot --raw-dir results/raw --summary results/tables/summary.csv --fig-dir results/figures
```

------

## 9. 实验输出

本项目生成的原始日志包括：

```text
results/raw/single_mnist_mlp_workers1_seed42.csv
results/raw/ps_mnist_mlp_workers2_seed42.csv
results/raw/ps_mnist_mlp_workers4_seed42.csv
results/raw/ddp_mnist_mlp_workers2_seed42.csv
results/raw/ddp_mnist_mlp_workers4_seed42.csv
```

汇总表：

```text
results/tables/summary.csv
```

图表：

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

## 10. 实验结果

实验结果以 `results/tables/summary.csv` 为准。该表汇总了每种系统的最终精度、平均 epoch 时间、平均吞吐量和通信量估算。

建议在报告最终版中从 `summary.csv` 提取以下列：

```text
system_label
final_test_acc
mean_epoch_time
mean_samples_per_sec
final_comm_mb
total_elapsed_time
```

可以使用如下命令打印关键结果：

```powershell
python -c "import pandas as pd; df=pd.read_csv('results/tables/summary.csv'); cols=['system_label','final_test_acc','mean_epoch_time','mean_samples_per_sec','final_comm_mb','total_elapsed_time']; print(df[cols].to_string(index=False))"
```

实验结果表建议格式如下：

| System | Final Test Acc      | Mean Epoch Time     | Mean Samples/s      | Communication MB / Epoch |
| ------ | ------------------- | ------------------- | ------------------- | ------------------------ |
| Single | 以 summary.csv 为准 | 以 summary.csv 为准 | 以 summary.csv 为准 | 0                        |
| PS-2   | 以 summary.csv 为准 | 以 summary.csv 为准 | 以 summary.csv 为准 | 以 summary.csv 为准      |
| DDP-2  | 以 summary.csv 为准 | 以 summary.csv 为准 | 以 summary.csv 为准 | 以 summary.csv 为准      |
| PS-4   | 以 summary.csv 为准 | 以 summary.csv 为准 | 以 summary.csv 为准 | 以 summary.csv 为准      |
| DDP-4  | 以 summary.csv 为准 | 以 summary.csv 为准 | 以 summary.csv 为准 | 以 summary.csv 为准      |

------

## 11. 结果分析

### 11.1 精度分析

从 `system_accuracy.png` 可以观察 Single、PS、DDP 的测试准确率随 epoch 的变化。正常情况下，三类系统的准确率都应随训练轮数整体上升，并在 MNIST + MLP 设置下达到较高测试准确率。

Single、PS 和 DDP 使用的是同一数据集、同一模型结构和同一优化器，因此最终准确率应处于相近范围内。如果 PS 或 DDP 的准确率明显异常，通常需要检查：

```text
1. 数据是否正确划分；
2. 学习率是否一致；
3. batch size 口径是否一致；
4. 梯度是否正确聚合；
5. DDP 是否正确使用 DistributedSampler；
6. 是否只在 rank 0 保存日志。
```

### 11.2 时间分析

从 `epoch_time_compare.png` 可以比较不同系统的平均 epoch 时间。

在本项目的单机 CPU 环境下，PS 或 DDP 不一定比 Single 更快。原因是 MNIST + MLP 计算量较小，而多进程通信、同步和序列化开销相对明显。

尤其是 Parameter Server 版本使用 Python multiprocessing Queue 传递模型参数和梯度，这种实现更适合教学和机制验证，不是高性能通信实现。

### 11.3 吞吐量分析

`samples_per_sec_compare.png` 展示不同系统的平均吞吐量。

吞吐量定义为：

$$
\mathrm{samples/s}=\frac{N}{T}
$$

其中，$N$ 是当前 epoch 处理的训练样本数，$T$ 是 epoch 耗时。

吞吐量比单纯 accuracy 更能体现系统训练效率。对于分布式训练系统，仅比较最终准确率是不够的，还需要关注同等精度下的训练时间和吞吐量。

### 11.4 通信量分析

`comm_mb_compare.png` 展示每个 epoch 的理论通信量估算。

Single Process 没有跨进程通信，因此通信量为 0。

Parameter Server 的通信量随 worker 数增加而增加，因为每个 worker 都需要与 server 交换参数和梯度：

$$
C_{\mathrm{PS}}\approx 2KS_{\mathrm{model}}
$$

DDP / Ring AllReduce 的每进程通信量为：

$$
C_{\mathrm{Ring}}\approx 2\frac{K-1}{K}S_{\mathrm{model}}
$$

当 worker 数增加时，每个 rank 的通信量趋近于 $2S_{\mathrm{model}}$。

需要强调的是，本项目中的通信量是理论估算值，不是真实网络抓包结果。

### 11.5 worker 扩展性分析

`scalability_workers.png` 展示 worker 数量与吞吐量之间的关系。

理想情况下，worker 数增加应提高吞吐量。但在本项目中，worker 数增加不一定带来加速，原因可能包括：

```text
1. 多进程启动开销；
2. Python Queue 序列化和反序列化开销；
3. CPU 资源竞争；
4. 同步等待慢进程；
5. MNIST + MLP 任务规模较小；
6. 通信开销抵消并行计算收益。
```

因此，worker 数增加不一定更快。这也是分布式训练系统中必须分析 scalability 的原因。

------

## 12. 关键问题回答

### 12.1 Parameter Server 和 AllReduce 有什么区别？

Parameter Server 是中心化结构。server 保存全局模型，worker 从 server 拉取参数并上传梯度。server 聚合所有 worker 的梯度后更新全局模型。

AllReduce 是去中心化同步结构。每个 rank 都持有完整模型副本，每个 rank 本地计算梯度，然后通过 AllReduce 在 rank 之间同步梯度。同步后每个 rank 得到相同的平均梯度，并各自执行参数更新。

核心区别如下：

| 维度     | Parameter Server    | AllReduce / DDP        |
| -------- | ------------------- | ---------------------- |
| 结构     | 中心化              | 去中心化               |
| 全局模型 | server 持有         | 每个 rank 持有模型副本 |
| 梯度聚合 | server 手动聚合     | AllReduce 自动同步     |
| 通信模式 | worker-server       | rank-rank              |
| 瓶颈     | server 可能成为瓶颈 | 无显式中心 server      |
| 实现方式 | 自己写通信和聚合    | PyTorch DDP 封装       |

------

### 12.2 DDP 中谁负责梯度平均？

DDP 负责梯度平均。

用户代码只需要执行：

```python
loss.backward()
```

在 DDP 包装后的模型中，反向传播过程中会自动触发梯度 AllReduce。用户不需要手写：

```python
dist.all_reduce(...)
```

同步后的梯度近似为：

$$
g_t=\frac{1}{K}\sum_{k=1}^{K}g_k$w_t$
$$

之后每个 rank 执行：

```python
optimizer.step()
```

由于每个 rank 的梯度已经同步，所以各 rank 的参数更新保持一致。

------

### 12.3 为什么 PS 可能成为 bottleneck？

PS 可能成为瓶颈的原因是所有 worker 都要和 server 通信：

```text
1. worker 从 server 拉取模型参数；
2. worker 向 server 上传梯度；
3. server 需要接收所有 worker 的结果；
4. server 需要聚合梯度；
5. server 需要更新全局模型。
```

当 worker 数量增加时，server 的通信压力和聚合压力会增加。如果 server 处理不过来，整体训练速度会被 server 限制。

------

### 12.4 为什么 DDP 需要 DistributedSampler？

DDP 中每个 rank 是一个独立进程。如果没有 `DistributedSampler`，每个 rank 可能读取相同的数据，导致多个 rank 重复训练同一批样本。

`DistributedSampler` 的作用是：

```text
1. 将训练集划分给不同 rank；
2. 保证不同 rank 读取不同数据分片；
3. 保证每个 epoch 可以重新 shuffle；
4. 保证分布式训练的数据统计口径正确。
```

因此，在 DDP 训练中必须使用 `DistributedSampler`。

------

### 12.5 rank 和 world_size 是什么？

在 DDP 中：

```text
rank       当前进程编号
world_size 总进程数量
```

例如运行：

```powershell
torchrun --standalone --nproc_per_node=2 ddp/train_ddp.py ...
```

会启动两个进程：

```text
rank 0
rank 1
world_size = 2
```

通常只让 rank 0 负责打印日志、保存 CSV 和测试结果，避免多个进程重复写文件。

------

### 12.6 为什么要统计 samples/s？

`samples/s` 表示每秒处理的训练样本数：

$$
\mathrm{samples/s}=\frac{N}{T}
$$

其中，$N$ 是样本数，$T$ 是耗时。

在分布式训练系统中，只看 accuracy 不够。两个系统可能最终准确率接近，但训练时间和吞吐量差异很大。因此 samples/s 是衡量系统效率的重要指标。

------

### 12.7 worker 数变多为什么不一定更快？

worker 数变多后，计算并行度提高，但系统开销也会增加。

可能导致不加速的原因包括：

```text
1. 多进程创建和调度开销；
2. 进程间通信开销；
3. Python Queue 序列化开销；
4. DDP AllReduce 同步开销；
5. CPU 资源竞争；
6. 同步训练等待慢 worker；
7. 当前任务 MNIST + MLP 太小，计算量不足以抵消通信开销。
```

因此，分布式训练需要同时分析计算量、通信量、同步方式和硬件资源。

------

### 12.8 PS 和 DML-Bench 中 Sync-SGD 的关系是什么？

DML-Bench 中的 Sync-SGD 是单进程模拟多 worker 的算法机制。它重点关注数学更新和算法行为。

本项目中的 PS 是真实 multiprocessing 多进程实现。它不仅包含同步 SGD 的数学更新，还包含：

```text
1. 多进程 worker；
2. server-worker 队列通信；
3. 参数下发；
4. 梯度上传；
5. server 聚合；
6. 进程同步；
7. 通信量和吞吐量统计。
```

因此，两者的数学更新类似，但系统实现层级不同。

------

## 13. 项目结论

本项目实现了 Single、Parameter Server 和 DDP / AllReduce 三类训练系统，并在 MNIST + MLP 设置下完成了精度、时间、吞吐量和通信量对比。

主要结论如下：

```text
1. Single Process 是后续系统对比的基础基线；
2. Parameter Server 能清楚展示中心化参数管理和梯度聚合机制；
3. DDP / AllReduce 展示了去中心化梯度同步机制；
4. DDP 中梯度同步由框架自动完成，不需要用户手写聚合；
5. PS 中 server 可能成为通信和聚合瓶颈；
6. 分布式训练不能只看 accuracy，还必须比较 epoch time、samples/s 和 communication cost；
7. worker 数增加不一定带来加速，通信和同步开销可能抵消并行收益；
8. 本项目完成了从单进程算法模拟到真实多进程分布式训练系统的过渡。
```

------

## 14. 不足与后续工作

当前项目仍有一些限制：

```text
1. 只在单机多进程环境下运行，没有真实多机网络；
2. PS 使用 Python Queue，通信效率不高；
3. 通信量是理论估算，不是真实网络测量；
4. 任务规模较小，MNIST + MLP 难以体现大规模分布式训练优势；
5. 第一版没有实现异步 Parameter Server；
6. 第一版没有分析 staleness；
7. 第一版没有使用 GPU + NCCL。
```

后续可以扩展：

```text
1. 实现异步 Parameter Server；
2. 加入 stale gradient 与 staleness 统计；
3. 扩展到 CIFAR-10 + CNN；
4. 对比 Local SGD、EASGD、Elastic Averaging；
5. 使用 GPU + NCCL 跑 DDP；
6. 实现 AllReduce 通信算法模拟器；
7. 复现 AllReduce、IBing 或 EASGD 相关论文；
8. 增加多 seed 实验，报告均值和标准差。
```

------

## 15. 最终完成情况

本项目第一版完成后，应具备以下输出：

```text
1. Single 训练代码；
2. Parameter Server 多进程训练代码；
3. DDP / AllReduce 训练代码；
4. 统一运行脚本；
5. CSV 原始日志；
6. summary.csv 汇总表；
7. 多张实验对比图；
8. README 项目说明；
9. 实验报告。
```

这说明项目已经完成从算法机制模拟到真实多进程分布式训练系统实践的核心过渡。
