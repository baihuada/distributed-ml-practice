# 分布式机器学习阶段性学习与实践总结

## 1. 背景说明

2026 年 4 月，我与张嘉鹏老师进行了初步交流。老师建议我先阅读《分布式机器学习：算法、理论与实践》，了解分布式机器学习中的基本概念、算法机制和编程技术。

此后，我围绕分布式机器学习方向进行了系统学习，并结合书中的内容完成了三个阶段性实践项目：

1. **DML-Bench：分布式优化算法模拟实践**
2. **PS vs AllReduce：Parameter Server 与 AllReduce 架构对比实践**
3. **IBing Reproduction：AllReduce 通信优化论文复现**

这三个项目分别对应分布式机器学习中的三个层次：

```text
分布式优化算法机制
        ↓
分布式训练系统架构
        ↓
AllReduce 通信优化
```

本总结用于概括目前的学习内容、实践成果、主要收获、当前不足以及后续希望深入的方向。

---

## 2. 已完成学习内容

### 2.1 理论学习

已系统阅读《分布式机器学习：算法、理论与实践》，重点学习了以下内容：

- 数据并行与模型并行；
- 分布式随机梯度下降；
- 同步 SGD 与异步 SGD；
- Local SGD 与模型平均；
- Parameter Server 架构；
- AllReduce 与 Ring AllReduce；
- 分布式训练中的通信开销；
- straggler 慢节点问题；
- 梯度同步与模型聚合机制；
- 分布式机器学习中的精度、通信量和训练时间权衡。

通过阅读和实践，我逐渐认识到，分布式机器学习的核心不仅是“把训练任务分到多个 worker 上”，更重要的是在以下目标之间进行权衡：

```math
\text{Accuracy}
\quad \mathrm{vs.} \quad
\text{Communication Cost}
\quad \mathrm{vs.} \quad
\text{Training Time}
```

---

## 3. Project 1：DML-Bench 分布式优化算法模拟实践

### 3.1 项目目标

DML-Bench 的目标是用一个统一的模拟框架理解分布式优化中的基本机制，包括：

- Centralized SGD；
- Sync-SGD；
- Async-SGD；
- Local SGD / Model Averaging；
- straggler 模拟；
- 通信频率分析；
- 通信量统计；
- 精度和训练时间对比。

该项目主要用于理解分布式优化算法本身，而不是追求真实多机系统性能。

---

### 3.2 核心优化目标

假设共有 $K$ 个 worker，第 $k$ 个 worker 拥有 $n_k$ 个样本，总样本数为 $n$。全局优化目标可以写为：

```math
F(w)
=
\sum_{k=1}^{K}
\frac{n_k}{n}
F_k(w)
```

其中：

```math
F_k(w)
=
\frac{1}{n_k}
\sum_{i \in \mathcal{D}_k}
\ell(w; x_i, y_i)
```

这里，$F_k(w)$ 表示第 $k$ 个 worker 上的局部目标函数，$\mathcal{D}_k$ 表示该 worker 的本地数据集。

---

### 3.3 已实现内容

DML-Bench 中主要实现了：

```text
Centralized SGD
Sync-SGD
Async-SGD
Local SGD / Model Averaging
Straggler simulation
Communication frequency analysis
Metric logging
Result plotting
```

---

### 3.4 Centralized SGD

Centralized SGD 作为单机训练 baseline，用于提供非分布式训练参考结果。

其参数更新形式为：

```math
w_{t+1}
=
w_t
-
\eta g_t
```

其中，$w_t$ 表示第 $t$ 步的模型参数，$\eta$ 表示学习率，$g_t$ 表示当前 mini-batch 上计算得到的随机梯度。

---

### 3.5 Sync-SGD

Sync-SGD 中，所有 worker 同步计算梯度，server 聚合后统一更新模型。

第 $k$ 个 worker 在全局模型 $w_t$ 上计算梯度：

```math
g_k(w_t)
=
\nabla F_k(w_t)
```

server 端进行加权平均：

```math
g_t
=
\sum_{k=1}^{K}
\frac{n_k}{n}
g_k(w_t)
```

然后更新全局模型：

```math
w_{t+1}
=
w_t
-
\eta g_t
```

该方法的优点是梯度新鲜、更新稳定；缺点是必须等待所有 worker 完成计算，因此容易受到最慢 worker 的影响。

---

### 3.6 Local SGD / Model Averaging

Local SGD 的思想是减少通信频率。每个 worker 不再每一步都通信，而是在本地训练 $E$ 步后再进行模型平均。

第 $r$ 轮通信开始时，server 广播全局模型 $w^r$。每个 worker 本地训练 $E$ 步后得到 $w_k^{r,E}$。server 端进行模型平均：

```math
w^{r+1}
=
\sum_{k=1}^{K}
\frac{n_k}{n}
w_k^{r,E}
```

其中，$E$ 表示每轮通信之间的本地训练步数。

该方法可以减少通信频率，但如果 $E$ 过大，或者数据分布高度 Non-IID，不同 worker 的模型可能发生明显漂移。

---

### 3.7 Async-SGD

Async-SGD 中，worker 不等待其他 worker，计算完成后立即向 server 提交梯度。

如果 worker 使用的是旧参数 $w_{t-\tau}$，则 server 当前参数为 $w_t$ 时仍然可能使用旧梯度更新：

```math
w_{t+1}
=
w_t
-
\eta g_k(w_{t-\tau})
```

其中，$\tau$ 表示 staleness，即梯度延迟。

Async-SGD 可以缓解 straggler 导致的等待问题，但 stale gradient 可能导致更新方向过时，从而影响收敛稳定性。

---

### 3.8 项目收获

通过 DML-Bench，我主要理解了：

- Sync-SGD 的核心问题是同步等待；
- Async-SGD 的核心问题是 stale gradient；
- Local SGD 的核心问题是通信频率和模型漂移之间的权衡；
- straggler 会显著影响同步训练效率；
- 分布式优化不能只看最终精度，还需要同时考虑通信量和训练时间。

---

## 4. Project 2：PS vs AllReduce 架构对比实践

### 4.1 项目目标

PS vs AllReduce 项目的目标是实现并比较两类典型分布式训练架构：

1. Parameter Server；
2. AllReduce / DDP-style training。

该项目主要用于理解分布式训练系统中的通信拓扑、梯度同步方式和系统瓶颈。

---

### 4.2 Parameter Server 架构

Parameter Server 架构中，server 保存全局模型参数，worker 负责本地计算梯度并上传。

典型流程如下：

```text
1. Worker pulls global parameters from server
2. Worker computes gradients on local data
3. Worker pushes gradients to server
4. Server aggregates gradients and updates global model
5. Repeat
```

如果模型参数量为 $P$，每个参数使用 `float32`，即 4 bytes，worker 数量为 $K$，则一次上传梯度的通信量近似为：

```math
\operatorname{UploadBytes}
=
K \times P \times 4
```

如果同时考虑 server 下发模型参数和 worker 上传梯度，则一次完整通信量可以近似表示为：

```math
\operatorname{CommBytes}
=
2KP \times 4
```

该架构实现直观，但当 worker 数量增加时，server 容易成为通信瓶颈。

---

### 4.3 AllReduce / DDP 架构

AllReduce 架构中，每个 worker 都保存完整模型副本。每个 worker 在本地 mini-batch 上计算梯度，然后通过 AllReduce 同步所有 worker 的梯度。

AllReduce 的目标是让每个 worker 都得到全局梯度和：

```math
g
=
\sum_{k=1}^{K}
g_k
```

如果使用平均梯度更新，则每个 worker 使用：

```math
\bar{g}
=
\frac{1}{K}
\sum_{k=1}^{K}
g_k
```

随后每个 worker 执行一致的模型更新：

```math
w_{t+1}
=
w_t
-
\eta \bar{g}
```

该架构避免了中心 server 瓶颈，更接近现代分布式深度学习中的 DDP 训练方式。

---

### 4.4 已实现内容

PS vs AllReduce 项目中主要实现了：

```text
Single-process training baseline
Parameter Server training
AllReduce / DDP-style training
Training accuracy comparison
Training time comparison
Communication cost analysis
Scalability analysis
```

---

### 4.5 项目收获

通过该项目，我主要理解了：

- Parameter Server 和 AllReduce 是两种典型的分布式训练通信拓扑；
- Parameter Server 结构清晰，但中心 server 可能成为瓶颈；
- AllReduce 更适合去中心化梯度同步；
- DDP 的核心思想是每个进程持有模型副本，并通过 collective communication 同步梯度；
- 系统架构会直接影响训练吞吐量、通信开销和扩展性。

---

## 5. Project 3：IBing AllReduce 通信优化论文复现

### 5.1 复现论文

复现论文为：

> Ruixing Zong, Jiapeng Zhang, Zhuo Tang, and Kenli Li.  
> **IBing: An Efficient Interleaved Bidirectional Ring All-Reduce Algorithm for Gradient Synchronization**.  
> ACM Transactions on Architecture and Code Optimization, 2025.

该论文关注分布式训练中的 AllReduce 梯度同步通信优化，提出了 Interleaved Bidirectional Ring AllReduce，即 IBing。

---

### 5.2 项目目标

IBing 复现项目的目标不是完整复现论文中的大规模超算集群实验，而是复现其核心算法机制，包括：

- Ring AllReduce 通信过程；
- IBing 交错双向通信调度；
- 单进程通信模拟；
- MPI 多进程实现；
- 正确性测试；
- benchmark；
- 绘图与实验结果分析。

---

### 5.3 Ring AllReduce

标准 Ring AllReduce 分为两个阶段：

```text
Reduce-Scatter
All-Gather
```

对于 $N$ 个 worker，Reduce-Scatter 阶段需要 $N-1$ 步，All-Gather 阶段也需要 $N-1$ 步。因此标准 Ring AllReduce 的总通信步数为：

```math
T_{\mathrm{Ring}}
=
2(N-1)
```

---

### 5.4 IBing AllReduce

IBing 将标准 Ring 的单向通信改为交错双向通信。每个 step 中，每个 rank 同时向左右两个方向发送不同 chunk，并从左右两个邻居接收不同 chunk。

IBing 的总通信步数为：

```math
T_{\mathrm{IBing}}
=
N-1
```

因此，相比标准 Ring AllReduce，IBing 的通信步数减少比例为：

```math
\operatorname{Reduction}
=
\frac{
T_{\mathrm{Ring}}
-
T_{\mathrm{IBing}}
}{
T_{\mathrm{Ring}}
}
\times 100\%
```

将 $T_{\mathrm{Ring}} = 2(N-1)$ 和 $T_{\mathrm{IBing}} = N-1$ 代入，得到：

```math
\operatorname{Reduction}
=
\frac{
2(N-1)
-
(N-1)
}{
2(N-1)
}
\times 100\%
=
50\%
```

---

### 5.5 IBing 通信调度公式

对于 rank $r$，step $i$，worker 总数 $N$，方向 1 的通信调度为：

```math
\operatorname{recv\_chunk}_1
=
(r-i-1+N)
\bmod N
```

```math
\operatorname{send\_chunk}_1
=
(r-i+N)
\bmod N
```

方向 2 的通信调度为：

```math
\operatorname{recv\_chunk}_2
=
(r+i+N+2)
\bmod N
```

```math
\operatorname{send\_chunk}_2
=
(r+i+N+1)
\bmod N
```

对于 rank $r$，左右邻居定义为：

```math
\operatorname{left}(r)
=
(r-1+N)
\bmod N
```

```math
\operatorname{right}(r)
=
(r+1)
\bmod N
```

---

### 5.6 已实现内容

IBing 复现项目中主要实现了：

```text
IBing scheduling simulation
IBing single-process data-flow simulation
Ring AllReduce single-process baseline
Correctness tests
MPI Ring AllReduce
MPI IBing AllReduce
MPI correctness batch test
MPI benchmark
Batch benchmark runner
Plot generation
Reproduction report
```

---

### 5.7 实验结果与解释

实验在 Windows 单机 MS-MPI 环境下完成。主要结果如下：

- Ring、IBing 和 MPI_Allreduce 均通过正确性测试；
- IBing 的通信步数确实为 $N-1$；
- Ring 的通信步数确实为 $2(N-1)$；
- IBing 相比 Ring 的理论通信步数减少 $50\%$；
- IBing 在部分数据规模下快于 Ring；
- 在更多 Windows 单机多进程场景下，IBing 性能没有稳定优于 Ring；
- MPI_Allreduce 在中大规模数据下通常表现较好。

IBing 在本机环境下未稳定复现论文中的性能优势，主要原因包括：

- 实验环境为 Windows 单机多进程，而不是真实多节点集群；
- 缺少真实独立的双向网络链路；
- Python 和 mpi4py 调用开销较高；
- NumPy buffer copy 带来额外内存开销；
- 本机进程调度和内存带宽影响通信时间；
- MPI_Allreduce 是 MPI 库内部高度优化的 collective communication 实现。

因此，该项目的主要结论是：

```text
Correctness: Passed
Scheduling reproduction: Passed
Step-count reduction: Passed
MPI implementation: Passed
Benchmark pipeline: Passed
Paper-level performance reproduction: Not fully reproduced on Windows single-machine environment
```

---

## 6. 当前整体收获

通过以上三个项目，我对分布式机器学习形成了更具体的理解。

### 6.1 算法层面

在算法层面，我理解了：

- Centralized SGD 是分布式算法的单机基线；
- Sync-SGD 本质上是多个 worker 的梯度平均；
- Async-SGD 通过减少等待缓解 straggler，但会产生 stale gradient；
- Local SGD 通过减少通信频率提升效率，但可能产生 worker drift；
- 分布式优化需要同时考虑收敛速度、最终精度、通信频率和训练时间。

---

### 6.2 系统层面

在系统层面，我理解了：

- Parameter Server 与 AllReduce 是两种典型通信架构；
- Parameter Server 容易出现中心 server 瓶颈；
- AllReduce 更适合现代数据并行训练；
- DDP 的核心在于每个进程持有模型副本，并通过 collective communication 同步梯度；
- 通信拓扑和系统实现会显著影响训练吞吐量。

---

### 6.3 通信优化层面

在通信优化层面，我理解了：

- Ring AllReduce 是理解梯度同步通信的重要基础；
- IBing 通过交错双向通信减少通信步数；
- 理论通信步数减少不必然等价于实际运行时间减少；
- 实验结果必须结合运行环境分析；
- 单机多进程实验和多节点集群实验之间存在明显差异。

---

## 7. 当前不足

目前仍存在以下不足：

1. 实验主要在 Windows 单机环境中完成，缺少真实多节点环境测试；
2. MPI 实现使用 Python 和 mpi4py，性能无法代表 C++/MPI 或 NCCL 级通信库；
3. IBing 复现重点是核心机制和通信步数，没有完整复现论文中的超算集群结果；
4. DML-Bench 和 PS vs AllReduce 仍偏学习型实践，工程化程度还有提升空间；
5. 目前还没有在 GPU 多卡或多机环境下测试分布式训练性能；
6. 对大规模分布式训练系统，如 DDP、FSDP、NCCL 等，还需要进一步深入学习。

---

## 8. 后续希望深入的方向

后续希望在老师指导下，继续沿以下方向之一深入：

### 8.1 AllReduce 与分布式训练通信优化

继续研究 Ring AllReduce、IBing、Topology-Aware AllReduce、Hierarchical AllReduce 等通信优化方法，并尝试在更接近真实多节点环境的条件下测试。

### 8.2 分布式训练系统 Benchmark

继续扩展 PS vs AllReduce 项目，对 DDP、FSDP、gradient accumulation、mixed precision、通信瓶颈和训练吞吐量进行系统 benchmark。

### 8.3 联邦学习中的通信压缩与 Non-IID 优化

结合已有联邦学习基础，进一步研究通信压缩、Non-IID 数据分布、个性化联邦学习和鲁棒聚合等问题。

### 8.4 老师组内具体课题

如果老师当前组内有更具体、更紧急的研究任务，我也希望能根据老师建议调整方向，尽快进入组内科研节奏。

---

## 9. 当前项目链接与说明

当前 GitHub 仓库包括三个实践项目：

```text
project1_dml_bench/
project2_ps_vs_allreduce/
project3_ibing_reproduction/
```

每个项目均包含代码、实验结果或报告文件。仓库主要用于展示阶段性学习成果和实践过程，不代表最终成熟研究成果。

---

## 10. 总结

总体来看，这一阶段的学习和实践使我完成了从理论阅读到代码实现、再到论文复现的初步闭环：

```text
阅读分布式机器学习基础教材
        ↓
完成分布式优化算法模拟
        ↓
实现分布式训练系统架构对比
        ↓
复现 AllReduce 通信优化论文
        ↓
整理实验结果、图表和复现报告
```

目前我已经具备了对分布式机器学习基本概念、算法机制、系统架构和通信优化问题的初步理解。后续希望在老师指导下，进一步选择一个具体方向深入，逐步从学习型实践过渡到科研型问题探索。