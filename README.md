# Distributed Machine Learning Practice

本仓库用于记录我在系统阅读《分布式机器学习：算法、理论与实践》之后完成的阶段性实践内容。项目围绕分布式机器学习中的三个核心层次展开：**分布式优化算法机制、分布式训练系统架构、AllReduce 通信优化论文复现**。

本仓库不是工业级分布式训练框架，而是一个面向学习、实验和复现的实践仓库。目标是通过可运行代码、可复现实验、结果图表和报告分析，系统理解分布式机器学习中的同步机制、梯度聚合、通信瓶颈、慢节点问题和 AllReduce 通信优化。

---

## 1. Repository Overview

本仓库目前包含三个主要项目：

| Project | Name | Focus | Status |
|---|---|---|---|
| Project 1 | DML-Bench | 分布式优化算法模拟，包括 Sync-SGD、Async-SGD、Local SGD、straggler 和通信频率分析 | Completed |
| Project 2 | PS vs AllReduce | Parameter Server 与 AllReduce / DDP 两类分布式训练架构对比 | Completed |
| Project 3 | IBing Reproduction | AllReduce 通信优化论文 IBing 的核心机制复现与 MPI benchmark | Completed |

整体学习路径为：

```text
Distributed Optimization Algorithms
        ↓
Distributed Training System Architectures
        ↓
AllReduce Communication Optimization
```

对应到三个项目：

```text
DML-Bench
        ↓
PS vs AllReduce
        ↓
IBing Reproduction
```

---

## 2. Repository Structure

推荐的仓库结构如下：

```text
distributed-ml-practice/
├── README.md
├── project1_dml_bench/
├── project2_ps_vs_allreduce/
├── project3_ibing_reproduction/
└── docs/
    └── progress_summary.md
```

其中：

```text
project1_dml_bench/           # 分布式优化算法模拟实践
project2_ps_vs_allreduce/     # Parameter Server 与 AllReduce 架构对比实践
project3_ibing_reproduction/  # IBing AllReduce 通信优化论文复现
docs/                         # 阶段性学习总结与说明文档
```

---

## 3. Background

本仓库的实践内容来源于对分布式机器学习基础内容的系统学习。学习主线包括：

- 数据并行训练；
- 分布式随机梯度下降；
- 同步 SGD 与异步 SGD；
- Local SGD 与模型平均；
- Parameter Server 架构；
- AllReduce / Ring AllReduce；
- DDP 分布式训练；
- 通信瓶颈与慢节点问题；
- AllReduce 通信调度优化。

在学习基础概念后，本仓库进一步通过三个实践项目将理论内容落到代码和实验中。

---

## 4. Project 1: DML-Bench

### 4.1 Project Goal

`DML-Bench` 是一个分布式优化算法模拟实践项目，目标是用统一代码框架模拟和比较典型分布式优化机制。

该项目重点关注：

- 同步更新与异步更新的差异；
- 梯度平均与模型平均的差异；
- 通信频率对训练效果的影响；
- 慢节点 straggler 对同步训练的影响；
- stale gradient 对异步训练的影响；
- 精度、通信量和训练时间之间的权衡。

### 4.2 Implemented Components

该项目主要实现了以下内容：

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

### 4.3 Core Ideas

#### Centralized SGD

单机基线，用于提供非分布式训练参考结果：

$$
w_{t+1} = w_t - \eta g_t
$$

其中，$w_t$ 表示第 $t$ 步的模型参数，$\eta$ 表示学习率，$g_t$ 表示当前 mini-batch 上计算得到的随机梯度。

#### Sync-SGD

所有 worker 同步计算梯度，server 聚合后统一更新：

$$
g_t = \sum_{k=1}^{K} \frac{n_k}{n} g_k(w_t)
$$

$$
w_{t+1} = w_t - \eta g_t
$$

其中，$K$ 表示 worker 数量，$n_k$ 表示第 $k$ 个 worker 的样本数，$n$ 表示总样本数，$g_k(w_t)$ 表示第 $k$ 个 worker 在全局模型 $w_t$ 上计算得到的梯度。

该方法梯度新鲜、更新稳定，但需要等待最慢 worker。

#### Local SGD / Model Averaging

每个 worker 本地训练若干步后再进行模型平均：

$$
w^{r+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_k^{r,E}
$$

其中，$r$ 表示通信轮次，$E$ 表示每轮通信之间的本地训练步数，$w_k^{r,E}$ 表示第 $k$ 个 worker 在第 $r$ 轮通信后经过 $E$ 步本地训练得到的模型参数。

该方法可以减少通信频率，但在 Non-IID 或 local steps 过大时可能出现 worker drift。

#### Async-SGD

worker 不等待其他 worker，计算完成后立即提交梯度：

$$
w_{t+1} = w_t - \eta g_k(w_{t-\tau})
$$

其中，$\tau$ 表示 staleness，即梯度延迟。异步更新可以缓解 straggler，但可能引入过期梯度导致训练不稳定。

### 4.4 Main Takeaways

通过该项目，可以理解：

- Sync-SGD 的核心问题是同步等待；
- Async-SGD 的核心问题是 stale gradient；
- Local SGD 的核心问题是通信频率与模型漂移之间的权衡；
- straggler 会显著影响同步训练效率；
- 分布式优化不能只看最终精度，还需要同时分析通信量和训练时间。

---

## 5. Project 2: PS vs AllReduce

### 5.1 Project Goal

`PS vs AllReduce` 是一个分布式训练系统架构对比实践项目，目标是实现并比较两类典型分布式训练架构：

1. Parameter Server；
2. AllReduce / DDP-style training。

该项目从系统实现角度理解分布式训练中的通信拓扑和性能瓶颈。

### 5.2 Implemented Components

该项目主要包含：

```text
Single-process baseline
Parameter Server implementation
AllReduce / DDP-style implementation
Training accuracy comparison
Training time comparison
Communication cost analysis
Scalability analysis
```

### 5.3 Parameter Server Architecture

Parameter Server 架构中，server 保存全局模型参数，worker 负责本地计算梯度并上传。

典型流程如下：

```text
1. Worker pulls global parameters from server
2. Worker computes gradients on local data
3. Worker pushes gradients to server
4. Server aggregates gradients and updates global model
5. Repeat
```

该架构的主要特点是实现直观，但 server 可能成为通信瓶颈。

从通信角度看，如果模型参数量为 $P$，每个参数使用 `float32`，即 4 bytes，则每一轮中 Parameter Server 至少需要处理 worker 上传的梯度通信量：

$$
\operatorname{UploadBytes} = K \times P \times 4
$$

如果同时考虑 server 向 worker 下发模型参数，则一次完整拉取与上传的通信量可以近似表示为：

$$
\operatorname{CommBytes} = 2KP \times 4
$$

其中，$K$ 表示 worker 数量。

### 5.4 AllReduce / DDP Architecture

AllReduce 架构中，每个 worker 都保存完整模型副本。反向传播后，各 worker 通过 AllReduce 同步梯度。

典型流程如下：

```text
1. Each worker holds a model replica
2. Each worker computes gradients on local mini-batch
3. Gradients are synchronized by AllReduce
4. Each worker updates its local model consistently
5. Repeat
```

该架构避免了中心 server 瓶颈，更接近现代分布式深度学习训练系统。

从优化目标看，AllReduce 需要让每个 worker 都得到全局梯度和：

$$
g = \sum_{k=1}^{K} g_k
$$

如果使用平均梯度更新，则每个 worker 使用：

$$
\bar{g} = \frac{1}{K} \sum_{k=1}^{K} g_k
$$

并执行本地一致的模型更新：

$$
w_{t+1} = w_t - \eta \bar{g}
$$

### 5.5 Main Takeaways

通过该项目，可以理解：

- Parameter Server 和 AllReduce 是两种典型分布式训练通信拓扑；
- Parameter Server 容易出现中心节点瓶颈；
- AllReduce 更适合去中心化梯度同步；
- DDP 的核心思想是每个进程持有模型副本，并通过 collective communication 同步梯度；
- 系统架构会直接影响训练吞吐量、通信开销和扩展性。

---

## 6. Project 3: IBing Reproduction

### 6.1 Paper

复现论文：

> Ruixing Zong, Jiapeng Zhang, Zhuo Tang, and Kenli Li.  
> **IBing: An Efficient Interleaved Bidirectional Ring All-Reduce Algorithm for Gradient Synchronization**.  
> ACM Transactions on Architecture and Code Optimization, 2025.

### 6.2 Project Goal

`IBing Reproduction` 复现了 IBing 论文中的核心通信调度思想，并实现了 Ring AllReduce 与 IBing 的单进程模拟、MPI 多进程版本、正确性测试、benchmark 和结果绘图。

本项目的目标不是完整复现论文中的大规模超算实验，而是验证：

- IBing 的通信调度公式是否可以实现；
- IBing 是否能够正确完成 AllReduce；
- IBing 是否将 Ring AllReduce 的通信步数从 $2(N-1)$ 降低到 $N-1$；
- 在 Windows 单机 MPI 环境下，IBing、Ring 和 MPI_Allreduce 的性能表现是否合理。

### 6.3 Implemented Components

该项目主要实现了：

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

### 6.4 Ring AllReduce

标准 Ring AllReduce 分为两个阶段：

```text
Reduce-Scatter: N - 1 steps
All-Gather:     N - 1 steps
```

因此总通信步数为：

$$
T_{\mathrm{Ring}} = 2(N - 1)
$$

其中，$N$ 表示参与 AllReduce 的 worker 数量。

### 6.5 IBing AllReduce

IBing 将 Ring 的单向通信改为交错双向通信。每个 step 中，每个 rank 同时：

```text
sends one chunk to the right and receives one chunk from the left
sends another chunk to the left and receives another chunk from the right
```

因此 IBing 的总通信步数为：

$$
T_{\mathrm{IBing}} = N - 1
$$

相比标准 Ring AllReduce，通信步数减少比例为：

$$
\operatorname{Reduction}
=
\frac{T_{\mathrm{Ring}} - T_{\mathrm{IBing}}}{T_{\mathrm{Ring}}}
\times 100\%
$$

将 $T_{\mathrm{Ring}} = 2(N-1)$ 和 $T_{\mathrm{IBing}} = N-1$ 代入可得：

$$
\operatorname{Reduction}
=
\frac{2(N-1)-(N-1)}{2(N-1)}
\times 100\%
=
50\%
$$

### 6.6 IBing Scheduling Formula

对于 rank $r$，step $i$，worker 总数 $N$，IBing 方向 1 的通信调度为：

$$
\operatorname{recv\_chunk}_1
=
(r - i - 1 + N) \bmod N
$$

$$
\operatorname{send\_chunk}_1
=
(r - i + N) \bmod N
$$

方向 2 的通信调度为：

$$
\operatorname{recv\_chunk}_2
=
(r + i + N + 2) \bmod N
$$

$$
\operatorname{send\_chunk}_2
=
(r + i + N + 1) \bmod N
$$

对于 rank $r$，左右邻居定义为：

$$
\operatorname{left}(r) = (r - 1 + N) \bmod N
$$

$$
\operatorname{right}(r) = (r + 1) \bmod N
$$

### 6.7 Experimental Results

本项目在 Windows 单机 MS-MPI 环境下完成测试。主要结果包括：

- Ring、IBing 和 MPI_Allreduce 均通过正确性测试；
- IBing 的通信步数确实为 $N-1$；
- Ring 的通信步数确实为 $2(N-1)$；
- IBing 在部分数据规模下快于 Ring；
- 在更多单机 Windows 多进程场景下，IBing 性能没有稳定优于 Ring；
- MPI_Allreduce 在中大规模数据下通常表现较好。

### 6.8 Result Interpretation

IBing 在本机环境下未稳定复现论文中的性能优势，主要原因包括：

- 实验环境为 Windows 单机多进程，而非多节点集群；
- 缺少真实双向物理链路；
- Python / mpi4py 调用开销较高；
- NumPy buffer copy 带来额外内存开销；
- 本机进程调度和内存带宽会影响通信时间；
- MPI_Allreduce 是 MPI 库内部高度优化的 collective 实现。

因此，本项目主要结论是：

```text
Correctness: Passed
Scheduling reproduction: Passed
Step-count reduction: Passed
MPI implementation: Passed
Benchmark pipeline: Passed
Paper-level performance reproduction: Not fully reproduced on Windows single-machine environment
```

---

## 7. Results and Reports

每个子项目均包含相应的结果记录、图表或报告。建议查看：

```text
project1_dml_bench/report/
project2_ps_vs_allreduce/report/
project3_ibing_reproduction/docs/
project3_ibing_reproduction/results/figures/
project3_ibing_reproduction/results/tables/
```

其中，IBing 复现项目已经生成：

```text
time_line_n3.png / .pdf
time_line_n4.png / .pdf
time_line_n5.png / .pdf
time_line_n8.png / .pdf
time_bar_n3.png / .pdf
time_bar_n4.png / .pdf
time_bar_n5.png / .pdf
time_bar_n8.png / .pdf
step_count_comparison.png / .pdf
ring_vs_ibing_speedup.png / .pdf
ring_vs_ibing_opt_rate.png / .pdf
mpi_speedup_summary.csv
mpi_benchmark_all.csv
```

---

## 8. Environment

主要使用环境如下：

```text
OS: Windows
Shell: PowerShell
Python: Conda environment
MPI: Microsoft MPI
Python packages:
  - numpy
  - pandas
  - matplotlib
  - torch
  - torchvision
  - mpi4py
```

不同子项目的具体依赖请查看各自目录中的 `requirements.txt`。

---

## 9. How to Run

### 9.1 DML-Bench

进入项目目录：

```powershell
cd project1_dml_bench
```

根据项目内 README 或脚本运行对应实验。例如：

```powershell
python run.py
```

如果项目使用模块化运行方式，可参考：

```powershell
python -m dmlbench.experiments.run_experiment
```

### 9.2 PS vs AllReduce

进入项目目录：

```powershell
cd project2_ps_vs_allreduce
```

运行单进程 baseline、Parameter Server 或 AllReduce / DDP 相关脚本：

```powershell
python single/train_single.py
python ps/train_ps.py
python ddp/train_ddp.py
```

具体命令以项目目录内脚本为准。

### 9.3 IBing Reproduction

进入项目目录：

```powershell
cd project3_ibing_reproduction
```

运行单进程模拟：

```powershell
python src/simulator/ibing_schedule.py --world_size 5
python src/simulator/ibing_sim.py --world_size 5 --verbose
python src/simulator/ring_sim.py --world_size 5 --verbose
```

运行 MPI 正确性测试：

```powershell
mpiexec -n 5 python src/mpi/ibing_mpi.py --chunk_size 4 --verbose
mpiexec -n 5 python src/mpi/ring_mpi.py --chunk_size 4 --verbose
python src/mpi/test_mpi_correctness.py
```

运行 benchmark：

```powershell
mpiexec -n 5 python src/mpi/benchmark.py --algo all --data_sizes_mb 1 10 50 --repeat 30 --warmup 5
```

批量运行 benchmark：

```powershell
python scripts/run_benchmark.py --world_sizes 3 4 5 8 --data_sizes_mb 1 10 50 --repeat 30 --warmup 5
```

生成图表：

```powershell
python scripts/plot_results.py --formats png pdf
```

---

## 10. Current Limitations

本仓库目前仍存在以下不足：

1. 实验主要在 Windows 单机环境下完成，缺少真实多节点网络测试；
2. MPI 版本使用 Python + mpi4py 实现，性能无法代表 C/C++ 或 NCCL 级别通信库；
3. IBing 复现重点在核心机制与通信步数，没有完整复现论文中的超算集群实验；
4. 部分实践项目更偏学习型实现，尚未达到工业级工程质量；
5. 后续仍需要在 Linux / 多机 / GPU 环境下进一步测试。

---

## 11. Future Work

后续可以继续从以下方向扩展：

1. 在 Linux / WSL / OpenMPI 环境下重新测试 IBing；
2. 在多机环境中测试 Ring、IBing 和 MPI_Allreduce；
3. 使用 C++ / MPI 重写 IBing 核心通信逻辑，降低 Python 层开销；
4. 将 IBing 接入真实分布式训练任务；
5. 继续复现 topology-aware AllReduce 或通信压缩类论文；
6. 深入研究分布式训练系统 benchmark，包括 DDP、FSDP 和通信瓶颈分析；
7. 结合联邦学习基础，进一步研究通信压缩、Non-IID 优化和个性化联邦学习。

---

## 12. Learning Summary

通过三个项目的实践，我对分布式机器学习形成了以下理解：

- 分布式训练的核心不是简单地增加 worker，而是在计算、通信和同步之间做权衡；
- Sync-SGD 稳定但容易受慢节点影响；
- Async-SGD 能减少等待，但会引入 stale gradient；
- Local SGD 能降低通信频率，但可能导致模型漂移；
- Parameter Server 和 AllReduce 代表了两类典型分布式训练架构；
- Ring AllReduce 是理解现代梯度同步通信的重要基础；
- IBing 通过交错双向通信减少通信步数，体现了通信调度优化的思想；
- 实验结果必须结合运行环境分析，不能只看是否达到论文性能数字。

---

## 13. References

[1] Ruixing Zong, Jiapeng Zhang, Zhuo Tang, and Kenli Li. *IBing: An Efficient Interleaved Bidirectional Ring All-Reduce Algorithm for Gradient Synchronization*. ACM Transactions on Architecture and Code Optimization, 2025.

[2] Message Passing Interface Forum. *MPI: A Message-Passing Interface Standard*.

[3] mpi4py documentation. *MPI for Python*.

[4] PyTorch documentation. *Distributed Data Parallel*.

[5] NumPy documentation. *NumPy array programming library*.
