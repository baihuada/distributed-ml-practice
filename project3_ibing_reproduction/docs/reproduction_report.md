# IBing 论文复现报告

## 1. 复现项目概述

本文对论文 **IBing: An Efficient Interleaved Bidirectional Ring All-Reduce Algorithm for Gradient Synchronization** 进行了阶段性复现。该论文提出了一种交错双向 Ring All-Reduce 算法 IBing，目标是在分布式机器学习梯度同步过程中减少 Ring All-Reduce 的通信步数，提高梯度同步效率。

本次复现并不直接复现论文中的大规模超算集群实验，而是按照由浅入深的方式完成以下内容：

1. 单进程通信调度模拟；
2. 单进程 IBing 数据流正确性模拟；
3. 单进程 Ring All-Reduce baseline；
4. IBing 与 Ring 的通信步数理论对比；
5. MPI 多进程版 IBing 实现；
6. MPI 多进程版 Ring baseline 实现；
7. MPI 正确性批量测试；
8. MPI 通信时间 benchmark；
9. 批量 benchmark 脚本；
10. 结果绘图与性能分析。

本复现的核心目标是验证：

- IBing 的通信调度公式是否可实现；
- IBing 是否能够正确完成 All-Reduce；
- IBing 是否将 Ring All-Reduce 的通信步数从 \(2(N-1)\) 降低到 \(N-1\)；
- 在 Windows 单机 MPI 环境下，IBing、Ring 和 MPI_Allreduce 的性能表现是否合理。

---

## 2. 原论文核心思想

### 2.1 All-Reduce 问题定义

在数据并行分布式训练中，假设共有 \(N\) 个 worker。每个 worker 持有本地梯度向量。为了同步模型参数，需要对所有 worker 的梯度执行 All-Reduce 操作。

假设第 \(r\) 个 worker 的数据为：

$$
X_r = [x_{r,0}, x_{r,1}, \dots, x_{r,N-1}]
$$

其中数据被切分为 \(N\) 个 chunk。All-Reduce 的目标是让每个 worker 最终都得到：

$$
Y =
\left[
\sum_{r=0}^{N-1} x_{r,0},
\sum_{r=0}^{N-1} x_{r,1},
\dots,
\sum_{r=0}^{N-1} x_{r,N-1}
\right]
$$

也就是说，最终任意 worker \(r\) 都应满足：

$$
X_r = Y
$$

---

### 2.2 标准 Ring All-Reduce

标准 Ring All-Reduce 分为两个阶段：

1. Reduce-Scatter；
2. All-Gather。

每个 worker 只与左右邻居通信。对于 rank \(r\)，其左右邻居定义为：

$$
left(r) = (r - 1 + N) \bmod N
$$

$$
right(r) = (r + 1) \bmod N
$$

在标准 Ring 中，通信方向固定为单向环：

$$
left(r) \rightarrow r \rightarrow right(r)
$$

Reduce-Scatter 阶段需要：

$$
N-1
$$

步。All-Gather 阶段也需要：

$$
N-1
$$

步。因此标准 Ring All-Reduce 的总通信步数为：

$$
T_{\text{Ring}} = 2(N-1)
$$

---

### 2.3 IBing 的核心思想

IBing 的核心思想是将标准 Ring 的单向通信改为交错双向通信。每个 step 中，每个 worker 同时执行两个方向的数据交换：

- 向右邻居发送一个 chunk，并从左邻居接收一个 chunk；
- 向左邻居发送另一个 chunk，并从右邻居接收另一个 chunk。

因此，IBing 每个 step 能够同时推进两个方向的数据传播。论文中给出的核心调度公式如下。

对于 rank \(r\)，step \(i\)，worker 总数 \(N\)，方向 1 的调度为：

$$
\mathrm{recv\_chunk}_1 = (r - i - 1 + N) \bmod N
$$

$$
\mathrm{send\_chunk}_1 = (r - i + N) \bmod N
$$

方向 2 的调度为：

$$
\mathrm{recv\_chunk}_2 = (r + i + N + 2) \bmod N
$$

$$
\mathrm{send\_chunk}_2 = (r + i + N + 1) \bmod N
$$

IBing 的总通信步数为：

$$
T_{\text{IBing}} = N-1
$$

因此，IBing 相比标准 Ring All-Reduce 的通信步数减少量为：

$$
\Delta T = T_{\text{Ring}} - T_{\text{IBing}}
$$

代入可得：

$$
\Delta T = 2(N-1) - (N-1) = N-1
$$

通信步数减少比例为：

$$
\text{Reduction}
=
\frac{T_{\text{Ring}} - T_{\text{IBing}}}{T_{\text{Ring}}}
\times 100\%
$$

即：

$$
\text{Reduction}
=
\frac{N-1}{2(N-1)}
\times 100\%
=
50\%
$$

所以从理论通信步数角度看，IBing 将 Ring All-Reduce 的通信步数减少了一半。

---

## 3. 复现环境

本次复现环境为个人 Windows 笔记本电脑，具体环境如下：

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows |
| Shell | PowerShell |
| Python 环境 | Conda 虚拟环境 `FedEnv` |
| MPI 实现 | Microsoft MPI |
| Python MPI 接口 | mpi4py |
| 数值计算库 | NumPy |
| 数据处理与绘图 | pandas, matplotlib |
| 运行方式 | 单机多进程 MPI |

需要说明的是，本次实验环境与论文中的大规模多节点超算集群环境存在显著差异。本文实验主要用于验证算法逻辑、通信调度、正确性和基本性能趋势，不能作为论文原始性能结果的严格复现。

---

## 4. 项目目录结构

本次复现项目的主要目录结构如下：

```text
IBing论文复现/
├── docs/
│   └── reproduction_report.md
├── results/
│   ├── raw/
│   │   ├── mpi_benchmark_n3_all_float32.csv
│   │   ├── mpi_benchmark_n4_all_float32.csv
│   │   ├── mpi_benchmark_n5_all_float32.csv
│   │   └── mpi_benchmark_n8_all_float32.csv
│   ├── tables/
│   │   ├── mpi_benchmark_all.csv
│   │   ├── mpi_benchmark_run_log.csv
│   │   └── mpi_speedup_summary.csv
│   └── figures/
│       ├── time_line_n3.png
│       ├── time_line_n4.png
│       ├── time_line_n5.png
│       ├── time_line_n8.png
│       ├── time_bar_n3.png
│       ├── time_bar_n4.png
│       ├── time_bar_n5.png
│       ├── time_bar_n8.png
│       ├── step_count_comparison.png
│       ├── ring_vs_ibing_speedup.png
│       └── ring_vs_ibing_opt_rate.png
├── scripts/
│   ├── run_benchmark.py
│   └── plot_results.py
└── src/
    ├── mpi/
    │   ├── ibing_mpi.py
    │   ├── ring_mpi.py
    │   ├── test_mpi_correctness.py
    │   └── benchmark.py
    ├── simulator/
    │   ├── ibing_schedule.py
    │   ├── ibing_sim.py
    │   ├── ring_sim.py
    │   └── compare_steps.py
    └── tests/
        └── test_correctness.py
```

------

## 5. 单进程模拟部分

### 5.1 IBing 调度模拟

首先实现了：

```text
src/simulator/ibing_schedule.py
```

该文件用于根据论文公式生成 IBing 在不同 rank、不同 step 下的通信调度表。

对于每个 rank，程序输出：

- 左邻居；
- 右邻居；
- 当前 step；
- 当前阶段；
- 方向 1 的 send chunk 和 recv chunk；
- 方向 2 的 send chunk 和 recv chunk。

例如 \(N=5\) 时，IBing 总步数为：

$$
N-1=4
$$
其中前 2 步为 Reduce-Scatter，后 2 步为 All-Gather。

该文件主要验证论文中的通信调度公式是否能够正确落地为程序实现。

------

### 5.2 IBing 单进程数据流模拟

随后实现了：

```text
src/simulator/ibing_sim.py
```

该文件在单进程中模拟多个 worker 的数据流。每个 worker 的初始数据为：

$$
\mathrm{workers}[\mathrm{rank}][\mathrm{chunk}] = \mathrm{rank} \times 10 + \mathrm{chunk}
$$
例如 \(N=5\) 时，初始数据为：

```text
rank 0: [0, 1, 2, 3, 4]
rank 1: [10, 11, 12, 13, 14]
rank 2: [20, 21, 22, 23, 24]
rank 3: [30, 31, 32, 33, 34]
rank 4: [40, 41, 42, 43, 44]
```

理论 All-Reduce 结果为：

```text
[100, 105, 110, 115, 120]
```

运行结果显示，所有 rank 最终均得到：

```text
[100, 105, 110, 115, 120]
```

说明 IBing 的单进程数据流模拟正确。

------

### 5.3 Ring 单进程 baseline

随后实现了：

```text
src/simulator/ring_sim.py
```

该文件实现标准 Ring All-Reduce 的单进程模拟版本。其通信逻辑分为：

- Reduce-Scatter 阶段；
- All-Gather 阶段。

标准 Ring 的 Reduce-Scatter 阶段第 \(i\) 步中，rank \(r\) 的发送和接收 chunk 为：

$$
\mathrm{send\_chunk} = (r - i + N) \bmod N
$$
$$
\mathrm{recv\_chunk} = (r - i - 1 + N) \bmod N
$$
All-Gather 阶段第 \(i\) 步中，rank \(r\) 的发送和接收 chunk 为：

$$
\mathrm{send\_chunk} = (r - i + 1 + N) \bmod N
$$
$$
\mathrm{recv\_chunk} = (r - i + N) \bmod N
$$
该文件作为 IBing 的 baseline，用于后续正确性测试和性能对比。

------

### 5.4 单进程正确性批量测试

为了避免手动逐个测试，进一步实现了：

```text
src/tests/test_correctness.py
```

该文件批量测试 IBing 和 Ring 在多个 world size 下的正确性。测试的典型规模包括：

```text
N = 3, 4, 5, 6, 7, 8
```

正确性检查包括：

1. 所有 rank 是否得到理论 All-Reduce 结果；
2. 所有 rank 的最终结果是否一致；
3. IBing 通信步数是否为 \(N-1\)；
4. Ring 通信步数是否为 \(2(N-1)\)。

测试结果全部通过。

------

### 5.5 通信步数对比

进一步实现了：

```text
src/simulator/compare_steps.py
```

该文件专门比较 Ring 和 IBing 的理论通信步数。

对于不同 \(N\)，理论步数如下：

| \(N\) | Ring steps | IBing steps | Reduction |
| ---- | ---------- | ----------- | --------- |
| 3    | 4          | 2           | 50%       |
| 4    | 6          | 3           | 50%       |
| 5    | 8          | 4           | 50%       |
| 8    | 14         | 7           | 50%       |

该结果与论文理论一致。

------

## 6. MPI 多进程实现

### 6.1 MPI 版 IBing

实现文件为：

```text
src/mpi/ibing_mpi.py
```

该文件将单进程 IBing 模拟扩展为真正的 MPI 多进程版本。每个 MPI 进程对应一个 worker，每个进程只知道：

- 当前 rank；
- world size；
- 左邻居；
- 右邻居；
- 本地 chunks。

核心通信使用 mpi4py 的非阻塞通信接口：

```python
comm.Irecv(...)
comm.Isend(...)
MPI.Request.Waitall(...)
```

IBing 每一步中执行两个方向的通信：

方向 1：

```text
left -> current -> right
```

方向 2：

```text
right -> current -> left
```

Reduce-Scatter 阶段执行加法：

$$
X_r[\mathrm{recv\_chunk}] \leftarrow X_r[\mathrm{recv\_chunk}] + received
$$
All-Gather 阶段直接保存：

$$
X_r[\mathrm{recv\_chunk}] \leftarrow received
$$
测试命令示例：

```powershell
mpiexec -n 5 python src/mpi/ibing_mpi.py --chunk_size 4 --verbose
```

测试结果显示：

```text
global_correct  : True
[PASS] MPI IBing correctness check passed.
```

------

### 6.2 MPI 版 Ring

实现文件为：

```text
src/mpi/ring_mpi.py
```

该文件实现标准 Ring All-Reduce 的 MPI 多进程版本。每一步中，每个 rank：

1. 从左邻居接收；
2. 向右邻居发送；
3. 在 Reduce-Scatter 阶段执行加法；
4. 在 All-Gather 阶段直接保存。

测试命令示例：

```powershell
mpiexec -n 5 python src/mpi/ring_mpi.py --chunk_size 4 --verbose
```

测试结果显示：

```text
global_correct       : True
[PASS] MPI Ring correctness check passed.
```

------

### 6.3 MPI 正确性批量测试

实现文件为：

```text
src/mpi/test_mpi_correctness.py
```

该脚本通过 `subprocess` 自动调用真实 MPI 命令，对 Ring 和 IBing 进行批量正确性测试。例如：

```powershell
python src/mpi/test_mpi_correctness.py --world_sizes 3 4 5 8 --chunk_sizes 4 1024
```

该脚本本身不是 MPI 程序，不需要使用 `mpiexec` 启动。它内部自动构造如下命令：

```powershell
mpiexec -n 5 python src/mpi/ibing_mpi.py --chunk_size 1024 --check
mpiexec -n 5 python src/mpi/ring_mpi.py --chunk_size 1024 --check
```

最终结果全部通过，说明 MPI 版 IBing 和 Ring 的多进程实现均正确。

------

## 7. MPI benchmark 实验

### 7.1 benchmark 文件

实现文件为：

```text
src/mpi/benchmark.py
```

该文件用于测试以下三种 All-Reduce 实现的通信时间：

| 算法          | 说明                            |
| ------------- | ------------------------------- |
| ring          | 手写 MPI 版标准 Ring All-Reduce |
| ibing         | 手写 MPI 版 IBing All-Reduce    |
| mpi_allreduce | MPI 库自带 Allreduce            |

其中，`mpi_allreduce` 调用的是 MPI 原生集合通信接口：

```python
comm.Allreduce(send_buf, recv_buf, op=MPI.SUM)
```

它不是手写 Ring，也不是手写 IBing，而是 MPI 库内部高度优化的 All-Reduce 实现。

------

### 7.2 数据规模设置

每个 rank 的数据形状为：

$$
[N,\ \mathrm{chunk\_size}]
$$
其中：

- \(N\)：MPI 进程数；
- \(chunk\_size\)：每个 chunk 的元素数量。

每个 rank 的总数据大小为：

$$
N \times \mathrm{chunk\_size} \times \mathrm{bytes}(\mathrm{dtype})
$$
本次实验使用 `float32`，每个元素 4 字节。

实验测试的数据规模为：

```text
1 MiB, 10 MiB, 50 MiB
```

测试的进程数为：

```text
N = 3, 4, 5, 8
```

每组实验设置：

```text
warmup = 5
repeat = 30
dtype = float32
```

------

### 7.3 计时方式

每轮通信前后均使用：

```python
comm.Barrier()
```

进行进程同步。

单轮通信耗时使用：

```python
time.perf_counter()
```

记录本地耗时。由于分布式通信的总完成时间取决于最慢的 rank，因此使用：

```python
comm.allreduce(elapsed, op=MPI.MAX)
```

取所有 rank 中最大的耗时作为该轮通信耗时。

最终统计：

- mean；
- std；
- min；
- max。

------

### 7.4 批量 benchmark

为了自动测试多个 world size，实现了：

```text
scripts/run_benchmark.py
```

该脚本会自动调用：

```powershell
mpiexec -n 3 python src/mpi/benchmark.py ...
mpiexec -n 4 python src/mpi/benchmark.py ...
mpiexec -n 5 python src/mpi/benchmark.py ...
mpiexec -n 8 python src/mpi/benchmark.py ...
```

并生成：

```text
results/raw/mpi_benchmark_n3_all_float32.csv
results/raw/mpi_benchmark_n4_all_float32.csv
results/raw/mpi_benchmark_n5_all_float32.csv
results/raw/mpi_benchmark_n8_all_float32.csv
```

同时合并为：

```text
results/tables/mpi_benchmark_all.csv
```

------

## 8. 绘图与结果汇总

### 8.1 绘图脚本

实现文件为：

```text
scripts/plot_results.py
```

该文件读取：

```text
results/tables/mpi_benchmark_all.csv
```

并生成以下图表：

```text
results/figures/time_line_n3.png
results/figures/time_line_n4.png
results/figures/time_line_n5.png
results/figures/time_line_n8.png

results/figures/time_bar_n3.png
results/figures/time_bar_n4.png
results/figures/time_bar_n5.png
results/figures/time_bar_n8.png

results/figures/step_count_comparison.png
results/figures/ring_vs_ibing_speedup.png
results/figures/ring_vs_ibing_opt_rate.png
```

同时生成 Ring vs IBing 的加速比汇总表：

```text
results/tables/mpi_speedup_summary.csv
```

------

### 8.2 绘图运行结果

绘图命令为：

```powershell
python scripts/plot_results.py --formats png pdf
```

运行结果显示所有图像均成功生成，并输出：

```text
[PASS] Plot generation completed.
```

因此，本次复现已经完成从算法实现、正确性验证、性能 benchmark 到结果绘图的完整流程。

------

## 9. 实验结果

### 9.1 Ring vs IBing 加速比汇总

最终绘图脚本输出的 Ring vs IBing 加速比结果如下：

| N    | Data(MiB) | Ring(ms) | IBing(ms) | Speedup | OptRate  |
| ---- | --------- | -------- | --------- | ------- | -------- |
| 3    | 1         | 0.893    | 0.746     | 1.197x  | 16.44%   |
| 3    | 10        | 12.827   | 14.099    | 0.910x  | -9.92%   |
| 3    | 50        | 75.730   | 73.974    | 1.024x  | 2.32%    |
| 4    | 1         | 0.302    | 0.657     | 0.460x  | -117.54% |
| 4    | 10        | 21.446   | 22.786    | 0.941x  | -6.25%   |
| 4    | 50        | 124.324  | 110.929   | 1.121x  | 10.77%   |
| 5    | 1         | 0.701    | 0.838     | 0.837x  | -19.52%  |
| 5    | 10        | 26.783   | 31.901    | 0.840x  | -19.11%  |
| 5    | 50        | 154.591  | 145.165   | 1.065x  | 6.10%    |
| 8    | 1         | 0.578    | 1.070     | 0.541x  | -84.93%  |
| 8    | 10        | 47.817   | 53.922    | 0.887x  | -12.77%  |
| 8    | 50        | 251.357  | 258.516   | 0.972x  | -2.85%   |

其中：

$$
\text{Speedup} = \frac{T_{\text{Ring}}}{T_{\text{IBing}}}
$$
$$
\text{OptRate} =
\frac{T_{\text{Ring}} - T_{\text{IBing}}}{T_{\text{Ring}}}
\times 100\%
$$
当 Speedup \(>1\) 或 OptRate \(>0\) 时，表示 IBing 快于 Ring。

当 Speedup \(<1\) 或 OptRate \(<0\) 时，表示 IBing 慢于 Ring。

------

### 9.2 结果现象

从最终结果可以观察到：

1. 在部分数据规模下，IBing 快于 Ring。
   - \(N=3\), 1 MiB：IBing 优化 16.44%；
   - \(N=3\), 50 MiB：IBing 优化 2.32%；
   - \(N=4\), 50 MiB：IBing 优化 10.77%；
   - \(N=5\), 50 MiB：IBing 优化 6.10%。
2. 在更多情况下，IBing 在 Windows 单机环境中慢于 Ring。
   - \(N=4\), 1 MiB：IBing 退化 117.54%；
   - \(N=8\), 1 MiB：IBing 退化 84.93%；
   - \(N=5\), 10 MiB：IBing 退化 19.11%。
3. 随着数据规模从 1 MiB 增加到 50 MiB，IBing 的退化幅度通常有所减小，部分情况下开始出现优化。
4. MPI_Allreduce 在中大数据规模下通常表现较好，说明 MPI 库内部 collective 实现具有较强工程优化。

------

## 10. 性能结果分析

### 10.1 为什么 IBing 步数减半但不一定更快

IBing 在理论上将通信步数从：

$$
2(N-1)
$$
减少到：

$$
N-1
$$
但通信步数减半并不意味着真实运行时间一定减半。原因是 IBing 每一步的通信负担更重。

标准 Ring 每一步通常包括：

```text
1 个 Irecv
1 个 Isend
1 个 Waitall
1 次本地更新
```

IBing 每一步包括：

```text
2 个 Irecv
2 个 Isend
1 个 Waitall
2 次本地更新
```

因此，IBing 的 step 数量较少，但每个 step 的操作复杂度更高。

在真实多节点网络中，IBing 能够利用双向链路并发通信，从而减少通信建立开销并提升带宽利用率。但在 Windows 单机多进程环境下，所有进程共享同一台机器的 CPU、内存和 MPI 本地通信路径，无法充分体现双向网络链路的收益。

------

### 10.2 Windows 单机环境的影响

本次实验使用的是 Windows 单机 MS-MPI 环境。该环境与论文中的多节点超算环境差异较大。

在单机环境中，主要影响通信时间的因素包括：

- Python 层函数调用开销；
- mpi4py 调用开销；
- NumPy buffer 拷贝开销；
- 进程调度开销；
- 本机共享内存通信路径；
- Windows 后台进程干扰；
- CPU 调频和内存带宽波动。

IBing 每一步需要两个发送 buffer 和两个接收 buffer，并且在代码中为了保证非阻塞通信正确性使用了 `.copy()`。这会引入额外内存拷贝开销。

因此，在本机环境中，IBing 的双向通信优势没有充分体现，反而可能因为每一步通信和内存操作更复杂而导致性能退化。

------

### 10.3 MPI_Allreduce 为什么表现较好

MPI_Allreduce 是 MPI 库提供的原生集合通信接口。它的语义是所有进程参与规约，并且所有进程都得到规约结果。

与手写 Ring 和 IBing 不同，MPI_Allreduce 的内部实现由 MPI 库根据：

- 进程数；
- 数据大小；
- 数据类型；
- 拓扑结构；
- 是否为单机通信；
- MPI 实现类型；

自动选择优化算法。

它可能采用：

- tree-based reduce + broadcast；
- recursive doubling；
- Rabenseifner algorithm；
- 分段 ring；
- 共享内存优化；
- 流水线算法；
- 向量化内存拷贝。

因此，MPI_Allreduce 在中大数据规模下表现较好是合理的。

本复现中的手写 Ring 和 IBing 主要用于验证算法机制和通信调度，不是工业级 collective 通信库实现。

------

## 11. 与论文结果差异说明

论文中的实验环境为大规模多节点高性能计算环境，而本复现使用 Windows 单机多进程环境。

论文关注的核心问题是：在多节点分布式训练中，随着节点数增加，All-Reduce 的通信建立开销和跨节点数据传输开销逐渐成为瓶颈。IBing 通过交错双向通信减少通信步数，从而降低通信开销。

而本复现环境中：

- 没有真实多节点网络；
- 没有独立双向物理链路；
- 没有超算互连网络；
- 进程位于同一台机器；
- 本地内存拷贝和 Python 调用开销占比更高。

因此，本复现不能严格复现论文中的性能提升比例。

但是，本复现已经成功验证：

1. IBing 调度公式可以正确实现；
2. IBing 能够正确完成 All-Reduce；
3. IBing 的理论通信步数确实为 \(N-1\)；
4. Ring 的理论通信步数确实为 \(2(N-1)\)；
5. IBing 相比 Ring 的通信步数减少 50%；
6. MPI 多进程版本能够正确运行；
7. benchmark 和绘图流程完整可复现。

------

## 12. 复现结论

本次复现完成了 IBing 论文核心算法的阶段性复现。

从正确性角度看，复现成功。单进程模拟和 MPI 多进程实现均通过正确性测试，所有 rank 最终均得到理论 All-Reduce 结果。

从通信调度角度看，复现成功。IBing 的交错双向通信调度公式被成功实现，并且能够按照论文设计完成 Reduce-Scatter 和 All-Gather。

从通信步数角度看，复现成功。标准 Ring All-Reduce 的通信步数为：

$$
T_{\text{Ring}} = 2(N-1)
$$
IBing 的通信步数为：

$$
T_{\text{IBing}} = N-1
$$
因此，IBing 相比 Ring 的通信步数减少：

$$
50\%
$$
从性能角度看，本机实验结果合理但未完全达到论文预期。IBing 在部分数据规模下快于 Ring，但在更多 Windows 单机多进程场景下存在性能退化。这主要是由于实验环境与论文大规模多节点环境不同，以及 Python/mpi4py/NumPy buffer copy 带来的额外开销。

综上，本次复现完成了 IBing 的核心机制复现和实验链路构建，验证了论文提出的通信步数优化思想，但未在个人 Windows 单机环境下稳定复现论文中的通信时间优化幅度。

------

## 13. 后续改进方向

后续可以从以下几个方向继续完善复现：

### 13.1 在 Linux 或 WSL 环境下重新测试

Windows MS-MPI 单机环境存在较多系统调度和通信栈影响。后续可以在 Linux 或 WSL 中使用 OpenMPI 重新运行：

```bash
mpirun -np 5 python src/mpi/benchmark.py --algo all --data_sizes_mb 1 10 50 --repeat 100 --warmup 10
```

对比 Windows 与 Linux 的差异。

------

### 13.2 在多机环境中测试

IBing 的优势更依赖真实网络通信环境。后续如果具备两台或多台机器，可以使用多机 MPI 测试：

```bash
mpirun -np 8 --hostfile hosts.txt python src/mpi/benchmark.py --algo all --data_sizes_mb 1 10 50
```

这样可以更接近论文中的多节点通信场景。

------

### 13.3 降低 Python 层开销

当前实现为了清晰和正确性，保留了较多 Python 层逻辑和 buffer 拷贝。后续可尝试：

- 复用 send buffer；
- 复用 recv buffer；
- 减少每轮临时对象创建；
- 使用更连续的内存布局；
- 减少 `.copy()` 次数；
- 使用 Cython、Numba 或 C++/MPI 实现核心通信。

------

### 13.4 接入分布式训练任务

当前实验只测试 All-Reduce 通信本身。后续可将 IBing 接入简单分布式训练流程，例如：

- Logistic Regression；
- MLP；
- CNN on MNIST/CIFAR；
- ResNet 小规模训练。

这样可以进一步评估 IBing 对端到端训练时间的影响。

------

### 13.5 对比更多 All-Reduce 算法

后续可进一步实现或调用：

- Recursive Doubling；
- Rabenseifner Allreduce；
- Tree-based Allreduce；
- NCCL Ring；
- Hierarchical Ring；
- 2D-Torus Ring。

这样可以使复现报告更加完整。

------

## 14. 运行命令汇总

### 14.1 单进程调度测试

```powershell
python src/simulator/ibing_schedule.py --world_size 5
```

### 14.2 IBing 单进程模拟

```powershell
python src/simulator/ibing_sim.py --world_size 5 --verbose
python src/simulator/ibing_sim.py --world_size 5 --trace
```

### 14.3 Ring 单进程模拟

```powershell
python src/simulator/ring_sim.py --world_size 5 --verbose
python src/simulator/ring_sim.py --world_size 5 --trace
```

### 14.4 单进程正确性批量测试

```powershell
python src/tests/test_correctness.py
```

### 14.5 通信步数对比

```powershell
python src/simulator/compare_steps.py
```

### 14.6 MPI 版 IBing 测试

```powershell
mpiexec -n 5 python src/mpi/ibing_mpi.py --chunk_size 4 --verbose
```

### 14.7 MPI 版 Ring 测试

```powershell
mpiexec -n 5 python src/mpi/ring_mpi.py --chunk_size 4 --verbose
```

### 14.8 MPI 正确性批量测试

```powershell
python src/mpi/test_mpi_correctness.py
```

### 14.9 单次 MPI benchmark

```powershell
mpiexec -n 5 python src/mpi/benchmark.py --algo all --data_sizes_mb 1 10 50 --repeat 30 --warmup 5
```

### 14.10 批量 benchmark

```powershell
python scripts/run_benchmark.py --world_sizes 3 4 5 8 --data_sizes_mb 1 10 50 --repeat 30 --warmup 5
```

### 14.11 绘图

```powershell
python scripts/plot_results.py --formats png pdf
```

------

## 15. 最终评价

本次复现已经形成了一个完整、可运行、可验证、可扩展的 IBing 复现项目。

复现结果可以概括为：

```text
算法正确性：通过
通信调度复现：通过
单进程模拟：通过
MPI 多进程实现：通过
通信步数理论验证：通过
benchmark 流程：通过
绘图流程：通过
论文性能优势：在 Windows 单机环境下未稳定复现
```

因此，本项目达到了阶段性复现目标。后续若希望进一步接近论文实验结果，应优先迁移到 Linux 多节点 MPI 环境，并尽量降低 Python 层和本机共享内存通信带来的额外开销。

------

## 参考文献

[1] Ruixing Zong, Jiapeng Zhang, Zhuo Tang, and Kenli Li. *IBing: An Efficient Interleaved Bidirectional Ring All-Reduce Algorithm for Gradient Synchronization*. ACM Transactions on Architecture and Code Optimization, 22(1), Article 35, 2025.

[2] Message Passing Interface Forum. *MPI: A Message-Passing Interface Standard*.

[3] mpi4py documentation. *MPI for Python*.

[4] NumPy documentation. *NumPy array programming library*.