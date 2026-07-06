# PS vs AllReduce：两类分布式训练系统实现与对比实践规划

## 一、当前阶段总目标

你现在第二个实践项目做 **PS vs AllReduce Core 版**。第一个 DML-Bench 项目已经完成了单机模拟多 worker 的分布式优化机制，包括 Centralized SGD、Sync-SGD、Local SGD、Async-SGD、straggler 和 staleness 分析。这个项目的目标是进一步从“算法模拟”进入“真实多进程分布式训练系统”。

本项目的核心不是继续增加优化算法，而是把《分布式机器学习算法、理论和实践》中关于 **Parameter Server、AllReduce、数据并行、梯度同步、通信瓶颈、系统吞吐量** 的内容真正落到代码实现中。

| 项目 | 内容 |
|---|---|
| 项目名称 | PS vs AllReduce：两类分布式训练系统实现与对比 |
| 第一版定位 | 单机多进程分布式训练，不做真实多机集群 |
| 数据集 | MNIST / Fashion-MNIST |
| 模型 | Logistic Regression / 简单 MLP |
| 核心系统 | Single Process、Parameter Server、PyTorch DDP / AllReduce |
| 核心实验 | 训练精度、epoch time、samples/s、通信量估算、worker 数扩展 |
| 暂不实现 | 多机部署、NCCL 深度优化、复杂 CNN、CIFAR-10、大模型训练 |
| 最终输出 | 代码仓库 + CSV 日志 + 曲线图 + 系统对比表 + 小报告 |

---

## 二、建议仓库结构

建议把第二个项目单独建一个目录，不要混在 DML-Bench 里面。DML-Bench 是算法模拟项目，PS vs AllReduce 是真实多进程系统项目，两者定位不同。

| 路径 | 作用 |
|---|---|
| `ps-vs-allreduce/` | 项目总目录 |
| `common/` | 公共模块：数据加载、模型、日志、随机种子、评估函数 |
| `single/` | 单进程训练 baseline |
| `ps/` | Parameter Server 多进程实现 |
| `ddp/` | PyTorch DDP / AllReduce 实现 |
| `utils/` | 通信量估算、画图、结果汇总 |
| `configs/` | 实验配置文件 |
| `scripts/` | 一键运行脚本 |
| `results/raw/` | 保存 CSV 原始结果 |
| `results/figures/` | 保存实验图 |
| `results/tables/` | 保存汇总表 |
| `report/` | 实验报告 |

推荐目录：

```text
ps-vs-allreduce/
├── README.md
├── requirements.txt
├── configs/
├── scripts/
├── common/
│   ├── __init__.py
│   ├── seed.py
│   ├── datasets.py
│   ├── models.py
│   ├── logger.py
│   └── metrics.py
├── single/
│   ├── __init__.py
│   └── train_single.py
├── ps/
│   ├── __init__.py
│   ├── train_ps.py
│   ├── server.py
│   └── worker.py
├── ddp/
│   ├── __init__.py
│   └── train_ddp.py
├── utils/
│   ├── __init__.py
│   ├── comm.py
│   ├── plot.py
│   └── summarize.py
├── results/
│   ├── raw/
│   ├── figures/
│   └── tables/
└── report/
    └── ps_vs_allreduce_report.md
```

---

## 三、总体实践步骤表

### 第 1 阶段：项目骨架与单进程 baseline

| 步骤 | 目标 | 需要实现的文件 | 具体任务 | 验收标准 |
|---|---|---|---|---|
| 1.1 | 建立项目结构 | 全部基础文件夹 | 创建 `common/`、`single/`、`ps/`、`ddp/`、`utils/`、`results/`、`report/` | 能正常识别 Python 包 |
| 1.2 | 配置环境 | `requirements.txt` | 安装 `torch`、`torchvision`、`numpy`、`pandas`、`matplotlib`、`tqdm`、`pyyaml` | `python -c "import torch"` 不报错 |
| 1.3 | 固定随机种子 | `common/seed.py` | 统一设置 `random`、`numpy`、`torch` 种子 | 多次运行结果基本一致 |
| 1.4 | 加载 MNIST | `common/datasets.py` | 下载并返回 train/test dataloader | 能打印训练集、测试集大小 |
| 1.5 | 实现模型 | `common/models.py` | 实现 Logistic Regression 和 MLP | forward 输出维度为 `[batch_size, 10]` |
| 1.6 | 实现评估函数 | `common/metrics.py` | 计算 loss、accuracy、samples/s | 能正确返回测试准确率 |
| 1.7 | 实现单进程训练 | `single/train_single.py` | 普通 PyTorch 训练循环 | MNIST 准确率能超过 90% |
| 1.8 | 保存日志 | `common/logger.py` | 每轮保存 loss、accuracy、epoch time、samples/s 到 CSV | `results/raw/*.csv` 正常生成 |

这一阶段的作用是建立单进程 baseline。后面 PS 和 DDP 的结果都要和这个 baseline 对比。

---

### 第 2 阶段：实现公共数据划分与通信量估算

| 步骤 | 目标 | 需要实现的文件 | 具体任务 | 验收标准 |
|---|---|---|---|---|
| 2.1 | 实现数据划分 | `common/datasets.py` | 支持按 worker 数切分训练集 | 每个 worker 样本数接近一致 |
| 2.2 | 支持 DistributedSampler | `ddp/train_ddp.py` | 为 DDP 训练准备分布式采样器 | 不同 rank 读取不同数据 |
| 2.3 | 统计模型参数量 | `utils/comm.py` | 计算模型参数数量和模型大小 | 能输出模型大小 MB |
| 2.4 | 估算 PS 通信量 | `utils/comm.py` | 估算 worker 拉取参数和上传梯度的通信量 | CSV 中有 `comm_mb` |
| 2.5 | 估算 DDP 通信量 | `utils/comm.py` | 估算 AllReduce 梯度同步通信量 | 能对比 PS 与 DDP 通信成本 |

核心公式：

$$
P=\sum_{i=1}^{L}|\theta_i|
$$

其中 \(P\) 是模型参数总量，\(\theta_i\) 是第 \(i\) 个参数张量。

若使用 float32，则模型大小为：

$$
S_{\mathrm{model}} = 4P \ \mathrm{bytes}
$$

Parameter Server 中，每个 worker 一轮通常需要下载模型并上传梯度，因此估算通信量为：

$$
C_{\mathrm{PS}} \approx 2K S_{\mathrm{model}}
$$

其中 \(K\) 是 worker 数量。

DDP / AllReduce 中，每轮反向传播后需要同步梯度。若使用 Ring AllReduce，理论上传输量通常近似为：

$$
C_{\mathrm{Ring}} \approx 2\frac{K-1}{K}S_{\mathrm{model}}
$$

每个进程承担的通信量随 \(K\) 增大趋近于 \(2S_{\mathrm{model}}\)。

---

### 第 3 阶段：实现 Parameter Server 多进程训练

| 步骤 | 目标 | 需要实现的文件 | 具体任务 | 验收标准 |
|---|---|---|---|---|
| 3.1 | 定义 PS Server | `ps/server.py` | server 保存全局模型参数和优化器 | 能接收 worker 梯度 |
| 3.2 | 定义 PS Worker | `ps/worker.py` | worker 拉取模型、计算梯度、上传梯度 | 能返回梯度和样本数 |
| 3.3 | 实现进程通信 | `ps/train_ps.py` | 使用 `multiprocessing.Queue` 实现 server-worker 消息传递 | 多个 worker 能并行运行 |
| 3.4 | 实现同步 PS-SGD | `ps/train_ps.py` | server 等待所有 worker 梯度后平均更新 | loss 能下降 |
| 3.5 | 记录系统指标 | `ps/train_ps.py` | 保存 epoch time、samples/s、comm_mb | CSV 正常生成 |
| 3.6 | 对比单进程 baseline | `scripts/run_ps.py` | 跑 Single vs PS | 准确率趋势合理 |

Parameter Server 同步 SGD 的核心公式与 Sync-SGD 一致：

$$
g_t=\sum_{k=1}^{K}\frac{n_k}{n}g_k(w_t)
$$

$$
w_{t+1}=w_t-\eta g_t
$$

但和 DML-Bench 的区别是：这里不再是单进程模拟，而是真实多进程 worker 通过队列和 server 交换数据。

这一阶段你要重点观察：Parameter Server 结构中，server 可能成为通信和聚合瓶颈。

---

### 第 4 阶段：实现 PyTorch DDP / AllReduce 训练

| 步骤 | 目标 | 需要实现的文件 | 具体任务 | 验收标准 |
|---|---|---|---|---|
| 4.1 | 初始化进程组 | `ddp/train_ddp.py` | 使用 `torch.distributed.init_process_group()` | 每个进程能获得 rank/world_size |
| 4.2 | 分布式采样 | `ddp/train_ddp.py` | 使用 `DistributedSampler` 划分数据 | 不同 rank 数据不重复 |
| 4.3 | 包装 DDP 模型 | `ddp/train_ddp.py` | 使用 `DistributedDataParallel(model)` | backward 后自动同步梯度 |
| 4.4 | 启动多进程 | `torchrun` 命令 | 使用 `torchrun --standalone --nproc_per_node=2` 启动 | 训练能正常运行 |
| 4.5 | 记录 rank 0 日志 | `ddp/train_ddp.py` | 只让 rank 0 保存 CSV 和打印结果 | 避免多进程重复写文件 |
| 4.6 | 对比 PS 架构 | `scripts/run_ddp.py` | 跑 Single vs PS vs DDP | 得到时间、吞吐量、准确率对比 |

DDP 的梯度同步可理解为：

$$
g_t=\frac{1}{K}\sum_{k=1}^{K}g_k(w_t)
$$

但这个平均过程不需要用户手动写聚合逻辑，而是在 `loss.backward()` 后由 DDP 自动触发 AllReduce 完成。

你要重点理解：DDP 是数据并行训练的工程实现。每个进程持有完整模型副本，不再有中心 server 持有唯一全局模型。

---

### 第 5 阶段：实现统一实验入口与运行脚本

| 步骤 | 目标 | 需要实现的文件 | 具体任务 | 验收标准 |
|---|---|---|---|---|
| 5.1 | 单进程运行脚本 | `scripts/run_single.py` | 调用 `single/train_single.py` | 一条命令跑 baseline |
| 5.2 | PS 运行脚本 | `scripts/run_ps.py` | 调用 `ps/train_ps.py` | 支持 `num_workers=2,4` |
| 5.3 | DDP 运行脚本 | `scripts/run_ddp.py` | 调用 `torchrun` 启动 DDP | 支持 `nproc_per_node=2,4` |
| 5.4 | 总运行脚本 | `scripts/run_all.py` | 顺序跑 Single、PS、DDP | 自动生成所有 CSV |
| 5.5 | 配置参数 | `configs/*.yaml` | 保存实验参数 | 可复现实验设置 |

建议总运行脚本执行以下实验：

```text
Single Process
PS, num_workers = 2
PS, num_workers = 4
DDP, num_workers = 2
DDP, num_workers = 4
```

如果当前电脑资源有限，第一版可以只做：

```text
Single Process
PS, num_workers = 2
DDP, num_workers = 2
```

---

### 第 6 阶段：实验汇总与图表绘制

| 步骤 | 目标 | 需要实现的文件 | 具体任务 | 验收标准 |
|---|---|---|---|---|
| 6.1 | 汇总 CSV | `utils/summarize.py` | 读取 `results/raw/*.csv` | 生成 `summary.csv` |
| 6.2 | 绘制精度曲线 | `utils/plot.py` | 画 test accuracy vs epoch | 有算法对比图 |
| 6.3 | 绘制时间曲线 | `utils/plot.py` | 画 epoch time vs epoch | 能比较训练时间 |
| 6.4 | 绘制吞吐量曲线 | `utils/plot.py` | 画 samples/s vs epoch | 能比较系统吞吐 |
| 6.5 | 绘制通信量图 | `utils/plot.py` | 画 estimated communication MB | 能比较通信成本 |
| 6.6 | 生成最终表格 | `results/tables/summary.csv` | 汇总 final acc、time、samples/s、comm_mb | 能放入报告 |

这一阶段输出的是最终展示材料，不再修改核心训练代码。

---

### 第 7 阶段：实验报告与项目整理

| 步骤 | 目标 | 需要实现的文件 | 具体任务 | 验收标准 |
|---|---|---|---|---|
| 7.1 | 写项目背景 | `report/ps_vs_allreduce_report.md` | 说明 PS 与 AllReduce 的研究意义 | 能解释为什么比较二者 |
| 7.2 | 写方法部分 | `report/ps_vs_allreduce_report.md` | 说明 Single、PS、DDP 的系统结构 | 有公式和流程图说明 |
| 7.3 | 写实验设置 | `report/ps_vs_allreduce_report.md` | 说明数据集、模型、参数、设备 | 实验可复现 |
| 7.4 | 写实验结果 | `report/ps_vs_allreduce_report.md` | 分析准确率、时间、吞吐量、通信量 | 结论清晰 |
| 7.5 | 整理 README | `README.md` | 说明环境、运行命令、结果文件 | 其他人可复现 |

报告需要重点回答：

| 问题 | 应该能回答 |
|---|---|
| Parameter Server 和 AllReduce 的系统结构有什么不同？ | PS 有中心 server，AllReduce 是去中心化同步 |
| 为什么 PS 可能出现 server bottleneck？ | 所有 worker 都要和 server 交换参数/梯度 |
| 为什么 DDP 不需要手写梯度平均？ | DDP 在 backward 后自动执行 AllReduce |
| 为什么要统计 samples/s？ | 它体现系统吞吐量 |
| 为什么只看 accuracy 不够？ | 分布式系统还要看时间、吞吐和通信成本 |
| worker 数增加后一定更快吗？ | 不一定，通信和进程开销可能抵消并行收益 |

---

## 四、推荐开发顺序总表

你接下来按这个顺序做，不要一开始直接写 DDP。

| 顺序 | 模块 | 优先级 | 预计时间 | 完成后你应该能做到 |
|---:|---|---|---|---|
| 1 | 项目结构 + 环境配置 | 必做 | 0.5 天 | 项目能运行 |
| 2 | 公共数据、模型、日志模块 | 必做 | 0.5 天 | 单进程和分布式共享代码 |
| 3 | Single Process baseline | 必做 | 0.5–1 天 | 有准确率和时间基线 |
| 4 | 通信量估算模块 | 必做 | 0.5 天 | 能统计模型大小和通信开销 |
| 5 | Parameter Server 同步版 | 必做 | 1.5–2 天 | 多进程 PS 能训练 |
| 6 | PS 不同 worker 数实验 | 必做 | 0.5 天 | 能观察扩展性 |
| 7 | DDP 基础训练 | 必做 | 1–1.5 天 | torchrun 能启动多进程 |
| 8 | DDP 日志与通信估算 | 必做 | 0.5 天 | 能和 PS 做统一对比 |
| 9 | 汇总脚本与画图 | 必做 | 0.5–1 天 | 有完整结果表和图 |
| 10 | 报告整理 | 必做 | 1 天 | 能形成展示材料 |
| 11 | 异步 PS | 可选 | 1–2 天 | 能比较同步/异步 PS |
| 12 | CIFAR-10 + CNN | 可选 | 1–2 天 | 扩展到更复杂任务 |
| 13 | 多 seed 实验 | 可选 | 1 天 | 结果更稳定 |
| 14 | IBing / AllReduce 模拟器衔接 | 进阶 | 2–3 天 | 为论文复现做准备 |

---

## 五、第一版实验矩阵

第一版不要做太多实验。先跑下面这些。

| 实验编号 | 对比内容 | 固定设置 | 变化变量 | 目的 |
|---|---|---|---|---|
| Exp-1 | Single vs PS | MNIST、MLP、epochs=10 | algorithm | 验证 Parameter Server 实现 |
| Exp-2 | Single vs DDP | MNIST、MLP、epochs=10 | algorithm | 验证 DDP / AllReduce 实现 |
| Exp-3 | PS vs DDP | MNIST、MLP、workers=2 | algorithm | 比较两类系统架构 |
| Exp-4 | worker 数扩展 | MNIST、MLP | workers = 1,2,4 | 观察扩展性和进程开销 |
| Exp-5 | 通信成本分析 | MNIST、MLP | algorithm | 比较 PS 与 AllReduce 通信估算 |

第一版不要上 CIFAR-10，也不要一开始做真实多机。MNIST + MLP 已经足够验证系统机制。

---

## 六、最终需要生成的结果

| 输出 | 文件位置 | 内容 |
|---|---|---|
| 单进程训练曲线 | `results/figures/single_curve.png` | Single baseline loss / accuracy |
| PS 训练曲线 | `results/figures/ps_curve.png` | PS loss / accuracy |
| DDP 训练曲线 | `results/figures/ddp_curve.png` | DDP loss / accuracy |
| 算法对比曲线 | `results/figures/system_accuracy.png` | Single、PS、DDP accuracy |
| 时间对比图 | `results/figures/epoch_time_compare.png` | 不同系统 epoch time |
| 吞吐量对比图 | `results/figures/samples_per_sec_compare.png` | 不同系统 samples/s |
| 通信量图 | `results/figures/comm_mb_compare.png` | PS vs DDP 通信估算 |
| worker 数扩展图 | `results/figures/scalability_workers.png` | workers vs samples/s |
| 汇总表 | `results/tables/summary.csv` | final acc、time、samples/s、comm MB |
| 项目报告 | `report/ps_vs_allreduce_report.md` | 方法、实验、分析、结论 |

---

## 七、你现在立刻要做的第一批文件

你现在先不要直接实现 Parameter Server 或 DDP。第一步先建立公共模块和单进程 baseline。

| 文件 | 当前阶段是否必须 | 功能 |
|---|---:|---|
| `requirements.txt` | 是 | 环境依赖 |
| `common/__init__.py` | 是 | Python 包识别 |
| `common/seed.py` | 是 | 固定随机种子 |
| `common/datasets.py` | 是 | 加载 MNIST / Fashion-MNIST |
| `common/models.py` | 是 | 定义 Logistic Regression 和 MLP |
| `common/metrics.py` | 是 | 计算 accuracy、loss、samples/s |
| `common/logger.py` | 是 | 保存 CSV |
| `single/__init__.py` | 是 | Python 包识别 |
| `single/train_single.py` | 是 | 单进程 baseline |
| `utils/comm.py` | 是 | 参数量和通信量估算 |
| `README.md` | 是 | 项目说明 |

第一步只跑通：

```bash
python -m single.train_single \
  --dataset mnist \
  --model mlp \
  --epochs 10 \
  --batch-size 64 \
  --lr 0.01 \
  --seed 42
```

当这个命令能正常训练、保存 CSV、打印 epoch time 和 samples/s 后，再进入 Parameter Server。

---

## 八、每个阶段你要真正学到什么

| 阶段 | 学习重点 |
|---|---|
| Single baseline | 单进程训练基线，后续系统对比的参考 |
| 公共模块 | 如何复用数据、模型、日志和评估代码 |
| Parameter Server | 中心化参数管理、server bottleneck、worker-server 通信 |
| DDP / AllReduce | 去中心化梯度同步、rank/world_size、DistributedSampler |
| 通信估算 | 分布式训练不仅看精度，还要估算通信成本 |
| 吞吐量统计 | samples/s 比单纯 epoch time 更能体现系统效率 |
| worker 扩展性 | worker 数增加不一定加速，通信和进程开销会影响收益 |
| 报告整理 | 把系统实现转化成可展示的工程实践材料 |

---

## 九、当前版本的完成标准

你第一版 PS vs AllReduce 完成时，至少要能回答这几个问题：

| 问题 | 你应该能给出的答案 |
|---|---|
| Parameter Server 和 AllReduce 有什么区别？ | PS 是中心化 server 聚合，AllReduce 是多进程间同步梯度 |
| DDP 中谁负责梯度平均？ | DDP 在 backward 后自动执行梯度 AllReduce |
| 为什么 PS 可能成为瓶颈？ | 所有 worker 都要向 server 上传梯度并拉取参数 |
| 为什么 DDP 需要 DistributedSampler？ | 保证不同 rank 读取不同训练数据分片 |
| `rank` 和 `world_size` 是什么？ | rank 是当前进程编号，world_size 是总进程数 |
| 为什么要统计 samples/s？ | 它衡量系统吞吐量，而不仅是模型精度 |
| worker 数变多为什么不一定更快？ | 进程开销、通信开销、同步等待可能抵消并行收益 |
| PS 和 DML-Bench 的 Sync-SGD 有什么关系？ | 数学更新类似，但 PS 是真实多进程通信实现 |

---

## 十、建议你现在的行动顺序

按这个顺序开始，不要跳到 DDP。

| 今天要做的顺序 | 任务 |
|---:|---|
| 1 | 建好完整目录结构 |
| 2 | 写 `requirements.txt` |
| 3 | 写 `common/seed.py` |
| 4 | 写 `common/datasets.py` |
| 5 | 写 `common/models.py` |
| 6 | 写 `common/metrics.py` |
| 7 | 写 `common/logger.py` |
| 8 | 写 `utils/comm.py` |
| 9 | 写 `single/train_single.py` |
| 10 | 跑通单进程 baseline |

等你完成 **Single Process baseline** 后，再继续实现 **Parameter Server 同步训练版本**。
