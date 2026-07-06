"""
DDP / AllReduce 训练模块。

本模块用于实现 PyTorch DistributedDataParallel 训练。

核心机制：
1. 每个 rank 持有完整模型副本；
2. 每个 rank 读取不同数据分片；
3. loss.backward() 后，DDP 自动执行梯度 AllReduce；
4. optimizer.step() 后，各 rank 的模型参数保持一致；
5. 只让 rank 0 保存日志和打印主要结果。
"""