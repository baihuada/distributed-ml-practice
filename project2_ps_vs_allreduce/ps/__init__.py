"""
Parameter Server 模块。

本模块用于实现单机多进程版本的同步 Parameter Server 训练。

核心结构：
1. server 进程保存全局模型；
2. worker 进程保存本地模型副本；
3. server 向 worker 广播参数；
4. worker 计算梯度并上传；
5. server 聚合梯度并更新全局模型。
"""