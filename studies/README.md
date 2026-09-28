# 架构与指令学习

此目录按机制组织 PTX、TMA 和 thread block cluster 的最小可运行例子。

每个例子应说明指令来源、所需 SM/CUDA 版本、输入输出语义、验证方法与观察 SASS 的命令。这里允许为学习目的保留本地 raw PTX；可复用的公共 wrapper 应放在 `include/cuda_lab/ptx_utils.cuh`。
