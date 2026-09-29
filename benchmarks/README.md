# 性能测量

存放存储、指令和算子的性能基准。

## 公共配置

- 记录设备、SM、CUDA 版本、输入、grid/block、计时范围、运行命令与测量结果；预热默认 20 次，计时默认 100 次，覆盖默认值时注明。
- 吞吐测试默认从 5 waves 开始，扫描约 5、10、20 waves；连续两次扩大规模后吞吐变化均不超过 5%，作为平台参考，同时记录测量波动。这是项目约定。
- 普通 block 网格：`waves ≈ grid_blocks / (SM_count × resident_blocks_per_SM)`；驻留容量按实际 kernel 计算，可用 Occupancy API 估算、NCU 核对。
- Cluster 网格：`waves ≈ grid_clusters / active_clusters`，设备驻留容量用 `cudaOccupancyMaxActiveClusters` 查询。
- 实现对比使用相同设备、输入和计时条件，记录各自 waves；保留真实小输入的延迟用例。特殊调度或延迟微基准另述配置依据。
- 记录工作集、缓存与时钟条件；waves 达标不代表吞吐饱和。指标定义与瓶颈判断见 [Roofline 方法](../docs/benchmarks/methodology/roofline.md)。

## 算子测量入口

| 算子 | 基准 | 分析与报告 |
| --- | --- | --- |
| Vector Add | [vector_add_bench.cu](operators/vector_add_bench.cu) | — |
| Softmax | [基础](operators/softmax_bench.cu) · [Cluster 对照](operators/softmax_cluster_bench.cu) | [NCU 脚本](operators/softmax_roofline.py) · [Roofline](../docs/operators/softmax/roofline.md) · [Cluster](../docs/operators/softmax/cluster.md) |
