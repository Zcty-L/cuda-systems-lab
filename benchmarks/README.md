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
| DWConv | [FP32 版本对比及诊断](operators/dwconv_bench.cu) · [FP16 基准](operators/dwconv_fp16_bench.cu) · [权重缓存](operators/dwconv_weights_bench.cu) · [FP32 opt](operators/dwconv_opt_bench.cu) · [bias 加载](operators/dwconv_bias_bench.cu) · [cuDNN 融合](operators/dwconv_cudnn_bench.py) | [输入预取范围测量](../docs/operators/dwconv/prefetch-benchmark.md) · [FP32 prefetch 与 baseline 对比](../docs/operators/dwconv/prefetch-fp32.md) · [FP16 报告](../docs/operators/dwconv/prefetch-fp16.md) · [权重缓存报告](../docs/operators/dwconv/weights-cache.md) · [FP32 opt 报告](../docs/operators/dwconv/opt-fp32.md) · [cuDNN 融合与布局对比](../docs/operators/dwconv/cudnn-fusion.md) · [FP32 全量对比](../docs/operators/dwconv/fp32-full.md) · [FP16 全量对比](../docs/operators/dwconv/fp16-full.md) |
| Softmax | [基础](operators/softmax_bench.cu) · [Cluster 对照](operators/softmax_cluster_bench.cu) | [NCU 脚本](operators/softmax_roofline.py) · [Roofline](../docs/operators/softmax/roofline.md) · [Cluster](../docs/operators/softmax/cluster.md) |

DWConv 结果绘图脚本：[dwconv_plot.py](operators/dwconv_plot.py)。按 H、C 排列子图，横轴为 batch size，纵轴为 TFLOPs/s；正式结果与图片归档到 [5070 Ti 实验目录](../docs/operators/dwconv/results/2026-10-13-rtx5070ti-laptop/README.md)和[RTX 4090 实验目录](../docs/operators/dwconv/results/2026-10-13-rtx4090/README.md)。图标题从 CSV 同目录的 `device.txt` 读取设备名；优先使用 Times New Roman，缺少时使用 Liberation Serif，也可通过 `--font-file` 指定 Times New Roman 字体文件。

```bash
conda activate py311
python benchmarks/operators/dwconv_plot.py --csv docs/operators/dwconv/results/2026-10-13-rtx5070ti-laptop/fp32/comparison.csv --dtype fp32 --k 11 --output docs/operators/dwconv/results/2026-10-13-rtx5070ti-laptop/fp32/plots/benchmark_fp32_k11.png
```

## 存储测量入口

[存储微基准](memory/README.md)：shared memory、L1、L2、DRAM 的带宽与延迟，共 8 项；[导入报告](../docs/benchmarks/memory/baseline.md)记录配置、计时范围和验证结果。
