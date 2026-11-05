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
| Vector Add | [原生 CUDA](operators/vector_add/vector_add_bench.cu) · [Python 多后端](operators/vector_add/benchmark.py) | [用例](operators/vector_add/cases.json) · [构建与运行](../docs/guides/build/usage.md) · [框架验证](../docs/operators/vector_add/validation.md) |
| DWConv | [多后端 Event 入口](operators/dwconv/compare/main.cpp) · [统一批量 CSV](operators/dwconv/dwconv_full_bench.py) | [输入预取范围测量](../docs/operators/dwconv/prefetch-benchmark.md) · [FP32 prefetch 与 baseline 对比](../docs/operators/dwconv/prefetch-fp32.md) · [FP16 报告](../docs/operators/dwconv/prefetch-fp16.md) · [权重缓存报告](../docs/operators/dwconv/weights-cache.md) · [FP32 opt 报告](../docs/operators/dwconv/opt-fp32.md) · [cuDNN 融合与布局对比](../docs/operators/dwconv/cudnn-fusion.md) · [cuDNN autotune](../docs/operators/dwconv/cudnn-autotune.md) · [FP32 全量对比](../docs/operators/dwconv/fp32-full.md) · [FP16 全量对比](../docs/operators/dwconv/fp16-full.md) |
| Softmax | [基础](operators/softmax/softmax_bench.cu) · [Cluster 对照](operators/softmax/softmax_cluster_bench.cu) | [NCU 脚本](operators/softmax/softmax_roofline.py) · [Roofline](../docs/operators/softmax/roofline.md) · [Cluster](../docs/operators/softmax/cluster.md) |

DWConv 的后端适配、数据准备、CPU 参考和 Event 计时统一放在 `benchmarks/operators/dwconv/`。单 shape 使用 `dwconv_compare_bench`，批量对比使用 `dwconv_full_bench.py`；构建、参数、输出及绘图见[模块说明](operators/dwconv/README.md)。DWConv 暂不注册原生测试，待 Python 绑定完成后添加接口测试。

## 存储测量入口

[存储微基准](memory/README.md)：shared memory、L1、L2、DRAM 的带宽与延迟，共 8 项；[导入报告](../docs/benchmarks/memory/baseline.md)记录配置、计时范围和验证结果。

## Python 多后端对比

从仓库根目录运行 `python -m benchmarks.operators.vector_add.benchmark --mode graph`。

- 算子目录的 `backends.py` 只准备输出和待测函数；输入、正确性、计时及结果由统一入口管理。公共工具放 `common/`。
- `--mode event` 用 CUDA Event 包围一批普通调用，可能包含主机提交间隙；`--mode graph` 捕获一批调用并计时 replay，按调用数摊销。两种口径分别报告，不能混为单 kernel 的纯执行时间。
- `--api-time` 另报 `api_batch_us`：使用主机计时，包含一批预分配输出的 API 调用和批末同步，按调用数摊销。它不包含输出分配，也不代表单次请求延迟。
- 每组重复采样，保存每次样本、中位数和最小/最大值。默认用例是延迟和接口对比示例；要建立吞吐平台，应按公共配置补做 waves 扫描、occupancy 与缓存条件记录。
- 结果目录包含 `environment.json`、`config.json`、`device-query.log`、`results.jsonl` 和 `summary.csv`。记录实际包路径、构建与运行提交状态；`speedup_vs_torch = torch_gpu_us / backend_gpu_us`。
- 空输入记为 `skip` 并说明原因；后端导入、正确性或计时失败记为 `error`，入口返回非零。错误结果不生成加速比。
- 临时结果留在 `build/`；正式实验选择数据、配置与配图后按[文档规则](../docs/README.md)归档。
