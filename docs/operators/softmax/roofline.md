# Softmax Roofline 采样报告

通用公式、指标与判断顺序见 [Roofline 方法](../../benchmarks/methodology/roofline.md)。

## 本次采样口径

- `benchmarks/operators/softmax_roofline.py` 采样标量 FP32 kernel，采用普通 FP32 峰值。
- 硬件计数包含 `exp` 展开后的普通浮点指令和归约冗余，与 [Softmax 算法估算](arithmetic-intensity.md) 分开使用；指数执行压力辅助观察 XU 指标。
- 峰值由 NCU 每周期持续峰值乘采样时钟得到；请求 `--clock-control base`，实际时钟以 CSV 为准。
- 输出 `fp32_peak_pct`、`dram_byte_peak_pct` 和 Roofline 利用率；`roof_side` 仅表示模型限制侧。

## 复现

普通基准通过 CMake 构建；NCU 采样为独立可选脚本，仅依赖 Python 标准库与已安装的 Nsight Compute，不增加项目配置依赖。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target softmax_bench --parallel 4
./build/benchmarks/operators/softmax_bench

conda activate py311
python benchmarks/operators/softmax_roofline.py --cols 1024 8192 65536 --output build/softmax-roofline
# 已有采样时，只重新汇总，不再启动 profiler：
python benchmarks/operators/softmax_roofline.py --summarize-only --output build/softmax-roofline
```

默认配置为 512 行、种子 2026；脚本支持 `--rows` 与 `--cols` 列表的所有组合，用 `softmax_bench --rows R --cols C` 选择单个配置。跳过 20 次预热，只采集下一次 kernel；使用 `--cache-control all` 在重放之间清缓存。输出目录保存完整命令、NCU 原始 CSV、应用日志和 `summary.csv`；512 行保留 `ncu-C.csv` 命名，其他形状使用 `ncu-RxC.csv`。应用日志里的 CUDA event 耗时受 profiler 干扰，**不用于性能结论**；本报告只使用 NCU 的 kernel duration。普通基准是连续调用且未清缓存，两者的耗时不能直接比较。采样工具的具体行为参见 [NCU 计时与缓存控制说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#workload-durations)。

指标在不同架构与 NCU 版本间可能变化；当前脚本在 SM 12.0、NCU 2026.1.1 验证。缺失的诊断指标留空并警告；缺失核心计数时终止汇总，避免产生无依据的上限数值。

## 本机结果

2026-09-29，RTX 5070 Ti Laptop GPU，SM 12.0，CUDA 13.2，Release，NCU 2026.1.1。以下每种配置只采样一次并经多次重放收集计数。采样时推导 FP32 峰值约 12.50～12.66 TFLOP/s，DRAM 峰值约 430～432 GB/s，拐点约 29 FLOP/Byte。

| 配置（512 行） | kernel 时间（µs） | 硬件 AI_DRAM | 实测 GFLOP/s | Roofline 上限（GFLOP/s） | 达到上限 | 理想下界（µs） | 上限倍率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warp，1024 列 | 19.072 | 3.270 | 365.960 | 1406.072 | 26.03% | 4.964 | 3.84× |
| block，8192 列 | 77.088 | 2.714 | 724.961 | 1169.671 | 61.98% | 47.779 | 1.61× |
| online，65536 列 | 1326.464 | 2.036 | 609.924 | 878.855 | 69.40% | 920.564 | 1.44× |

| 配置 | SM 吞吐 | DRAM 吞吐 | L2 吞吐 | XU 吞吐 | 实际占用率 | eligible warp/调度器 | waves/SM | long scoreboard 周期/发射 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warp，1024 列 | 12.96% | 26.03% | 24.81% | 3.60% | 11.42% | 0.21 | 0.35 | 2.94 |
| block，8192 列 | 25.92% | 61.98% | 46.80% | 7.18% | 31.18% | 0.43 | 5.57 | 5.71 |
| online，65536 列 | 12.45% | 69.40% | 38.52% | 7.43% | 92.62% | 0.16 | 1.86 | 76.61 |

三种配置都位于该 DRAM Roofline 模型的内存侧。本次证据支持以下判断：

- 1024 列仅 64 个 block，不足一整波，实际占用率低，DRAM 和执行管线都远未饱和；优先检查小网格和并行度，不能简单称为带宽跑满。
- 8192 列的 DRAM 利用率升至约 62%，距离带宽上限仍有空间；占用率约 31%，102 个寄存器/线程，访存、资源占用和同步都值得继续定位。
- 65536 列的 DRAM 利用率约 69%，long scoreboard 等待明显，即便占用率约 93%，eligible warp 仍很少。证据更倾向于访存路径和延迟隐藏受限，XU 利用率没有显示全局饱和。

这是冷缓存、受控时钟下的单次采样判断，不代表所有规模和运行条件。小 kernel 的计数还可能受显示负载及多次重放波动影响；上述倍率仅为该模型中的理想空间。

固定 1024 列、增大 rows 后的结果及占用率上限推导见 [行数与并行度](row-scaling.md)。
