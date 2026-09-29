# Softmax 行数与并行度

## 512×1024 为什么不足一整波

当前 1024 列路径每个 block 有 128 个线程，即 4 个 warp；每个 warp 依次处理 2 行，因此一个 block 处理 8 行。这里的 8 行不是由 8 个 warp 同时计算。

本机 RTX 5070 Ti Laptop GPU 有 46 个 SM。当前编译结果每个线程使用 124 个寄存器，按资源分配粒度折算为 128；NCU 给出的寄存器限制为每个 SM 最多驻留 4 个 block。于是：

```text
512 行的 grid = ceil(512 / 8) = 64 blocks
当前 kernel 一整波的容量 = 46 SM × 4 blocks/SM = 184 blocks
waves = 64 / 184 ≈ 0.35

一整波所需行数 = 184 × 8 = 1472 rows
当前 kernel 每 SM 的驻留上限 = 4 blocks × 4 warps = 16 warps
硬件每 SM 上限 = 48 warps
当前 kernel 理论占用率 = 16 / 48 ≈ 33.3%
```

因此 64 个 block 虽多于 46 个 SM，但平均只有约 1.39 blocks/SM，未填满这版 kernel 能驻留的 block 容量。增大 rows 可以提供更多独立 block、更多可用于隐藏延迟的 warp，并摊薄启动和收尾开销。当前寄存器用量造成的 33.3% 理论占用率上限，则不会仅因 rows 增大而消失。waves 和 occupancy 指标的定义见 [NCU 指标说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#launch-metrics)。

这里的 waves 是容量估算，并非 GPU 等待整波完成后才统一调度下一波；实际执行还存在启动、收尾和负载分布的影响。当前 warp launcher 的 grid 上限为 65535 blocks，超过后通过行循环处理，不能无限按 `ceil(rows/8)` 增长。

## 实验配置与结果

2026-09-29，同一 RTX 5070 Ti Laptop GPU、SM 12.0、CUDA 13.2、NCU 2026.1.1、Release 构建。固定 1024 列，仅改变 rows；随机种子 2026，均跳过 20 次预热，采集接下来的一次 kernel，冷缓存、请求 base 时钟。硬件 GFLOP/s 和 Roofline 的计数方法见 [公共方法](../../benchmarks/methodology/roofline.md)。

| rows | blocks | waves/SM | 实际占用率 | DRAM 利用率 | kernel 时间（µs） | 硬件 GFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 64 | 0.35 | 11.44% | 28.99% | 18.528 | 376.705 |
| 1472 | 184 | 1.00 | 28.47% | 45.65% | 30.944 | 648.472 |
| 4096 | 512 | 2.78 | 29.20% | 65.55% | 74.464 | 749.848 |
| 16384 | 2048 | 11.13 | 30.16% | 75.38% | 381.312 | 585.732 |

增大 rows 后，实际占用率从约 11% 升至接近当前 kernel 的 33.3% 理论上限，支持“小网格限制并行度”的判断。但吞吐没有一直上升：16384 行时 DRAM 利用率更高，GFLOP/s 却低于 4096 行。此时数据规模及实际 DRAM 字节量也改变了，性能需要结合缓存和访存指标判断，不能只看 block 数。所有数字是本次单次采样结果，未作重复统计。

行数增加也增加了总工作量，所以应比较吞吐或单位元素耗时，不能要求总耗时下降。若真实输入就是 512 行，应保留该用例评估小批量延迟；大 rows 用例用于观察吞吐及饱和行为，不代表原 512 行场景得到了优化。

## 复现

`softmax_bench` 新增 `--rows`，默认仍为 512。指定 `--cols` 时只运行一种 FP32 配置；仅指定 `--rows` 时，对全部 FP32 与 INT8 配置使用该行数。AI 不随行数改变；GFLOP/s 的总运算量随实际 rows 缩放。

```bash
cmake --build build --target softmax_bench --parallel 4
./build/benchmarks/operators/softmax_bench --rows 4096 --cols 1024

conda activate py311
python benchmarks/operators/softmax_roofline.py \
    --rows 512 1472 4096 16384 --cols 1024 --output build/softmax-row-scaling
python benchmarks/operators/softmax_roofline.py \
    --rows 512 1472 4096 16384 --cols 1024 --output build/softmax-row-scaling --summarize-only
```
