# DWConv cuDNN autotune 小范围测试

2026-10-29，RTX 5070 Ti Laptop / SM120，CUDA 13.2、cuDNN 9.23.1。

本轮 FP32 与 FP16 各五个 shape、各 15 次 autotune 全部选择 engine 0，与原 heuristic 的 engine 一致。尚未发现更快的其他 engine。小输入普通 launch 波动显著，不能把两组中位数差异直接归因为选优收益，也不能据此断言所有 shape 上 heuristic 已最优。

## 方法

- NCHW，stride=1，same padding，bias 融合，FP32 禁用 TF32；FP16 输入输出、FP32 累加，保持原精度筛选条件。
- 原 `cudnn`：A 首个可构建方案，无方案才 FALLBACK。
- 新 `cudnn-autotune`：收集 A/B/FALLBACK 的可构建候选，workspace 上限 256 MiB；每个候选 20 次预热、100 次连续 launch，三轮中位数选优。不同 heuristic 可能返回重复配置，本轮保留，不以 engine ID 去重，避免丢弃同 engine 的不同 knob 配置。因此候选数不是唯一算法数；这不是穷举所有 engine/knob。
- variant pack 在准备阶段绑定；搜索使用最大候选 workspace 并保留，日志初始 workspace 是分配量，各候选行才是该 plan 的需求。搜索结果由当前对象复用，不持久化。
- 正式测量在选优之外：100 次预热、500 次连续 launch 的 CUDA event 均值；每组独立进程运行三次，交替两种入口顺序。表中给出三次中位数及最小–最大值。
- 不使用 CUDA Graph replay，不锁频、不清缓存。小输入的主机提交间隙、运行状态波动和 autotune 较长的准备阶段影响可比性；本轮不足以证明小幅收益。
- 数据分配、H2D、plan 构建与搜索均不计入 kernel 时间。

## FP32 结果

单位 μs；相对耗时为 autotune / 原 heuristic，原 heuristic=1.0。

| case | K | N×C×H×W | heuristic 中位数（范围） | autotune 中位数（范围） | 相对耗时 |
| --- | --- | --- | --- | --- | --- |
| 7 | 3 | 1×32×43×43 | 13.280 (12.973–30.614) | 13.475 (9.875–13.508) | 1.015 |
| 5 | 5 | 1×32×80×80 | 17.360 (16.963–37.914) | 15.866 (15.572–32.199) | 0.914 |
| 4 | 7 | 1×32×43×43 | 16.523 (14.087–32.390) | 14.601 (13.018–31.446) | 0.884 |
| 401 | 9 | 1×32×40×40 | 16.620 (15.984–36.419) | 14.311 (14.183–14.635) | 0.861 |
| 8 | 11 | 1×32×43×43 | 19.699 (19.394–20.530) | 20.492 (19.843–20.556) | 1.040 |

K=11 两组差异约 4%，且测得范围接近，不能称为 autotune 提升。其余 shape 的单次波动更大。要进一步分离调度影响，应在同一进程、相同 CUDA Graph replay 口径下比较，并扩展到更大输入。

## FP16

| case | N×C×H×W | K | cuDNN 中位数（范围） | autotune 中位数（范围） | 相对耗时 | cuDNN / autotune engine |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 1×32×43×43 | 3 | 11.794 (10.594–12.719) | 12.485 (11.204–12.626) | 1.059 | 0 / 0 |
| 5 | 1×32×80×80 | 5 | 19.631 (19.159–19.839) | 17.639 (17.594–38.827) | 0.898 | 0 / 0 |
| 4 | 1×32×43×43 | 7 | 26.792 (13.897–32.024) | 15.115 (15.018–16.977) | 0.564 | 0 / 0 |
| 401 | 1×32×40×40 | 9 | 16.598 (15.808–18.167) | 16.545 (16.142–36.754) | 0.997 | 0 / 0 |
| 8 | 1×32×43×43 | 11 | 22.995 (22.144–24.403) | 21.718 (20.392–34.939) | 0.944 | 0 / 0 |

FP16 同样未选出其他 engine；K=7 的两组耗时范围尤其宽，不能将中位数比值解释为选优带来的加速。FP16 为后续独立补测，与此前 FP32 不构成同时段精度性能对照。

## 验证及复现

构建成功；case 7 的 FP32/FP16 autotune 均通过 CPU 数值校验，最大绝对误差分别为 5.39511e-08、3.8147e-06。其余性能用例关闭 CPU 校验。FP16 补测完成 30 组执行（5 个 shape × 2 个入口 × 3 次重复），性能测量关闭 CPU 校验。

```bash
cmake --build build --target dwconv_test device_query -j 8
./build/tools/device_query
./build/tests/native/dwconv_test --kernel cudnn-autotune --dtype all --case 7 --check-cpu
conda activate py311
python benchmarks/operators/dwconv/dwconv_autotune_bench.py
python benchmarks/operators/dwconv/dwconv_autotune_bench.py --dtype fp16 --output build/dwconv-autotune-fp16
```

[汇总对比表](results/2026-10-29-rtx5070ti-laptop-cudnn-autotune/comparison.md)包含配置、三轮耗时中位数与范围、engine 及相对耗时。正式归档仅保留该表；详细日志保留在本地 `build/dwconv-autotune/`（FP32）与 `build/dwconv-autotune-fp16/`（FP16），不纳入正式结果目录。
