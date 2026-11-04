# Softmax 计算强度 AI

计算强度（Arithmetic Intensity）定义为 `AI = FLOPs / Bytes`，单位为 FLOP/Byte，参见 [NVIDIA Nsight Compute Roofline 说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#roofline-charts)。本基准输出 `AI_estimate`，使用下面的算法计数模型；耗时仍由 CUDA event 独立测量。

## 计数口径

- 浮点加、减、乘、除各计 1 FLOP，乘加计 2 FLOPs。
- `exp`、最大值比较、整数运算、类型转换和量化舍入不计入 FLOPs。`exp` 通常需要多条指令，其执行代价未由这个 AI 模型描述。
- 归约按有效标量运算量计数：C 个数求和计 C−1 次加法。忽略初始化加零、补齐元素和 shuffle 归约产生的冗余算术；不预测编译器实际生成的指令数。
- Bytes 按 kernel 的输入读取与输出写入次数估算，不含共享内存和寄存器流量。warp/block 读取输入一次，online 读取两次；均写出一次。不计缓存、内存事务粒度及寄存器溢出造成的变化，因此不是实测 DRAM 流量。
- INT8 路径内部仍以 FP32 计算；整数最大值、减法、zero point 加法及饱和操作不计为浮点运算。

## 每行推导

设每行 C 列，online 每行 T=256 个线程。总行数在分子和分母中抵消。

| 路径 | 普通浮点运算量 | 访存字节量 | AI（FLOP/Byte） |
| --- | --- | --- | --- |
| FP32 warp/block | `C + (C−1) + 1 + C = 3C` | `4C + 4C = 8C` | `0.375` |
| INT8→FP32 | `C + (C−1) + 1 + C = 3C` | `C + 4C = 5C` | `0.600` |
| INT8→INT8 | `3C + C = 4C` | `C + C = 2C` | `2.000` |
| FP32 online | `4C + U + 3T`，`0 <= U <= C` | `4C + 4C + 4C = 12C` | `[(4C+3T)/(12C), (5C+3T)/(12C)]` |

FP32 warp/block 的四项分别为减最大值、求指数和、求倒数、乘倒数。INT8→FP32 的第一项是乘输入 scale；INT8→INT8 额外计入除输出 scale 的 C 次浮点运算。

online 的 U 是线程局部最大值变大、需要重缩放累计和的更新次数，其值与输入和遍历顺序有关。每次更新包含一次减法、一次加法，重缩放时再加一次乘法，共 `2C+U`；线程局部和转换到 block 最大值基准需要 `2T` 次运算；归约及倒数共 `T` 次；第二遍减最大值并乘倒数共 `2C` 次。因此基准输出一个保守区间，不额外模拟输入相关的 U。

报告表中的 AI 均为按此口径计算的估算值，不是由历史耗时反推得到。若需要实测 Roofline AI，应使用性能分析器采集对应层级的浮点运算与数据流量；本模型也不能单独反映指数函数的吞吐瓶颈。

## 运算吞吐与硬件上限

普通基准同时输出 `GFLOP/s_estimate = FLOPs_estimate / (latency_ms * 1e6)`，与 AI 使用相同的算法运算量。online 路径仍输出上下界。历史报告中的 GFLOP/s 由保留的历史耗时计算得到。

硬件计数与算法估算分开使用，通用定义见 [Roofline 方法](../../benchmarks/methodology/roofline.md)，本算子结果见 [采样报告](roofline.md)。
