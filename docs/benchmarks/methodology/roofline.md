# Roofline 指标与判断方法

## 计算口径

算法估算描述有效工作量；硬件计数描述实际执行量。AI、吞吐和上限计算须使用一致的运算、流量与时间口径，避免混用。

设运算量为 `F`，所选存储层级的字节量为 `Q`，kernel 时间为 `t`，对应算力峰值为 `P_peak`，该层级带宽峰值为 `B_peak`：

| 指标 | 公式 |
| --- | --- |
| 计算强度 | `AI = F / Q`，单位 FLOP/Byte |
| 运算吞吐 | `P_actual = F / t`；除以 `1e9` 得 GFLOP/s |
| 拐点 | `AI_ridge = P_peak / B_peak` |
| Roofline 上限 | `P_roof = min(P_peak, AI × B_peak)` |
| Roofline 利用率 | `P_actual / P_roof` |
| 理想耗时下界 | `t_lower = max(F / P_peak, Q / B_peak)` |
| 上限倍率 | `P_roof / P_actual = t / t_lower` |

峰值须匹配数据类型、执行管线和时钟条件；上限倍率是理想模型中的空间，不是承诺的加速比。

### NCU 标量 FP32 / DRAM 示例

```text
F = FADD_count + FMUL_count + 2 * FFMA_count
t = gpu__time_duration.sum（换算为秒）
Q = dram__bytes.sum
P_peak = 2 * sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained
           * sm__cycles_elapsed.avg.per_second
B_peak = dram__bytes.sum.peak_sustained * dram__cycles_elapsed.avg.per_second
```

此处使用 NCU 持续峰值与采样时钟；Tensor Core 等路径需使用对应计数和峰值。普通浮点计数包含编译器展开和冗余运算，不包含特殊函数指令本身。减少冗余可能同时降低耗时与硬件 GFLOP/s，优化效果仍按相同输入下的正确性和耗时评价。

## 指标与判断顺序

1. **看模型限制侧**：`AI < AI_ridge` 为内存侧，否则为计算侧；这只说明哪个模型上限较低。
2. **看资源是否饱和**：结合对应存储层级或执行管线利用率，判断实际限制来源。
3. **两侧利用率都低时查等待**：检查网格规模、资源占用、访存延迟、依赖链与同步。

| 指标 | 用途与注意事项 |
| --- | --- |
| DRAM、L2、L1 吞吐与命中率 | 定位存储层级；总项 Memory Throughput 不等于 DRAM 吞吐 |
| SM、FMA、XU 利用率 | 定位执行资源；XU 包含特殊函数和类型转换，不是纯指数指标 |
| occupancy、寄存器、waves/SM | 检查并行度和资源限制；高占用率不保证高吞吐 |
| eligible warps、issue active | 检查是否有可发射工作；活跃 warp 多仍可能长期等待 |
| long scoreboard、barrier、wait | 分别辅助定位 L1TEX 数据依赖、block 同步和固定延迟依赖；按具体指标区分周期与百分比 |

Roofline 未覆盖所有特殊函数、依赖、同步和启动开销。需要细分时，用分层 Roofline 比较 L1/L2/DRAM，并结合源码视图定位热点。

## 测量条件

- 记录工具版本、计数单位、缓存策略与时钟；不同架构的指标可用性可能不同。
- NCU 重放和缓存控制会改变执行条件；采样时使用 NCU kernel duration，应用日志中的 event 时间可能受 profiler 干扰。
- 普通连续调用与清缓存采样分别报告。工作集跨越缓存容量时，重新检查实际流量；DRAM 字节数不必等于逻辑输入输出字节数。
- 输入规模与重复测量规则见 [基准公共配置](../../../benchmarks/README.md)。

依据：[NVIDIA NCU Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)、[waves 定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#launch-metrics)、[CUDA Occupancy API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html)。
