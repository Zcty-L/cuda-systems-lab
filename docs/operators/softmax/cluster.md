# Softmax Cluster 对照

## 实现与选择

新增显式实验入口 `launch_softmax_variant`，自动入口 `launch_softmax` 保持原分派。查询与调用示例：

```cpp
cuda_lab::SoftmaxVariantInfo info{};
auto variant = cuda_lab::SoftmaxVariant::Cluster2;
// 检查返回值后再分配、启动；接口位于 softmax_experimental.cuh。
auto status = cuda_lab::query_softmax_variant(variant, cols, info);
if(status == cudaSuccess)
    status = cuda_lab::launch_softmax_variant(input, output, rows, cols, variant, stream);
```

| 实现 | 每行配置 | 最大 cols | 本机 registers/thread | 本机 local bytes/thread | 设备驻留容量 |
| --- | --- | ---: | ---: | ---: | ---: |
| online | 1 × 256 threads | 流式处理 | 34 | 0 | 276 blocks |
| block512 | 1 × 512 threads，每线程 32 元素 | 16384 | 82 | 0 | 46 blocks |
| block1024 | 1 × 1024 threads，每线程 32 元素 | 32768 | 64 | 64 | 46 blocks |
| cluster2 | 2 × 256 threads，每线程 16/32 元素 | 16384 | 48/80 | 0 | 46 clusters |
| cluster4 | 4 × 256 threads，每线程 16/32 元素 | 32768 | 48/80 | 0 | 23 clusters |

Cluster 每个 block 处理 `ceil(cols / K)` 列，按分段大小选择 16/32 元素模板。分段尾部和空分段均参与同步。局部计算 `m_b=max(x)`、`e_i=exp(x_i-m_b)`、`s_b=sum(e_i)`，指数保留在寄存器；通过 DSM 交换 `(m_b,s_b)`，各 block 合并得到：

`m=max(m_b)`，`s=sum(s_b*exp(m_b-m))`，`y_i=e_i*exp(m_b-m)/s`。

两次 `cluster.sync()` 分别保证统计量可见、远端读取完成；只交换统计量。扩大线程数的单 block 复用原归约过程，单独设置 `launch_bounds`。1024 线程对照需要限制寄存器才能启动，当前编译结果包含 64 bytes/thread 栈空间（计入 local memory）。

Cluster 要求 CUDA ≥ 12、SM90+，查询 `cudaDevAttrClusterLaunch` 和 cluster occupancy API；不支持时查询返回 `cudaErrorNotSupported`。各显式入口要求正尺寸、有限 FP32 输入、连续且不重叠的设备缓冲区，`rows * cluster_blocks <= INT_MAX`，支持指定 stream。设备与构建的选择依据见 [NVIDIA Cluster 模型](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters)。

## 测量配置

2026-09-29，RTX 5070 Ti Laptop GPU，SM12.0、46 SM、36 MiB L2；CUDA 13.2，Linux 驱动接口 595.58.04 / 宿主驱动 596.21，Release，所有对照编译为 SM120。

- CUDA event 包围含 100 次 kernel 的 CUDA Graph，除以 100；排除分配、传输、图构建及实例化，每轮预热 20 次。重复 5 轮并轮换实现顺序，下面报告中位数。
- 相同形状使用相同输入与缓冲区，随机种子 2026，范围 `[-10,10]`；复用缓存、不主动清空，时钟自适应。图计时仍包含节点调度开销，不能直接与旧基准的主机连续提交计时比较。
- rows 为 `1,8,32,115,230,460,512,920,1380,2760,5520`，覆盖各实现自身的 5/10/20 waves；每种形状运行全部适用实现。Cluster waves 使用设备 active clusters，与普通 blocks 分开计算，见 [公共规则](../../../benchmarks/README.md)。
- 随 rows 扩大，工作集跨过 L2 容量；所有实现均未在自身 5→10→20 waves 上满足“连续两次吞吐变化 ≤5%”。本轮没有确认吞吐平台，不把 waves 达标等同于饱和。
- AI/GFLOP/s 算法估算沿用 [基础口径](arithmetic-intensity.md)。Cluster 每行计 `3C + K(3K+2)` FLOPs、`8C` bytes，额外项为各 block 合并统计量及计算系数；不含 exp、max、整数指令及冗余归约。单 block 的名义访存量不包含溢出流量；实际硬件计数单列。

## CUDA event 对照

单位为 µs，表中为 5 轮测量的中位数。

| cols | rows | online | block512 | cluster2 | cluster4 | block1024 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16384 | 1 | 8.300 | 4.223 | 4.142 | 2.938 | 5.221 |
| 16384 | 32 | 8.678 | 5.270 | 5.723 | 5.762 | 7.590 |
| 16384 | 230 | 26.710 | 29.200 | 27.840 | 33.888 | 44.681 |
| 16384 | 512 | 202.063 | 169.720 | 173.851 | 226.800 | 308.046 |
| 16384 | 5520 | 1981.886 | 1599.603 | 1751.294 | 2351.980 | 2429.092 |
| 32768 | 1 | 44.585 | — | — | 6.598 | 17.098 |
| 32768 | 32 | 17.755 | — | — | 8.751 | 10.327 |
| 32768 | 230 | 203.824 | — | — | 144.449 | 163.344 |
| 32768 | 512 | 474.427 | — | — | 333.391 | 380.176 |
| 32768 | 5520 | 5060.111 | — | — | 3463.719 | 3553.340 |

- `1×16384`：cluster4 相对 online 为 2.82×，相对 block512 为 1.44×；少 rows 时行内并行有收益。
- `32×16384`：block512 的中位数低于两种 cluster；`230×16384` 则 online 更低，收益不能只按 cols 判断。
- `512×16384`：cluster2 与 block512 各轮范围分别为 125.709～362.142、123.582～205.012 µs，重叠明显，不能确认稳定领先。
- `512×32768`：cluster4 相对 online 的中位数比值为 1.42×；各轮范围为 320.463～473.620、389.231～501.134 µs，仍有重叠。整体测量波动明显，不将中位数比值视为固定加速比。

## NCU 诊断

另采样 `512×16384/32768`，`--cache-control all --clock-control base`，取每种实现首次启动的一次 kernel；图内 event 输出不用于分析。方法见 [公共 Roofline](../../benchmarks/methodology/roofline.md)。

| cols | 实现 | 耗时（µs） | 硬件 AI | GFLOP/s | Roofline 利用率 | 下界（µs） |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16384 | online | 216.160 | 3.468 | 947.824 | 63.340% | 136.917 |
| 16384 | block512 | 180.800 | 2.106 | 618.025 | 68.028% | 122.995 |
| 16384 | cluster2 | 178.912 | 2.177 | 618.039 | 65.810% | 117.742 |
| 16384 | cluster4 | 196.544 | 1.810 | 570.613 | 46.986% | 92.348 |
| 16384 | block1024 | 259.232 | 1.887 | 473.573 | 58.158% | 150.763 |
| 32768 | online | 633.344 | 2.154 | 641.513 | 69.016% | 437.110 |
| 32768 | cluster4 | 402.464 | 1.840 | 549.620 | 69.202% | 278.513 |
| 32768 | block1024 | 446.048 | 1.706 | 500.907 | 68.052% | 303.543 |

- 所有样本位于标量 FP32 / DRAM Roofline 的 memory 侧；DRAM 吞吐约为上限的 47%～69%，不能仅据模型判定已经带宽饱和。
- 32768 列：online/cluster4 的 long-scoreboard 等待约为 60.81/4.49 cycles/issue，实际 occupancy 约为 91.53%/32.23%。Cluster 通过少读一遍输入、减少 online 更新指令降低耗时，较低 occupancy 并未阻止收益；仍存在访存与同步开销。
- 16384 列的 cluster4 样本 DRAM 时钟约 13.98 GHz，其余约 8.99 GHz；即使请求 base clock，也不能视为严格同频。每个样本的 Roofline 上限使用自身计数计算，这些单次诊断不用于确认微小性能差异。
- 硬件 GFLOP/s 包含 exp 展开的普通浮点指令和冗余计算；减少指令后它可能下降，性能比较以相同工作量的耗时为主。

**决定：保留显式 2/4-block cluster 和单 block 对照，暂不修改自动分派。** 当前数据支持 cluster 的可行性和特定形状的收益，尚不足以建立跨设备、跨 rows 的稳定阈值。

## 验证与复现

CTest 2/2 通过；Softmax 共 111 组 FP32、9 组 INT8 精度用例及参数检查。新增覆盖 2/4 blocks、模板切换边界、奇数列、空分段、偏移地址、1/257 行、均匀与极端输入，全部使用非阻塞 stream。FP32 逐元素容差 `1e-6 + 1e-5*abs(reference)`，行和容差 `2e-5`。Memcheck 为 0 errors，racecheck 为 0 errors / 0 warnings；关闭 cluster 构建及默认 SM90 PTX 在本机运行也通过。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCUDA_LAB_SOFTMAX_CLUSTER_ARCHITECTURES=120
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/tests/softmax_test
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/tests/softmax_test

# 默认取各实现 5/10/20 waves 行数的并集，附加少行用例。
./build/benchmarks/operators/softmax_cluster_bench --output build/cluster.csv
# 单形状对照；也可用 --variant cluster2/cluster4/block512/block1024/online。
./build/benchmarks/operators/softmax_cluster_bench --rows 512 --cols 16384 --output build/cluster-512.csv

conda activate py311
python benchmarks/operators/softmax_roofline.py --rows 512 --cols 16384 \
  --variants online block512 cluster2 cluster4 block1024 --output build/cluster-ncu-16384
python benchmarks/operators/softmax_roofline.py --rows 512 --cols 32768 \
  --variants online cluster4 block1024 --output build/cluster-ncu-32768
```

本次 event 结果来自最终代码的完整默认扫描。采样文件与日志仅保存在本地 `build/` 下。

Cluster 对象目标独立指定架构，默认 `90`（含 compute_90 PTX，可在后续支持设备上 JIT）；本机实验显式设置 `120`。可用 `-DCUDA_LAB_ENABLE_SOFTMAX_CLUSTER=OFF` 关闭，保留基础算子与单 block 对照，测试会标明 cluster 跳过；CUDA < 12 默认关闭。
