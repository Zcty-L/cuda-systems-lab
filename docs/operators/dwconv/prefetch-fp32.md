# DWConv FP32：baseline 与 prefetch 对比

> 当前 FP32 baseline 已删除循环内冗余同步，和 prefetch 一样仅在权重初始化后同步。下文保留修改前的历史测量，不代表当前版本性能。

新版 prefetch 保持 baseline 的输入加载语义，实测大卷积核耗时小幅下降；尚无证据证明输入输入预取本身带来稳定加速。本文合并原加载时机对照与原因诊断，数据采于 2026-10-10、代码整理之前。

## 实现差别

两个入口均在 [dwconv.cu](../../../ops/dwconv/dwconv.cu)，使用 NCHW、每线程计算 4 通道的一个输出位置，block=256，Kh×Kw≤128。

后续已将 `Conv2DParam` 全部字段统一为 `int`，当前两版均使用 signed 除余。本文性能、指令数和资源数据来自改类型前的 unsigned Kw 版本，不代表改类型后的结果；尚未重新测速。

| 项目 | baseline | prefetch |
| --- | --- | --- |
| 权重 | 一次加载到 2 KiB shared | 相同 |
| 输入 | 每次加载 4 tap，随后计算 | A/B 两份 4×4 寄存器片段，提前加载下一片段 |
| 循环 | 每轮 4 tap，有 block 屏障 | 每轮展开 8 tap，仅初始化后同步 |
| 地址与边界 | signed Kw 除余、空间 guard、普通 C++ 加载 | 相同；尾 tap 依靠零权重，输出尾线程不写回 |
| 乘加 | 每片段四通道，按 tap 顺序累加 | 相同 |
| 寄存器/thread、blocks/SM | 40、6 | 60、4 |

命名采用 `prefetch`（输入寄存器预取）：两组缓冲是实现手段，名称突出提前加载输入的行为，不承诺硬件执行时完全重叠。

预取顺序为 `加载 A → 加载 B → 计算 A → 加载下一 A → 计算 B`。无数组复制、无异步拷贝指令；由编译器调度独立加载与计算。两者 local memory 均为 0。

当前仅保留一个 prefetch 实现，删除 serial 与旧 unsigned 诊断实现。历史 serial 与 prefetch 共用加载、计算和两轮展开，仅改变加载位置，用于区分预取与去屏障等改动的贡献。

## 测量与结果

设备 RTX 5070 Ti Laptop，SM120、46 SM、36 MiB L2；CUDA 13.2，NCU 2026.1.1，CMake Release。配置 C=64、H=W=128、stride=1、padding=K/2、grid=(64,16,N)。

历史测量统一 shared carveout=100%，每版本 20 次直接预热和一次 100 节点 CUDA Graph 预热；12 轮轮换六版本顺序。CUDA Event 测量每轮 100 次 kernel 的平均时间，下表取中位数，单位 µs。分配、上传、校验和构图不计时。设备承担显示且未锁频；NCU 另行采集，不使用 replay 耗时作为性能结果。

| 扫描 | N | K | baseline | prefetch | prefetch 耗时变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 7 | 693.07 | 688.45 | -0.67% |
| 1 | 8 | 11 | 1415.43 | 1353.41 | -4.38% |
| 1 | 16 | 3 | 392.08 | 370.32 | -5.55% |
| 1 | 16 | 7 | 1304.50 | 1259.08 | -3.48% |
| 1 | 16 | 11 | 2774.89 | 2719.73 | -1.99% |
| 1 | 32 | 7 | 2464.69 | 2445.48 | -0.78% |
| 1 | 32 | 11 | 5576.91 | 5438.47 | -2.48% |
| 2 | 16 | 3 | 528.93 | 519.33 | -1.81% |
| 2 | 16 | 7 | 1329.72 | 1298.58 | -2.34% |
| 2 | 16 | 11 | 2848.22 | 2789.88 | -2.05% |

耗时变化为 `prefetch / baseline - 1`。第二轮 N=16、K=7 的 baseline/prefetch 范围为 1216.0–1718.2 / 1173.8–1632.4 µs，K=11 为 2804.1–3428.8 / 2717.5–3219.3 µs。波动大于中位数差异，不能据此承诺稳定加速。

## 为什么收益较小

**输入量与乘加量没有增加。** N=16、K=7 的动态 warp 指令计数如下；计数不是 DRAM 流量。

| 指标 | baseline | prefetch |
| --- | ---: | ---: |
| 总指令 | 306839552 | 303169536（-1.20%） |
| LDG | 27656192 | 27656192 |
| FFMA | 27262976 | 27262976 |
| BAR | 1835008 | 131072 |

**不能只归因于 blocks/SM。** 历史实验用不访问的动态 shared 将 serial/prefetch 同时限制为 4 blocks/SM，机器码不变。K=7 两轮 prefetch 相对 serial 耗时变化为 -0.43% / -1.44%，K=11 为 +0.08% / -1.37%，仍接近持平。K=7 的 NCU 数据说明预取确实隐藏了部分输入延迟：

| 指标 | serial，4 blocks/SM | prefetch，4 blocks/SM |
| --- | ---: | ---: |
| 实际 occupancy | 63.10% | 64.26% |
| long scoreboard / issue active | 1.460 | 0.527 |
| short scoreboard / issue active | 0.378 | 0.966 |
| math pipe throttle / issue active | 1.583 | 1.876 |
| issue active | 72.30% | 74.27% |
| 总指令 | 295960576 | 303169536 |

等待 ratio 不是总耗时百分比：某个 warp 等待时，其他 warp 仍可执行。最终发射利用率相对提高 2.72%，指令量增加 2.44%，净改善很小。相比 baseline 的变化还包含去屏障与循环展开，不能全部算作输入预取收益。

**剩余开销有具体指令证据。** K=7 prefetch 的整数指令占 58.50%，FFMA 占 8.99%；FMAHeavy 活跃率约 80.86%，DRAM 吞吐约峰值的 20.62%。FMAHeavy 也执行整数乘加，不能等同于有效卷积 FLOPS。PC 采样的 math-throttle 热点包含 IMAD，支持索引计算仍有执行管线压力。

输入提前读取后，shared 权重仍临近使用才加载。历史 SASS 在 `LDS.128 R20` 后仅隔两条 FFMA 就消费 R20；该消费位置 1831 个样本中有 1510 个 short-scoreboard、0 个 long-scoreboard。serial/prefetch 的 shared excessive wavefront 均为 0，因此该热点支持 LDS→FFMA 依赖等待，未显示 bank conflict。short-scoreboard 并不全来自 shared，这里只对明确寄存器依赖的热点归因。

综上，输入延迟有所隐藏，但索引运算和权重消费等待仍在，发射效率改善有限。指标语义见 NVIDIA [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)；采样支持上述解释，不能精确分配各因素的时间贡献。

## 旧版诊断与数据留存

重写前 prefetch 改用了 signed 除余并增加 guard/地址选择，K=7/11 指令比 baseline 多 23.1%/24.7%，耗时慢约 12%–13%。仅恢复 unsigned 除余后，指令减少约 6%，两轮耗时分别改善 6.5%/3.5%（K=7）、8.3%/7.8%（K=11）。同一 baseline 限制到 4 blocks/SM 后，K=7 仅慢 1.5%/1.8%，K=11 接近持平；K=3 则慢 5.8%/8.2%。这些结果排除了“大核退化主要来自驻留量降低”的解释，但不能推广到所有配置。

新版恢复 baseline 加载语义后，大幅增加指令的现象已消失。旧版范围数据见[历史测量](prefetch-benchmark.md)，不能混用为当前版本结果。

本地原始日志及 NCU/SASS 保留在忽略目录 `build/dwconv-matched/`：`run1.log`、`run2.log`、`profile.ncu-rep`、`source.ncu-rep`、`source.csv`；旧诊断日志与消融数据位于 `build/dwconv-diagnose/`。删除诊断入口后，历史六版本实验不能直接用当前源码重跑。

## 当前验证与运行

整理后已通过 17 组 CPU 参考正确性测试及独立 prefetch 测试，覆盖非方形核、stride、输出尾块及 tap 尾部；memcheck 为 0 errors。两个版本 `--profile` 单次校验通过，寄存器与驻留量和表中一致。整理后的基准仅比较 baseline/prefetch，9 轮、默认 carveout；与上述历史六版本测量口径有区别，重新测量应单独记录。

```bash
cmake --build build --target dwconv_test dwconv_bench -j 8
./build/tools/device_query
ctest --test-dir build -R '^dwconv_test$' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/tests/dwconv_test --kernel prefetch --check-cpu --warmup 0 --iterations 1
./build/benchmarks/operators/dwconv_bench --large-batch
ncu --set full --clock-control none ./build/benchmarks/operators/dwconv_bench --profile
```

另见 [FP16 prefetch 对比](prefetch-fp16.md)。
