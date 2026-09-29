# Softmax block 与 online 实现

## 接入范围

在 `feature/softmax` 中接入参考文件 `/home/if/Codes/CUDAOP/op/softmax/softmax.cu` 的 `softmax_block_kernel` 和 `softmax_online_kernel`，继续通过 `cuda_lab::launch_softmax` 自动分派：

- `cols <= 1024`：原 warp 实现。
- `1025 <= cols <= 8192`：每行一个 256 线程 block，按列数使用每线程 8、16 或 32 个寄存器元素，分别归约最大值与指数和。
- `cols > 8192`：每行一个 256 线程 block，在线累计局部最大值及指数和，再归约并第二次读取输入，写出概率。支持 `float4` 对齐读取、标量前缀与尾部，以及输出未对齐时的标量写回。

block 和 online 路径要求 `rows <= INT_MAX`；INT8 的两种输出接口仍使用原 warp 路径，最多支持 1024 列。输入输出须为不重叠的连续设备缓冲区，FP32 输入须为有限数值。

参考代码在读取共享内存中的 block 最大值后，直接复用该内存保存各 warp 的指数和。本次在两种新 kernel 中各增加一次 block 同步，确保所有 warp 读取完成后才开始覆盖共享内存。

## 验证

Release 构建及 CTest 通过，测试总计 2/2。Softmax 包含 32 组 FP32 配置和 9 组 INT8 配置，另有非法参数检查。

- 分派边界：1024/1025、2048/2049、4096/4097、8192/8193，最长测试至 65537 列。
- 对齐：online 输入和输出偏移 1～3 个 float，覆盖相同对齐、不同对齐、标量前缀和尾部。
- 数值：随机输入、全 1000 的相同行、交替出现 ±1000 的输入；CPU 参考和行和检查采用 FP64 累加。
- 所有 FP32 用例在非阻塞 CUDA stream 上执行上传、算子和回传。
- FP32 逐元素容差为 `1e-6 + 1e-5 * abs(reference)`，行和误差不超过 `2e-5`；INT8 输出与 CPU 量化参考最多相差 1。
- Compute Sanitizer：`memcheck` 为 0 errors，`racecheck` 为 0 errors、0 warnings。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/tests/softmax_test
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/tests/softmax_test
./build/benchmarks/operators/softmax_bench
```

## 测量配置与结果

2026-09-29，NVIDIA GeForce RTX 5070 Ti Laptop GPU，SM 12.0，CUDA runtime 宏版本 13020，Release，native 架构。所有配置均为 512 行，预热 20 次、计时 200 次；FP32 随机输入种子为 2026，INT8 为 2027。INT8 输入 scale=1/32、zero point=-3。

CUDA event 测量连续 kernel 调用所在流的时间区间，除以 200 得到平均耗时，不含内存分配、主机设备传输和预热。区间可能包含主机提交不足造成的 GPU 空闲。AI 按普通浮点运算量除以访存字节数估算，其中 online 计入两遍输入读取；详见 [AI 计算说明](arithmetic-intensity.md)。

接入前重新运行原测试与基准，正确性通过；512×1024 的 FP32、INT8→FP32、INT8→INT8 平均耗时分别为 0.014130、0.011075、0.014801 ms。原接口不支持更宽的 FP32 输入，因而没有相同宽行配置的旧实现结果。

接入后单次运行耗时如下，AI 为后续按统一口径补充的理论估算；online 因重缩放次数依赖输入而采用区间：

| 路径 | 列数 | 平均耗时（ms） | AI 估算（FLOP/Byte，不含 exp） | GFLOP/s 估算（不含 exp） |
| --- | ---: | ---: | ---: | ---: |
| FP32 warp | 1024 | 0.012586 | 0.375000 | 124.969 |
| FP32 block | 1025 | 0.025308 | 0.375000 | 62.210 |
| FP32 block | 2048 | 0.014091 | 0.375000 | 223.244 |
| FP32 block | 4096 | 0.014954 | 0.375000 | 420.721 |
| FP32 block | 8192 | 0.026805 | 0.375000 | 469.424 |
| FP32 online | 8193 | 0.025887 | 0.341145～0.424478 | 663.363～825.406 |
| FP32 online | 16384 | 0.211830 | 0.337240～0.420573 | 160.259～199.860 |
| FP32 online | 65536 | 0.793374 | 0.334310～0.417643 | 169.669～211.962 |
| INT8→FP32 | 1024 | 0.011140 | 0.600000 | 141.191 |
| INT8→INT8 | 1024 | 0.013522 | 2.000000 | 155.092 |

这些结果用于记录新增宽行支持的性能，尚未进行多次重复统计或同形状不同 kernel 的比较，不据此推断分派阈值最优或已有路径获得加速。

上表的 GFLOP/s 根据原始耗时和算法运算量补充计算；不能直接与硬件指令计数混用。实测上限、Roofline 利用率及诊断指标见 [Roofline 分析](roofline.md)。
