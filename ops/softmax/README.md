# Softmax 算子

接口见 `softmax.cuh`。输入为行主序连续设备矩阵，按行计算 softmax，要求 `rows > 0`、`cols > 0`。调用者管理设备内存和同步；函数返回 CUDA 错误码，支持传入 CUDA stream。输入和输出使用不重叠的缓冲区，FP32 输入应为有限数值。

FP32 → FP32 通过 `launch_softmax` 按列数自动选择实现：

| 列数 | 实现 | 配置 |
| --- | --- | --- |
| 1～1024 | warp softmax | 每个 block 4 个 warp，每个 warp 依次处理 2 行 |
| 1025～8192 | block softmax | 每行一个 256 线程 block，寄存器保存输入，分别归约最大值与指数和 |
| 大于 8192 | online-softmax | 每行一个 256 线程 block，第一遍累计局部最大值与指数和，归约后第二遍读取输入并归一化 |

block 与 online 路径要求 `rows <= INT_MAX`。online 使用对齐的 `float4` 读取，处理不对齐前缀和尾部；输出地址不满足向量对齐时使用标量写回。

实验接口见 `softmax_experimental.cuh`：显式选择 online、512/1024 线程单 block、2/4-block cluster，并查询资源与驻留容量。Cluster 需要 SM90+，构建配置、限制及结果见 [Cluster 对照](../../docs/operators/softmax/cluster.md)；当前不参与自动分派。

INT8 → FP32、INT8 → INT8 继续支持 `1 <= cols <= 1024`，使用 warp 实现。INT8 输入按 `(q - input_zero_point) * input_scale` 解释，其中 `input_scale` 必须为有限正数，`input_zero_point` 位于 `[-128, 127]`。INT8 输出固定使用 `scale=1/256`、`zero_point=-128`，按最近偶数舍入后饱和到 INT8 范围。

从仓库根目录执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build -R softmax_test --output-on-failure
./build/benchmarks/operators/softmax_bench
```

性能基线与环境见 [导入基线](../../docs/operators/softmax/baseline.md)，宽行实现的验证与测量见 [block 与 online-softmax](../../docs/operators/softmax/block-online.md)。

基准输出平均耗时、计算强度 AI 与 GFLOP/s；估算口径见 [AI 计算说明](../../docs/operators/softmax/arithmetic-intensity.md)。可用 `softmax_bench --rows 4096 --cols 1024` 单独运行一种 FP32 配置；默认 512 行，仅传 `--rows` 时会以该行数运行全部 FP32 和 INT8 配置。硬件上限和瓶颈采样使用 `benchmarks/operators/softmax_roofline.py`，支持 `--rows`、`--cols` 列表的所有组合；复现方法见 [Roofline 分析](../../docs/operators/softmax/roofline.md)，增大行数的实验见 [行数与并行度](../../docs/operators/softmax/row-scaling.md)。
