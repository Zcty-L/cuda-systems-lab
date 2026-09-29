# Softmax 导入基线

本次将 `/home/if/Codes/CUDAOP/op/softmax/softmax.cu` 的三个算子接口拆入独立模块，正确性测试和性能测量分别接入 CTest 与 `benchmarks/operators/`。下表是迁入后、进一步优化前的基线，仅代表本次设备与运行条件。

## 环境与复现

- 设备：NVIDIA GeForce RTX 5070 Ti Laptop GPU，SM 12.0。
- CUDA runtime 宏版本：13020；CMake 构建类型：Release，架构由 `CMAKE_CUDA_ARCHITECTURES=native` 决定。
- 输入：512 × 1024 连续行主序矩阵。FP32 输入由固定种子 2026 生成，INT8 输入由固定种子 2027 生成；INT8 输入 scale=1/32、zero point=-3。INT8 输出 scale=1/256、zero point=-128。
- 每项预热 20 次，随后使用 CUDA event 测量 200 次 kernel 调用的总时间并除以次数。计时不含分配、主机设备拷贝及预热。计算强度 AI 按普通浮点运算量除以访存字节数估算，详细口径见 [AI 计算说明](arithmetic-intensity.md)。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/benchmarks/operators/softmax_bench
```

## 正确性与测量结果

CTest：`vector_add_test`、`softmax_test` 共 2/2 通过。Softmax 测试覆盖 9 组 FP32 和 9 组 INT8 配置，包括 1、32、33、1023、1024 列以及不同输入量化参数。

| 路径 | 平均耗时 | AI 估算（FLOP/Byte，不含 exp） | GFLOP/s 估算（不含 exp） |
| --- | ---: | ---: | ---: |
| FP32 → FP32 | 0.015490 ms | 0.375 | 101.541 |
| INT8 → FP32 | 0.022880 ms | 0.600 | 68.744 |
| INT8 → INT8 | 0.022118 ms | 2.000 | 94.817 |

耗时保留原单次运行结果，尚未作多次重复统计；AI 和 GFLOP/s 为后续按统一算法口径补充的估算。硬件 Roofline 采样见 [Roofline 分析](roofline.md)。
