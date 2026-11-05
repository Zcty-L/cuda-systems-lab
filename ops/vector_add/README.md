# Vector Add 示例

接口：输入 `a`、`b` 与输出 `out` 都是设备端 FP32 连续数组，元素数为 `count`，计算 `out[i] = a[i] + b[i]`。传入的 `stream` 决定 kernel 所在 CUDA 流；调用者管理内存与同步。

本模块是算子开发的最小完整示例，展示原生实现、PyTorch 接入、Python API、正确性测试和多后端对比；kernel 使用每个线程计算一个元素的基础实现。

| 环节 | 入口 |
| --- | --- |
| 原生 API | [vector_add.cuh](vector_add.cuh) |
| 原生实现与目标 | [vector_add.cu](vector_add.cu)、[CMakeLists.txt](CMakeLists.txt) |
| Tensor 适配与注册 | [vector_add.cc](../../bindings/torch/vector_add.cc)、[register.cc](../../bindings/torch/register.cc) |
| Python API | [vector_add.py](../../python/cuda_lab/vector_add.py) |
| 正确性 | [原生测试](../../tests/native/vector_add_test.cu)、[Python 测试](../../tests/python/test_vector_add.py) |
| 性能对比 | [Vector Add 基准](../../benchmarks/operators/vector_add/) |

Python 的 `vector_add(a, b)` 分配并返回新张量；`vector_add_out(a, b, out)` 写入预分配输出并返回 `out`。输入必须是同设备、同 shape、连续的 FP32 CUDA 张量，不广播、不隐式转型。支持标量、空张量和尾块；输出缓冲区必须独立于输入。当前 Python 示例仅提供前向接口，不注册 FakeTensor 或 autograd；预分配输出接口拒绝需要梯度的张量。二者均使用当前 stream。

构建、测试、打包与后续算子接入见[开发指南](../../docs/guides/operators/development.md)。
