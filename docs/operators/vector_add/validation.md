# Vector Add 框架验证

## 目的与环境

以 Vector Add 验证原生算子、PyTorch 接入、Python 包、正确性测试和多后端测量的完整流程。验证日期为 2026-11-02。

| 项目 | 配置 |
| --- | --- |
| 设备 | NVIDIA GeForce RTX 5070 Ti Laptop GPU，SM120，46 SM |
| CUDA Toolkit | 13.2.78 |
| Python | Python 3.11.15 |
| PyTorch | 2.13.0+cu130 |
| 构建目标 | `vector_add`，默认 `native` 架构 |

## 初始完整实现的验证结果

| 环节 | 结果 |
| --- | --- |
| 原生构建与 CTest | 通过；包含空输入、单元素、尾块和整块 |
| 仅构建工具 | `CUDA_LAB_OPS` 为空时成功构建设备查询工具 |
| CMake 直接安装 | 正常加载并通过数值、autograd、opcheck 和动态 shape 编译验证 |
| editable 安装 | 从当前 worktree 加载 Python 源码，从安装位置加载二进制；pytest 通过 |
| wheel | 构建成功；独立目录安装后，关闭 editable 路径注入，pytest 通过 |
| Event / Graph 对比 | 每种模式均为 4 组输入 × 2 个后端，全部正确性检查通过并生成结构化结果 |
| 边界与失败处理 | 验证空输入 skip、标量和尾块；注入错误计算后记录 error、无成功耗时和加速比，返回非零 |

初始 Python 测试覆盖 FP32 数值、空张量、标量、尾块、错误 shape/dtype/layout/device、输出完全或部分重叠、预分配输出、反向传播、非默认 stream、CUDA Graph、FakeTensor/schema 契约和 `aot_eager` 完整图动态 shape。

## 测量配置与复现

多后端对比使用元素数 `256`、`1003`、`1048576`、`16777216`，固定 seed=42。预热 20 次，每组 100 次调用，重复 5 组，分别运行 Event 和 Graph 模式，并测量 API 批量摊销耗时。输入和输出预分配，正确性参考为 FP64 加法后转 FP32。

```bash
cmake --preset native
cmake --build --preset native --parallel 4
ctest --preset native
conda activate py311
python -m pip install --no-build-isolation --no-deps -e . -Cbuild.tool-args=-j4
python -m pytest -q
python -m benchmarks.operators.vector_add.benchmark --mode event --api-time
python -m benchmarks.operators.vector_add.benchmark --mode graph --api-time
```

依赖准备和环境路径见[构建指南](../../guides/build/usage.md)。原始结果默认保存在 `build/results/vector_add/`，每次运行包含环境、命令、用例、构建信息、逐组计时样本、JSONL 和 CSV 汇总。

本轮验证的是接口与测量流程；默认用例用于延迟和实现对比。吞吐平台分析继续按[公共规范](../../../benchmarks/README.md)记录实际 occupancy、waves、缓存和时钟条件；Event、Graph 和 API 批量计时分别解读。
