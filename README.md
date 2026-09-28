# CUDA Systems Lab

从 GPU 架构原语、性能测量到可复用 CUDA 算子的实验仓库。**cuda-ops** 是其中的算子集合。

## 项目地图

| 目录 | 内容 |
| --- | --- |
| `ops/` | 可复用 CUDA 算子及其参考实现 |
| `benchmarks/` | GPU 存储层次、指令和算子性能测量 |
| `studies/` | PTX、TMA、cp.async、cluster 等最小验证实验 |
| `include/cuda_lab/` | 公共工具及可复用的 PTX wrapper |
| `tests/` | 跨模块正确性和集成测试 |
| `docs/` | 方法、架构笔记和实验报告 |

`vector_add` 给出了算子实现、正确性测试和性能基准的完整构建方式。

## 构建与测试

需要 CMake 3.24+、支持的 NVIDIA GPU 和 CUDA Toolkit。默认使用当前 GPU 架构；交叉编译时可设置 `CMAKE_CUDA_ARCHITECTURES`。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

例如指定 SM 120：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120
```

`vector_add` 可通过 `build/tests/vector_add_test` 验证，通过 `build/benchmarks/operators/vector_add_bench` 测量。微基准分别记录 GPU 型号、SM、CUDA 版本、输入配置、计时范围和原始结果。

## 新内容放在哪里

- 要被其他代码调用的实现放 `ops/{name}/`，正确性测试放 `tests/` 或模块内。
- 测带宽、延迟、吞吐或算子对比的程序放 `benchmarks/`。
- 为理解一条指令或一个架构机制写的最小程序放 `studies/`。
- 正式文档只放 `docs/`；对应原始结果和运行命令一起保存。

贡献与分支约定见 [AGENTS.md](AGENTS.md)。
