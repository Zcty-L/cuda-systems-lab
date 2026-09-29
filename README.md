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
| `tools/` | 设备查询等公共辅助工具 |
| `docs/` | 方法、架构笔记和实验报告 |

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

## 算子入口

设备信息查询：构建后运行 `./build/tools/device_query`，输出设备、SM、存储层次与执行限制。详见[设备查询指南](docs/guides/device-query/usage.md)。

可执行文件分别位于 `build/tests/` 和 `build/benchmarks/operators/`。

| 算子 | 正确性测试 | 性能基准 |
| --- | --- | --- |
| [vector_add](ops/vector_add/) | `vector_add_test` | `vector_add_bench` |
| [softmax](ops/softmax/README.md) | `softmax_test` | `softmax_bench`、`softmax_cluster_bench` |

存储微基准：[运行入口](benchmarks/memory/README.md) · [导入报告](docs/benchmarks/memory/baseline.md)，可执行文件位于 `build/benchmarks/memory/`。

测量规范见 [benchmarks](benchmarks/README.md)，报告见 [文档索引](docs/README.md)。

## 新内容放在哪里

- 要被其他代码调用的实现放 `ops/{name}/`，正确性测试放 `tests/` 或模块内。
- 测带宽、延迟、吞吐或算子对比的程序放 `benchmarks/`。
- 为理解一条指令或一个架构机制写的最小程序放 `studies/`。
- 设备查询等公共辅助程序放 `tools/`。
- 正式文档按 `docs/<大类>/<主题>/<文档>.md` 归档；分类与索引见 [文档规则](docs/README.md)。

贡献与分支约定见 [AGENTS.md](AGENTS.md)。
