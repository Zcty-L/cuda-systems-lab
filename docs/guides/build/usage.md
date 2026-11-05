# 构建、Python 安装与测试

## 依赖与环境

原生部分需要 CMake 3.24+、C++17 编译器、CUDA Toolkit。PyTorch 接入层使用 C++20，需要 Python 3.10+ 和 PyTorch 2.4+；打包使用 `scikit-build-core>=0.10`，接口测试使用 `pytest>=8`。

## 原生构建

在仓库根目录执行：

```bash
cmake --preset native
cmake --build --preset native --parallel 4
ctest --preset native
./build/tools/device_query
./build/benchmarks/operators/vector_add/vector_add_bench
```

`native` preset 使用 `build/`，默认构建示例算子、原生测试、基准和工具。原有 `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` 用法继续有效。正式性能测量前必须先运行设备查询。

已注册 `vector_add`、`softmax` 和 `dwconv`；只构建所选算子及其测试和基准。构建全部算子时执行：

```bash
cmake --preset native -DCUDA_LAB_OPS="vector_add;softmax;dwconv"
cmake --build --preset native --parallel 4
ctest --preset native
```

存储微基准随 `CUDA_LAB_BUILD_BENCHMARKS` 启用，不依赖算子选择。DWConv 的 cuDNN 对比仅在选择 `dwconv` 时查找依赖，缺少 cuDNN 会跳过该后端；可通过 `DWCONV_TEST_CUDNN=OFF` 关闭。Softmax cluster 的架构限制沿用模块配置，见[Cluster 对照](../../operators/softmax/cluster.md)。当前 Python 接口仅提供 Vector Add，启用 Python 时 `CUDA_LAB_OPS` 必须包含 `vector_add`。

| 选项 | 默认值 | 用途 |
| --- | --- | --- |
| `CUDA_LAB_OPS` | `vector_add` | 算子列表，以分号分隔；空值仅保留其他启用目标 |
| `CUDA_LAB_BUILD_PYTHON` | `OFF` | PyTorch 库与 Python 包安装规则 |
| `CUDA_LAB_BUILD_BENCHMARKS` | `ON` | 原生基准 |
| `CUDA_LAB_BUILD_TOOLS` | `ON` | 设备查询工具 |
| `CUDA_LAB_BUILD_STUDIES` | `OFF` | 已注册的架构实验 |
| `BUILD_TESTING` | `ON` | CTest 原生测试 |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | 默认 CUDA 架构，专用目标可自行收窄支持范围 |

例如仅构建显式 SM120 的算子与工具：

```bash
cmake -S . -B build/minimal -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DCUDA_LAB_OPS=vector_add -DBUILD_TESTING=OFF -DCUDA_LAB_BUILD_BENCHMARKS=OFF
cmake --build build/minimal --parallel 4
```

## Python 开发安装

```bash
conda activate py311
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest
```

`pyproject.toml` 默认只构建算子和 PyTorch 库，关闭原生工具、基准和测试；原生验证使用上一节的构建目录。Python 扩展链接同一份 `cuda_ops_vector_add` 目标。`--no-build-isolation` 使用当前 PyTorch，避免在隔离环境中另装一个版本；`--no-deps` 要求运行依赖事先准备好。

需要限制编译并行度或显式架构时：

```bash
python -m pip install --no-build-isolation --no-deps -e . -Cbuild.tool-args=-j4 -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=120
```

修改 CUDA/C++ 代码后再次执行开发安装，以更新二进制。Python 源码由 editable 安装直接引用；普通 wheel 安装需要重新构建并安装。

也可直接使用 CMake 构建并安装到一个独立目录，适合排查打包问题：

```bash
cmake --preset python -DPython_EXECUTABLE="$(command -v python)"
cmake --build --preset python --parallel 4
cmake --install build/python-cmake --prefix "$PWD/build/python-install" --component python
PYTHONPATH="$PWD/build/python-install" python -m pytest
```

这里的 `PYTHONPATH` 指向完整安装目录，不直接指向源码或临时编译文件。常规开发使用 editable 安装。

## Wheel

环境已有 `build` 和构建后端时：

```bash
python -m build --wheel --no-isolation
```

产物位于 `dist/`；wheel 包含 Python 代码、`_C.so` 和 `_build_info.json`。当前示例使用常规 ATen ABI，构建与运行保持相同 PyTorch 版本；默认 `native` wheel 面向本机架构，跨机器发布需显式选择架构并在目标环境验证。通过 `cuda_lab.build_info()` 查看实际加载路径、构建版本、提交和架构配置。

## Python 性能对比

完成原生构建和 Python 安装后：

```bash
python -m benchmarks.operators.vector_add.benchmark --mode event
python -m benchmarks.operators.vector_add.benchmark --mode graph --api-time
```

使用 `--help` 查看用例文件、采样次数、后端、设备与输出目录选项；计时和结果口径见[基准规范](../../../benchmarks/README.md)。
