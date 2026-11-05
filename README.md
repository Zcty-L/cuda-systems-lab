# CUDA Systems Lab

从 GPU 架构原语、性能测量到可复用 CUDA 算子的实验仓库。**cuda-ops** 是其中的算子集合。

## 项目地图

| 目录 | 内容 |
| --- | --- |
| `ops/` | 可复用 CUDA 算子及其参考实现 |
| `bindings/torch/` | PyTorch Tensor 适配、schema 与 CUDA 实现注册 |
| `python/cuda_lab/` | 公开 Python API 和算子库加载 |
| `benchmarks/` | GPU 存储层次、指令和算子性能测量 |
| `studies/` | PTX、TMA、cp.async、cluster 等最小验证实验 |
| `include/cuda_lab/` | 公共工具及可复用的 PTX wrapper |
| `tests/native/`、`tests/python/` | 原生正确性测试与 Python 接口测试 |
| `cmake/` | 公共算子目标与构建辅助代码 |
| `tools/` | 设备查询等公共辅助工具 |
| `docs/` | 方法、架构笔记和实验报告 |

## 构建与测试

需要 CMake 3.24+、支持的 NVIDIA GPU 和 CUDA Toolkit。默认使用当前 GPU 架构；交叉编译时可设置 `CMAKE_CUDA_ARCHITECTURES`。

```bash
cmake --preset native
cmake --build --preset native --parallel 4
ctest --preset native
```

例如指定 SM 120：

```bash
cmake --preset native -DCMAKE_CUDA_ARCHITECTURES=120
```

Python 环境：

```bash
conda activate py311
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest
```

`--no-build-isolation` 复用当前环境中的 PyTorch；`--no-deps` 避免安装过程中替换现有依赖。CMake 需在激活后的环境中可用。构建选项、打包和 worktree 隔离见[构建指南](docs/guides/build/usage.md)。

```python
import torch
import cuda_lab

a = torch.randn(1003, device="cuda", dtype=torch.float32)
b = torch.randn_like(a)
y = cuda_lab.vector_add(a, b)
```

## 算子入口

设备信息查询：构建后运行 `./build/tools/device_query`，输出设备、SM、存储层次与执行限制。详见[设备查询指南](docs/guides/device-query/usage.md)。

原生测试位于 `build/tests/native/`，算子基准位于 `build/benchmarks/operators/{name}/`。

默认选择 `vector_add`；通过 `-DCUDA_LAB_OPS="vector_add;softmax;dwconv"` 启用全部算子，或单独选择所需算子。

| 算子 | 正确性测试 | 性能基准 |
| --- | --- | --- |
| [vector_add](ops/vector_add/) | CTest + [pytest](tests/python/test_vector_add.py) | [原生 / PyTorch 对比](benchmarks/operators/vector_add/) |
| [softmax](ops/softmax/README.md) | `softmax_test` | `softmax_bench`、`softmax_cluster_bench` |
| [dwconv](ops/dwconv/README.md) | 待 Python 绑定后补充 | [多后端测速](benchmarks/operators/dwconv/README.md) |

DWConv cuDNN 的布局、执行路径与候选方案选择见 [cuDNN 初始化与执行方式](docs/operators/dwconv/cudnn-execution.md)。

完成原生构建和 Python 安装后，在仓库根目录运行：

```bash
python -m benchmarks.operators.vector_add.benchmark --mode graph
```

入口会先运行设备查询，再比较预分配输出的 `cuda_lab.vector_add_out` 与 `torch.add(out=)`；结果默认保存到 `build/results/vector_add/`。

存储微基准：[运行入口](benchmarks/memory/README.md) · [导入报告](docs/benchmarks/memory/baseline.md)，可执行文件位于 `build/benchmarks/memory/`。

测量规范见 [benchmarks](benchmarks/README.md)，报告见 [文档索引](docs/README.md)。

## 新内容放在哪里

- 要被其他代码调用的实现放 `ops/{name}/`，原生测试放 `tests/native/`，Python 接口测试放 `tests/python/`。
- Tensor 适配与注册放 `bindings/torch/`，公开 Python 接口放 `python/cuda_lab/`。
- 测带宽、延迟、吞吐或算子对比的程序放 `benchmarks/`。
- 为理解一条指令或一个架构机制写的最小程序放 `studies/`。
- 设备查询等公共辅助程序放 `tools/`。
- 正式文档按 `docs/<大类>/<主题>/<文档>.md` 归档；分类与索引见 [文档规则](docs/README.md)。

贡献与分支约定见 [AGENTS.md](AGENTS.md)。
新增算子可参照 [Vector Add 的完整流程](docs/guides/operators/development.md)。
