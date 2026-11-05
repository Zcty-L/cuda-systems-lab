# 新增算子：以 Vector Add 为例

## 分层和接口

```text
Python API                python/cuda_lab/vector_add.py
    ↓ torch.ops.cuda_lab.vector_add
PyTorch schema / CUDA     bindings/torch/register.cc、vector_add.cc
    ↓ 指针、shape、stream
原生启动接口               ops/vector_add/vector_add.cuh
    ↓
CUDA kernel               ops/vector_add/vector_add.cu
```

原生层不引用 PyTorch；调用者管理内存和同步。接入层验证输入、设置输入设备、取得当前 stream 并分配输出；Python 层提供调用 `torch.ops` 的公共 API。当前示例不注册 FakeTensor 或 autograd，输入检查由接入层负责。

简单算子直接平铺文件；多架构、多策略实现增多后，在 `ops/{name}/` 下添加 `generic/`、`sm80/`、`sm120/` 等目录，并保留稳定的启动接口。SM 专用目标显式声明 CUDA 架构与 Toolkit 要求，不能仅凭目录名称推断兼容性。

## 接入步骤

1. 在 `ops/{name}/` 添加启动接口、实现与 `CMakeLists.txt`。简单目标调用 `cuda_lab_add_operator(name SOURCES ...)`，公共依赖和 PIC 由辅助函数统一设置；专用目标自行声明架构和可选依赖。
2. 在 `ops/CMakeLists.txt` 的显式目录表中注册算子。测试和基准按 `if(TARGET cuda_ops_<name>)` 注册，确保只构建所选算子。
3. 在 `bindings/torch/` 添加 schema 和 Tensor 适配实现，链接已注册的原生目标。新增多个可选算子时，将各自源文件、注册与 Python 模块都按构建选择启用。
4. 在 `python/cuda_lab/` 添加调用 `torch.ops` 的公开接口，并在 `__init__.py` 显式导出。后续需要 FakeTensor 或 autograd 时，先加载库中的 schema，再添加相应注册。
5. 在 `tests/native/` 和 `tests/python/` 添加正确性测试。
6. 在 `benchmarks/operators/{name}/` 添加用例与后端准备函数，复用 `benchmarks/common/` 的计时和环境记录。正确性参考与性能基线明确精度、布局和算子语义，失败不能继续输出成功耗时。
7. 更新根 README、模块 README、基准入口与文档索引；正式算子说明、基线和优化报告放 `docs/operators/{name}/`。

测试使用安装后的 `cuda_lab`，不拼接 `sys.path`。安装与命令见[构建指南](../build/usage.md)，测量要求见[基准规范](../../../benchmarks/README.md)。
