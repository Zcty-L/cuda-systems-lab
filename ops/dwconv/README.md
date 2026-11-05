# Depthwise Conv2D

支持 FP32/FP16 前向，以及 PyTorch 原生 CUDA 反向。

<a id="pytorch-原生-fp32fp16-前向与反向"></a>

## 接口

| 实现 | 入口与说明 |
| --- | --- |
| 公共参数 | [common.h](common.h)：`Conv2DParam` |
| FP32 | [dwconv_fp32.cuh](dwconv_fp32.cuh)：baseline、输入预取、权重 L2 实验 |
| FP16 | [dwconv_fp16.cuh](dwconv_fp16.cuh)：baseline |
| opt | [dwconv_opt.cuh](dwconv_opt.cuh)：FP32/FP16 优化实现 |
| PyTorch native | [dwconv_pytorch.cuh](dwconv_pytorch.cuh)：抽离的前向与反向；当前选择器固定为 `CudaDepthwise2d`，不包含完整 Torch 后端分派；许可见 [PYTORCH-LICENSE](PYTORCH-LICENSE) |
| cuDNN | [dwconv_cudnn.cuh](dwconv_cudnn.cuh)：NCHW/NHWC，Graph、Separate、Fused |
| TensorRT | [dwconv_tensorrt.cuh](dwconv_tensorrt.cuh)：原生卷积及 baseline/opt Plugin |

## 调用约束

- baseline、prefetch、weights-L2 和 opt 使用连续 NCHW，`groups = in_ch = out_ch`、dilation 为 1、`1 <= Kh * Kw <= 128`；FP32 要求 `C % 4 == 0`，FP16 要求 `C % 8 == 0`。
- baseline 的 `weight_stride=0` 按核面积选择 32/64/128，也可指定容量；FP32 prefetch 同样支持。
- FP16 baseline/opt 权重按相邻通道打包为 `half2`；默认 FP32 累加，`accumulate_fp32=false` 改用 FP16 累加。
- baseline/opt 通过 `param.use_bias` 控制 bias，关闭时可传 `nullptr`；直接调用 `launch_pytorch_forward_*` 时以 bias 指针是否为空决定。
- 调用方保证尺寸、派生字段、索引与 grid 有效，并管理设备内存、stream 和同步；具体约束见各头文件。

## 构建与测速

要求 CUDA Toolkit 11.8+；cuDNN、TensorRT 为可选依赖。仓库根目录执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_LAB_OPS=dwconv -DBUILD_TESTING=OFF
cmake --build build --target dwconv_compare_bench device_query --parallel
./build/tools/device_query
./build/benchmarks/operators/dwconv/dwconv_compare_bench --list-cases
./build/benchmarks/operators/dwconv/dwconv_compare_bench --kernel opt --dtype fp16 --case 7
```

后端适配、数据准备、CPU 参考与计时全部位于 [benchmarks/operators/dwconv](../../benchmarks/operators/dwconv/README.md)，批量对比使用 `dwconv_full_bench.py`。`--check-cpu` 可在计时前校验；当前不保留 DWConv 原生测试，待 Python 绑定完成后再添加接口测试。

详细说明与结果见[文档索引](../../docs/README.md)：[cuDNN](../../docs/operators/dwconv/cudnn-execution.md)、[TensorRT](../../docs/operators/dwconv/tensorrt.md)、[FP32 对比](../../docs/operators/dwconv/fp32-full.md)、[FP16 对比](../../docs/operators/dwconv/fp16-full.md)。
