#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include "common.h"

enum class PytorchLayout
{
    NCHW,
    NHWC
};

enum class PytorchDepthwiseKernel
{
    Auto,
    Native,
    Cudnn
};

enum class PytorchForwardBackend
{
    CudaDepthwise2d,
    Cudnn,
    Slow2d // C=1 且不使用 cuDNN 时的 Torch 后端；当前执行接口不支持。
};

// 默认值来源：PyTorch 9a3eac3，aten/src/ATen/Context.h:502-503、522、529、540。
// 参数是独立于 Python 进程的显式配置，不读取 torch.backends 的运行时状态。
// 当前固定使用 CudaDepthwise2d；cuDNN/override/调优选项保留，暂不参与后端选择。
struct PytorchForwardOptions
{
    PytorchLayout input_layout = PytorchLayout::NCHW;
    bool weight_channels_last = false; // 权重实际字节序相同，保留 Tensor 的布局提示用于分派。
    bool cudnn_enabled = true;
    PytorchDepthwiseKernel depthwise_kernel = PytorchDepthwiseKernel::Auto;
    bool benchmark = false;
    bool deterministic = false; // 对应 Torch 两个确定性开关 OR 后的有效值。
    bool allow_tf32 = true;
    size_t workspace_limit_bytes = size_t(256) * 1024 * 1024;
};

// 纯元数据选择器的环境输入，可在不创建 CUDA/cuDNN 上下文时测试后端决策。
struct PytorchDispatchContext
{
    bool cudnn_available = false;
    size_t cudnn_version = 0;
    int device_major = 0;
    bool cudnn_v8_enabled = true;
};

// 当前无条件返回 CudaDepthwise2d；原 eager 分派逻辑及停用原因保留在实现文件的注释中。
PytorchForwardBackend select_pytorch_forward_backend(const Conv2DParam &p, int n, bool fp16,
                                                     const PytorchForwardOptions &options,
                                                     const PytorchDispatchContext &context);

struct PytorchForwardInfo
{
    PytorchForwardBackend backend;
    PytorchLayout output_layout; // 当前 native 输出固定为 NCHW。
    size_t cudnn_version;
    size_t workspace_bytes;
};

// 二维 groups=in_ch=out_ch，multiplier=1、dilation=1；普通连续 NCHW 或 channels-last 输入。
// 当前固定使用 CudaDepthwise2d；原 cuDNN 适配保留，待抽离 Torch 原生执行路径后再启用。
// 构造完成资源准备，bind 只绑定指针；当前 autotune() 不进行调优。
// p.use_bias 对应 Torch optional bias：true 时 bias 必须非空，false 时不读取。
// NHWC 输入在同一 stream 转为 NCHW；调用方按 info().output_layout 解释输出缓冲区。
// 同一对象不支持并发调用，销毁前须由调用方同步 stream。
class PytorchForward
{
  public:
    PytorchForward(const Conv2DParam &p, int n, bool fp16, const PytorchForwardOptions &options = {},
                   cudaStream_t stream = nullptr);
    ~PytorchForward();
    PytorchForward(const PytorchForward &) = delete;
    PytorchForward &operator=(const PytorchForward &) = delete;

    void bind(const void *input, const void *weights, const void *bias, void *output);
    void autotune();
    void launch();
    PytorchForwardInfo info() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// 抽离 PyTorch 原生 CUDA kernel：来源提交及适配范围见 README.md，许可见 PYTORCH-LICENSE。
// 连续 NCHW，in_ch=out_ch，通道倍增=1，dilation=1；权重为普通 [C, Kh, Kw]。
// 指针不重叠；异步启动，接口检查启动错误，调用方负责同步及异步执行错误检查。
// 输出覆盖写入，无需清零；仅调用 CUDA kernel，不经过 PyTorch 框架的 cuDNN 后端选择。
// bias 指针是否为空决定是否加偏置，不读取 p.use_bias。
// bias 可为空；反向的 grad_input / grad_weight / grad_bias 可分别为空以跳过计算。
// 前向保留上游 1/3/5 核尺寸特化和 generic 路径，block=1024。
void launch_pytorch_forward_fp32(const float *input, const float *weight, const float *bias, float *output,
                                 const Conv2DParam &p, int n, cudaStream_t stream = nullptr);

// dX 保留核尺寸 1/3/5/通用与 stride 1/2/通用的 12 种组合，block=1024。
// dWeight 沿用上游 block=min(n*32,256) 及 BlockReduceSum；grad_bias 是本地兼容归约。
// 比较上游原生 backward 时传 grad_bias=nullptr，仅计入 dWeight 和 dX。
void launch_pytorch_backward_fp32(const float *input, const float *weight, const float *grad_output, float *grad_input,
                                  float *grad_weight, float *grad_bias, const Conv2DParam &p, int n,
                                  cudaStream_t stream = nullptr);

// FP16 将上游 c10::Half 适配为 __half；所有乘加与归约使用 FP32，写回时舍入到 FP16。
void launch_pytorch_forward_fp16(const __half *input, const __half *weight, const __half *bias, __half *output,
                                 const Conv2DParam &p, int n, cudaStream_t stream = nullptr);

void launch_pytorch_backward_fp16(const __half *input, const __half *weight, const __half *grad_output,
                                  __half *grad_input, __half *grad_weight, __half *grad_bias, const Conv2DParam &p,
                                  int n, cudaStream_t stream = nullptr);
