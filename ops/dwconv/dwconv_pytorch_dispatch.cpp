// 后端决策抽离自 PyTorch 9a3eac3b8ffba6e54e61df7ebecfb0afcc5e5a74。
// 下文来源行号均对应上述固定提交；许可见 PYTORCH-LICENSE。
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "dwconv_pytorch.cuh"

// 原分派辅助函数暂时保留，待抽离 Torch 原生 cuDNN 执行路径后恢复。
// clang-format off
/*
namespace
{

int64_t numel(int a, int b, int c, int d = 1)
{
    int64_t result = 1;
    for(int value : {a, b, c, d})
    {
        if(value <= 0 || result > std::numeric_limits<int64_t>::max() / value)
            throw std::invalid_argument("PyTorch dispatch 要求正尺寸且元素数不超过 int64 范围");
        result *= value;
    }
    return result;
}

// 来源：aten/src/ATen/native/Convolution.cpp:222-254。
// Tensor 的 shape 访问适配为 Conv2DParam；保留分支顺序和阈值。
bool check_cudnn_depthwise_workload_with_filter(const Conv2DParam &p, int n)
{
    if(p.in_h == 1 && p.Sw == 1)
        return true;
    if(p.Kh != p.Kw)
        return false;
    const int filter = p.Kw;
    if(filter != 1 && filter != 3 && filter != 5)
        return false;
    if(p.in_w < 7)
        return false;
    if(p.Sw == 1)
        return true;
    if(p.Sw != 2)
        return false;
    if(n == 1)
    {
        if(filter == 1 && p.in_w <= 28)
            return true;
        if(filter == 3 || filter == 5)
            return true;
    }
    else
    {
        if(filter == 1 && n <= 16 && p.in_ch >= 128 && p.in_w <= 7)
            return true;
        if(filter == 3 || filter == 5)
            if(p.in_ch >= 512 || (p.in_ch >= 256 && p.in_w >= 28))
                return true;
    }
    return false;
}

} // namespace
*/
// clang-format on

// 来源：aten/src/ATen/native/Convolution.cpp:381-488、646-653、1320-1339。
// 范围固定为 CUDA、FP32/FP16、二维、multiplier=1、dilation=1、非转置；不涉及 Triton/Inductor。
PytorchForwardBackend select_pytorch_forward_backend(const Conv2DParam &p, int n, bool fp16,
                                                     const PytorchForwardOptions &options,
                                                     const PytorchDispatchContext &context)
{
    // 当前仅比较抽离的 ATen native CUDA depthwise 实现，统一使用 CudaDepthwise2d。
    // 原 cuDNN 分支复用本仓库 CudnnForward::Separate(v7)，未抽离 Torch 的 cuDNN 执行代码。
    // Torch 默认走 Conv_v8.cpp 的 plan 搜索/缓存和 cudnnBackendExecute；禁用 v8 才走 v7。
    // 两者即使选择同一个 Cudnn 后端，具体算法、workspace、缓存和 benchmark 策略也可能不同。
    // 直接调用 at::cudnn_convolution 依赖 ATen Tensor、Torch handle/stream 和内存分配器，
    // 需要链接 LibTorch；本项目 ops/ 保持不依赖 PyTorch。后续可独立抽离其 v8/v7 调用逻辑，
    // 完成框架依赖适配后再恢复以下分派；当前保留原代码为注释，不将本地 v7 当作 Torch 基线。
    // 来源：Convolution.cpp:1637-1644；aten/src/ATen/native/cudnn/Conv_v8.cpp:1464-1506。
    return PytorchForwardBackend::CudaDepthwise2d;

    // clang-format off
    /*
    if(p.in_ch <= 0 || p.out_ch != p.in_ch || p.Sh <= 0 || p.Sw <= 0 || p.Ph < 0 || p.Pw < 0 || p.Kh <= 0 || p.Kw <= 0)
        throw std::invalid_argument("PyTorch dispatch 要求 groups=in_ch=out_ch、正 stride/kernel 和非负 padding");

    if(options.input_layout != PytorchLayout::NCHW && options.input_layout != PytorchLayout::NHWC)
        throw std::invalid_argument("PyTorch dispatch 的输入布局无效");

    if(options.depthwise_kernel != PytorchDepthwiseKernel::Auto &&
       options.depthwise_kernel != PytorchDepthwiseKernel::Native &&
       options.depthwise_kernel != PytorchDepthwiseKernel::Cudnn)
        throw std::invalid_argument("PyTorch depthwise kernel 选项无效");

    const int64_t h = int64_t(p.in_h) + int64_t(2) * p.Ph - p.Kh;
    const int64_t w = int64_t(p.in_w) + int64_t(2) * p.Pw - p.Kw;
    if(h < 0 || w < 0 || h / p.Sh + 1 != p.out_h || w / p.Sw + 1 != p.out_w)
        throw std::invalid_argument("PyTorch dispatch 的输出尺寸与卷积公式不一致");

    const int64_t input = numel(n, p.in_ch, p.in_h, p.in_w);
    const int64_t weight = numel(p.out_ch, p.Kh, p.Kw);
    const int64_t output = numel(n, p.out_ch, p.out_h, p.out_w);
    constexpr int64_t limit = std::numeric_limits<int>::max();
    const bool enabled = options.cudnn_enabled && context.cudnn_available;

    // 来源：aten/src/ATen/native/ConvUtils.h:348-374，input 或 weight 的布局提示均参与选择。
    const bool channels_last = enabled && (options.input_layout == PytorchLayout::NHWC || options.weight_channels_last);

    // 来源：aten/src/ATen/native/Convolution.cpp:381-436。
    const bool use_cudnn = enabled && ((input / n <= limit && output / n <= limit) ||
                                       (context.cudnn_version >= 90300 && context.cudnn_v8_enabled));

    // C=1 不属于 Torch 的 is_depthwise，不能把 Slow2d 悄悄替换成 CudaDepthwise2d。
    if(p.in_ch == 1)
        return use_cudnn ? PytorchForwardBackend::Cudnn : PytorchForwardBackend::Slow2d;
    if(!enabled)
        return PytorchForwardBackend::CudaDepthwise2d;

    // 来源：aten/src/ATen/native/Convolution.cpp:444-461。
    // 连续张量的严格边界来源：aten/src/ATen/native/IndexingUtils.cpp:6-30。
    if(!(input < limit && weight < limit))
    {
        if(channels_last && context.cudnn_version > 91000 && context.cudnn_version < 91500)
            return PytorchForwardBackend::CudaDepthwise2d;
        return context.cudnn_version >= 90300 && context.cudnn_v8_enabled ? PytorchForwardBackend::Cudnn
                                                                          : PytorchForwardBackend::CudaDepthwise2d;
    }

    // NHWC 分支在 override 之前，depthwise_kernel=Native 也不覆盖它。
    if(channels_last)
        return PytorchForwardBackend::Cudnn;

    // CUDAHooks 能力条件来源：aten/src/ATen/cuda/detail/CUDAHooks.cpp:288-303。
    if(context.device_major < 7 || !use_cudnn || !fp16 || (p.Sh != p.Sw && p.in_h != 1) || p.in_ch < 32)
        return PytorchForwardBackend::CudaDepthwise2d;

    if(options.depthwise_kernel == PytorchDepthwiseKernel::Native)
        return PytorchForwardBackend::CudaDepthwise2d;

    if(options.depthwise_kernel == PytorchDepthwiseKernel::Cudnn)
        return PytorchForwardBackend::Cudnn;

    return check_cudnn_depthwise_workload_with_filter(p, n) ? PytorchForwardBackend::Cudnn
                                                            : PytorchForwardBackend::CudaDepthwise2d;
    */
    // clang-format on
}
