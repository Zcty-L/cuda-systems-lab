#include "dwconv_pytorch.cuh"
#include <cuda_lab/cuda_check.cuh>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#ifdef DWCONV_PYTORCH_HAS_CUDNN
#include "dwconv_cudnn.cuh"
#include <cudnn.h>
#endif

namespace
{

// 本地布局适配，对应 Torch input.contiguous(memory_format)；卷积 kernel 保留原实现。
template <typename T>
__global__ void convert_input_layout(const T *input, T *output, int64_t count, int channels, int hw, bool to_nhwc)
{
    for(int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += int64_t(blockDim.x) * gridDim.x)
    {
        int64_t source;
        if(to_nhwc)
        {
            const int c = i % channels;
            const int64_t pos = i / channels;
            source = (pos / hw * channels + c) * hw + pos % hw;
        }
        else
        {
            const int pos = i % hw;
            const int64_t nc = i / hw;
            source = (nc / channels * hw + pos) * channels + nc % channels;
        }
        output[i] = input[source];
    }
}

int checked_product(int a, int b)
{
    const int64_t product = int64_t(a) * b;
    if(a <= 0 || b <= 0 || product > std::numeric_limits<int>::max())
        throw std::invalid_argument("PytorchForward 的公共参数及单 batch 大小须在正 int 范围内");
    return int(product);
}

} // namespace

struct PytorchForward::Impl
{
    Conv2DParam p{};
    int n = 0;
    bool fp16 = false;
    PytorchForwardOptions options;
    cudaStream_t stream{};
    PytorchForwardInfo info{};
    const void *input{};
    const void *weights{};
    const void *bias{};
    void *output{};
    void *converted_input{}; // layout 使用 NHWC 触发，默认 NCHW，不触发
#ifdef DWCONV_PYTORCH_HAS_CUDNN
    std::unique_ptr<CudnnForward> cudnn;
#endif

    ~Impl()
    {
        if(converted_input)
            cudaFree(converted_input);
    }

    template <typename T> void convert()
    {
        const int64_t count = int64_t(n) * p.inBatchNumel;
        const int blocks = int(std::min<int64_t>(65535, (count - 1) / 256 + 1));
        convert_input_layout<<<blocks, 256, 0, stream>>>(static_cast<const T *>(input),
                                                         static_cast<T *>(converted_input), count, p.in_ch, p.inHW,
                                                         info.output_layout == PytorchLayout::NHWC);
        cuda_lab::check_cuda(cudaGetLastError(), "PyTorch contiguous conversion");
    }
};

PytorchForward::PytorchForward(const Conv2DParam &p, int n, bool fp16, const PytorchForwardOptions &options,
                               cudaStream_t stream)
    : impl_(std::make_unique<Impl>())
{
    // 本地公共结构体适配：核对输出公式，重新生成 native kernel 所需派生字段。
    if(n <= 0 || p.in_h <= 0 || p.in_w <= 0 || p.Kh <= 0 || p.Kw <= 0 || p.Sh <= 0 || p.Sw <= 0 || p.Ph < 0 || p.Pw < 0)
        throw std::invalid_argument("PytorchForward 要求正尺寸/stride 和非负 padding");
    // 选择器当前直接返回 native；执行接口继续检查 native kernel 的通道和布局前提。
    if(p.in_ch <= 0 || p.out_ch != p.in_ch)
        throw std::invalid_argument("PytorchForward 要求 groups=in_ch=out_ch 且通道数为正");
    if(options.input_layout != PytorchLayout::NCHW && options.input_layout != PytorchLayout::NHWC)
        throw std::invalid_argument("PytorchForward 的输入布局无效");

    const int64_t h = int64_t(p.in_h) + int64_t(2) * p.Ph - p.Kh;
    const int64_t w = int64_t(p.in_w) + int64_t(2) * p.Pw - p.Kw;
    if(h < 0 || w < 0 || h / p.Sh + 1 != p.out_h || w / p.Sw + 1 != p.out_w)
        throw std::invalid_argument("PytorchForward 的输出尺寸与卷积公式不一致");

    auto &self = *impl_;
    self.p = p;
    self.n = n;
    self.fp16 = fp16;
    self.options = options;
    self.stream = stream;
    self.p.inHW = checked_product(p.in_h, p.in_w);
    self.p.outHW = checked_product(p.out_h, p.out_w);
    self.p.KhKw = checked_product(p.Kh, p.Kw);
    self.p.inBatchNumel = checked_product(p.in_ch, self.p.inHW);
    self.p.outBatchNumel = checked_product(p.out_ch, self.p.outHW);
    self.p.kernelWeightsCount = self.p.inChKhKw = checked_product(p.out_ch, self.p.KhKw);
    self.p.biasWeightsCount = p.out_ch;
    PytorchDispatchContext context;

    // 来源：aten/src/ATen/native/ConvUtils.h:96-105；独立于 Python 的显式参数仍优先控制 enabled。
    const char *disabled_v8 = std::getenv("TORCH_CUDNN_V8_API_DISABLED");
    context.cudnn_v8_enabled = !disabled_v8 || std::string(disabled_v8) != "1";

#ifdef DWCONV_PYTORCH_HAS_CUDNN
    context.cudnn_available = true;
    context.cudnn_version = cudnnGetVersion();
    int device;
    cudaDeviceProp prop{};
    cuda_lab::check_cuda(cudaGetDevice(&device), "PyTorch dispatch cudaGetDevice");
    cuda_lab::check_cuda(cudaGetDeviceProperties(&prop, device), "PyTorch dispatch cudaGetDeviceProperties");
    context.device_major = prop.major;
#endif

    self.info.backend = select_pytorch_forward_backend(p, n, fp16, options, context);
    self.info.cudnn_version = context.cudnn_version;
    self.info.output_layout = PytorchLayout::NCHW;

    if(self.info.backend == PytorchForwardBackend::Slow2d)
        throw std::invalid_argument("Torch 对 C=1 且禁用 cuDNN 选择 Slow2d；当前 PytorchForward 未抽离该后端");

    if(self.info.backend == PytorchForwardBackend::Cudnn)
    {

#ifdef DWCONV_PYTORCH_HAS_CUDNN
        // 来源：aten/src/ATen/native/ConvUtils.h:348-374，cuDNN 的建议 memory_format。
        if(options.input_layout == PytorchLayout::NHWC || options.weight_channels_last)
            self.info.output_layout = PytorchLayout::NHWC;
        CudnnForwardOptions config;
        // Torch 普通 forward 为卷积后另加 bias：Convolution.cpp:1637-1644（固定 9a3eac3）。
        // 此处复用本地 v7 Separate 执行器，只保证后端决策和分离 bias，plan 搜索并非 Torch v8。
        config.mode = CudnnForwardMode::Separate;
        config.layout = self.info.output_layout == PytorchLayout::NHWC ? CudnnLayout::NHWC : CudnnLayout::NCHW;
        config.selection = options.benchmark ? CudnnPlanSelection::Autotune : CudnnPlanSelection::Heuristic;
        config.deterministic = options.deterministic;
        config.allow_tf32 = options.allow_tf32;
        config.workspace_limit_bytes = options.workspace_limit_bytes;
        self.cudnn = std::make_unique<CudnnForward>(self.p, n, fp16, config, stream);
        self.info.workspace_bytes = self.cudnn->info().workspace_bytes;
#endif
    }
    else if(int64_t(n) * self.p.inBatchNumel >= std::numeric_limits<int>::max() ||
            int64_t(n) * self.p.outBatchNumel >= std::numeric_limits<int>::max())
        throw std::invalid_argument("选中的 Torch native kernel 不支持当前输入/输出的 32 位索引");

    if(options.input_layout != self.info.output_layout)
    {
        const size_t count = size_t(n) * self.p.inBatchNumel;
        if(count > std::numeric_limits<size_t>::max() / (fp16 ? sizeof(__half) : sizeof(float)))
            throw std::invalid_argument("PyTorch contiguous 转换缓冲区大小溢出");
        cuda_lab::check_cuda(cudaMalloc(&self.converted_input, count * (fp16 ? sizeof(__half) : sizeof(float))),
                             "PyTorch contiguous buffer");
    }
}

PytorchForward::~PytorchForward() = default;

void PytorchForward::bind(const void *input, const void *weights, const void *bias, void *output)
{
    auto &self = *impl_;

    if(!input || !weights || !output || (self.p.use_bias && !bias))
        throw std::invalid_argument("PytorchForward 要求非空 input/weight/output；use_bias=true 时 bias 必须非空");

    if(!self.p.use_bias)
        bias = nullptr;

#ifdef DWCONV_PYTORCH_HAS_CUDNN
    if(self.cudnn)
        self.cudnn->bind(self.converted_input ? self.converted_input : input, weights, bias, output);
#endif

    self.input = input;
    self.weights = weights;
    self.bias = bias;
    self.output = output;
}

void PytorchForward::autotune()
{
    auto &self = *impl_;
    if(!self.input)
        throw std::logic_error("PytorchForward 必须先 bind 再 autotune");

#ifdef DWCONV_PYTORCH_HAS_CUDNN
    if(self.cudnn && self.options.benchmark)
    {
        if(self.converted_input)
        {
            if(self.fp16)
                self.convert<__half>();
            else
                self.convert<float>();
        }
        self.cudnn->autotune();
    }
#endif
}

void PytorchForward::launch()
{
    auto &self = *impl_;
    if(!self.input)
        throw std::logic_error("PytorchForward 必须先 bind 再 launch");

    if(self.converted_input)
    {
        if(self.fp16)
            self.convert<__half>();
        else
            self.convert<float>();
    }

#ifdef DWCONV_PYTORCH_HAS_CUDNN
    if(self.cudnn)
    {
        self.cudnn->launch();
        return;
    }
#endif

    const void *input = self.converted_input ? self.converted_input : self.input;
    if(self.fp16)
        launch_pytorch_forward_fp16(static_cast<const __half *>(input), static_cast<const __half *>(self.weights),
                                    static_cast<const __half *>(self.bias), static_cast<__half *>(self.output), self.p,
                                    self.n, self.stream);
    else
        launch_pytorch_forward_fp32(static_cast<const float *>(input), static_cast<const float *>(self.weights),
                                    static_cast<const float *>(self.bias), static_cast<float *>(self.output), self.p,
                                    self.n, self.stream);
}

PytorchForwardInfo PytorchForward::info() const
{
    return impl_->info;
}
