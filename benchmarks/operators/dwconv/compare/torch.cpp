#include <dwconv_pytorch.cuh>
#include "common.h"

namespace dwconv_bench
{

namespace
{

class TorchForward final : public ForwardOperation
{
  public:
    explicit TorchForward(const ForwardCase &context) : context_(context)
    {
        if(context.kernel != "pytorch")
            return;

        PytorchForwardOptions config;
        config.input_layout = context.options.layout == "nhwc" ? PytorchLayout::NHWC : PytorchLayout::NCHW;
        forward_ = std::make_unique<PytorchForward>(context.param, context.shape.n, context.fp16, config);

        const auto info = forward_->info();
        const bool pytorch_cudnn = info.backend == PytorchForwardBackend::Cudnn;
        output_nhwc = info.output_layout == PytorchLayout::NHWC;
        fp16_tolerance = pytorch_cudnn ? 2e-3 : 0;

        std::cout << "[PyTorch dispatch] backend=" << (pytorch_cudnn ? "Cudnn" : "CudaDepthwise2d")
                  << " input_layout=" << context.options.layout << " output_layout=" << (output_nhwc ? "nhwc" : "nchw")
                  << " cudnn_version=" << info.cudnn_version << " workspace=" << info.workspace_bytes
                  << " bytes, 当前固定使用 ATen native CUDA, cuDNN 分派暂时停用\n";

        forward_->bind(context.input, context.weights, context.shape.bias ? context.bias : nullptr, context.output);
        if(pytorch_cudnn)
            forward_->launch(); // 完成首次执行与懒初始化。
    }

    void launch() override
    {
        if(forward_)
            forward_->launch();
        else if(context_.fp16)
            launch_pytorch_forward_fp16(static_cast<const __half *>(context_.input),
                                        static_cast<const __half *>(context_.weights),
                                        context_.shape.bias ? static_cast<const __half *>(context_.bias) : nullptr,
                                        static_cast<__half *>(context_.output), context_.param, context_.shape.n);
        else
            launch_pytorch_forward_fp32(static_cast<const float *>(context_.input),
                                        static_cast<const float *>(context_.weights),
                                        context_.shape.bias ? static_cast<const float *>(context_.bias) : nullptr,
                                        static_cast<float *>(context_.output), context_.param, context_.shape.n);
    }

  private:
    ForwardCase context_;
    std::unique_ptr<PytorchForward> forward_;
};

bool compatible(const ShapeCase &, const std::string &kernel, const std::string &, const Options &options)
{
    return kernel == "pytorch" || options.layout == "nchw";
}

std::unique_ptr<ForwardOperation> prepare(const ForwardCase &context)
{
    return std::make_unique<TorchForward>(context);
}

} // namespace

Backend torch_backend()
{
    Backend backend;
    backend.name = "torch";
    backend.kernels = {"pytorch", "pytorch-native"};
    backend.compatible = compatible;
    backend.prepare = prepare;
    return backend;
}

} // namespace dwconv_bench
