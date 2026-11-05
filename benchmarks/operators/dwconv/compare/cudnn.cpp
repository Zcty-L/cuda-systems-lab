#include <dwconv_cudnn.cuh>
#include "common.h"

namespace dwconv_bench
{

namespace
{

class CudnnOperation final : public ForwardOperation
{
  public:
    explicit CudnnOperation(const ForwardCase &context)
    {
        CudnnForwardOptions config;
        config.mode = context.kernel == "cudnn-fused"      ? CudnnForwardMode::Fused
                      : context.kernel == "cudnn-separate" ? CudnnForwardMode::Separate
                                                           : CudnnForwardMode::Graph;
        config.layout = context.options.layout == "nhwc" ? CudnnLayout::NHWC : CudnnLayout::NCHW;

        if(context.kernel == "cudnn-heuristic" || context.kernel == "cudnn-fused")
            config.selection = CudnnPlanSelection::Heuristic;

        const bool autotune = config.selection == CudnnPlanSelection::Autotune;
        output_nhwc = config.layout == CudnnLayout::NHWC;
        fp16_tolerance = 2e-3;

        forward_ = std::make_unique<CudnnForward>(context.param, context.shape.n, context.fp16, config);
        const auto info = forward_->info();

        std::cout << (autotune ? "[cuDNN 初始方案 / 搜索分配] version=" : "[cuDNN] version=") << info.version
                  << " algo=" << info.algorithm << " engine=" << info.engine << " workspace=" << info.workspace_bytes
                  << " bytes, layout=" << context.options.layout << ", candidates=" << info.candidate_count
                  << ", plan_workspace=" << info.plan_workspace_bytes << ", FP32 累加"
                  << (context.fp16 ? "，允许 Tensor Core" : "，允许 TF32") << "，执行 convolution"
                  << (context.shape.bias ? (context.kernel == "cudnn-separate" ? " + add bias" : " fused bias") : "")
                  << "\n";

        forward_->bind(context.input, context.weights, context.bias, context.output);

        if(autotune)
            forward_->autotune();

        // 首次执行验证支持情况并完成库的懒初始化。
        launch();
        const auto selected = forward_->info();
        algorithm = selected.algorithm;
        engine = selected.engine;
        workspace_bytes = selected.workspace_bytes;
    }

    void launch() override
    {
        try
        {
            forward_->launch();
        }
        catch(const CudnnUnsupported &error)
        {
            throw Unsupported(error.what());
        }
    }

  private:
    std::unique_ptr<CudnnForward> forward_;
};

bool compatible(const ShapeCase &shape, const std::string &kernel, const std::string &, const Options &)
{
    return kernel != "cudnn-fused" || shape.bias;
}

std::unique_ptr<ForwardOperation> prepare(const ForwardCase &context)
{
    try
    {
        return std::make_unique<CudnnOperation>(context);
    }
    catch(const CudnnUnsupported &error)
    {
        throw Unsupported(error.what());
    }
}

} // namespace

Backend cudnn_backend()
{
    Backend backend;
    backend.name = "cudnn";
    backend.kernels = {"cudnn", "cudnn-fused", "cudnn-separate", "cudnn-heuristic"};
    backend.compatible = compatible;
    backend.prepare = prepare;
    return backend;
}

} // namespace dwconv_bench
