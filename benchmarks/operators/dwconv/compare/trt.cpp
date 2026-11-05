#include <dwconv_tensorrt.cuh>
#include "common.h"

namespace dwconv_bench
{

namespace
{

class TrtForward final : public ForwardOperation
{
  public:
    explicit TrtForward(const ForwardCase &context)
    {
        stream = stream_.value;
        workspace_bytes.reset(); // 当前封装未暴露 TensorRT context 的实际内存占用。
        fp16_tolerance = context.kernel == "tensorrt" ? 5e-3 : 0;
        const auto mode = context.kernel == "tensorrt-opt"        ? TensorrtMode::OptPlugin
                          : context.kernel == "tensorrt-baseline" ? TensorrtMode::BaselinePlugin
                                                                  : TensorrtMode::Native;
        forward_ = std::make_unique<TensorrtForward>(context.param, context.shape.n, context.fp16, context.host_weights,
                                                     context.host_bias, stream, mode);
        std::cout << forward_->info() << "\n";
        forward_->bind(context.input, context.output);
        forward_->launch(); // 首次执行完成库的懒初始化。
    }

    void launch() override
    {
        forward_->launch();
    }

  private:
    Stream stream_;
    std::unique_ptr<TensorrtForward> forward_;
};

bool compatible(const ShapeCase &shape, const std::string &kernel, const std::string &dtype, const Options &options)
{
    if(options.layout != "nchw")
        return false;
    return kernel == "tensorrt" || (shape.c % (dtype == "fp16" ? 8 : 4) == 0 && shape.kh * shape.kw <= 128);
}

std::unique_ptr<ForwardOperation> prepare(const ForwardCase &context)
{
    return std::make_unique<TrtForward>(context);
}

} // namespace

Backend trt_backend()
{
    Backend backend;
    backend.name = "trt";
    backend.kernels = {"tensorrt", "tensorrt-baseline", "tensorrt-opt"};
    backend.host_weights = true;
    backend.compatible = compatible;
    backend.prepare = prepare;
    return backend;
}

} // namespace dwconv_bench
