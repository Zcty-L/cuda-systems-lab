#include <dwconv_fp32.cuh>
#include <dwconv_fp16.cuh>
#include <dwconv_opt.cuh>
#include "common.h"

namespace dwconv_bench
{

namespace
{

class BaselineOptForward final : public ForwardOperation
{
  public:
    explicit BaselineOptForward(const ForwardCase &context)
        : kernel_(context.kernel), param_(context.param), n_(context.shape.n), fp16_(context.fp16),
          input_(context.input), weights_(context.weights), bias_(context.bias), output_(context.output)
    {
    }

    void launch() override
    {
        const int capacity = kernel_ == "baseline-k32"    ? 32
                             : kernel_ == "baseline-k64"  ? 64
                             : kernel_ == "baseline-k128" ? 128
                                                          : 0;
        if(fp16_)
        {
            const auto *x = static_cast<const __half *>(input_);
            const auto *w = static_cast<const __half2 *>(weights_);
            const auto *b = static_cast<const __half *>(bias_);
            auto *y = static_cast<__half *>(output_);
            if(kernel_ == "opt")
                launch_dwconv_opt_fp16(x, w, b, y, param_, n_);
            else
                launch_dwconv_fp16(x, w, b, y, param_, n_, nullptr, capacity);
        }
        else
        {
            auto *x = static_cast<float *>(input_);
            auto *w = static_cast<float *>(weights_);
            auto *b = static_cast<float *>(bias_);
            auto *y = static_cast<float *>(output_);
            if(kernel_ == "opt")
                launch_dwconv_opt_fp32(x, w, b, y, param_, n_);
            else if(kernel_ == "prefetch")
                launch_dwconv_fp32_prefetch(x, w, b, y, param_, n_, nullptr, 128);
            else if(kernel_ == "gmem" || kernel_ == "gmem-l2-entry")
                launch_dwconv_fp32_weights_l2(x, w, b, y, param_, n_, nullptr, kernel_ == "gmem-l2-entry");
            else
                launch_dwconv_fp32(x, w, b, y, param_, n_, nullptr, capacity);
        }
    }

  private:
    std::string kernel_;
    Conv2DParam param_;
    int n_;
    bool fp16_;
    void *input_, *weights_, *bias_, *output_;
};

bool compatible(const ShapeCase &s, const std::string &kernel, const std::string &dtype, const Options &options)
{
    if(options.layout != "nchw" || !s.bias || s.c % (dtype == "fp16" ? 8 : 4) != 0)
        return false;
    if(dtype == "fp16" && (kernel == "prefetch" || kernel == "gmem" || kernel == "gmem-l2-entry"))
        return false;
    const int capacity = kernel == "baseline-k32" ? 32 : kernel == "baseline-k64" ? 64 : 128;
    return s.kh * s.kw <= capacity;
}

std::unique_ptr<ForwardOperation> prepare(const ForwardCase &context)
{
    return std::make_unique<BaselineOptForward>(context);
}

} // namespace

Backend baseline_opt_backend()
{
    return {"baseline-opt",
            {"baseline", "baseline-k128", "baseline-k64", "baseline-k32", "prefetch", "opt", "gmem", "gmem-l2-entry"},
            true,
            false,
            compatible,
            prepare};
}

} // namespace dwconv_bench
