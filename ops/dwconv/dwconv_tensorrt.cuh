#pragma once


#include <memory>
#include <string>
#include <cuda_runtime.h>
#include "common.h"

enum class TensorrtMode
{
    Native,
    BaselinePlugin,
    OptPlugin
};

// 固定 shape、连续 NCHW，groups=in_ch=out_ch，dilation=1
// host_weights 为 [C,Kh,Kw]，host_bias 为 [C]
// p.use_bias=false 时忽略 host_bias，可传 nullptr；true 时 host_bias 必须非空
// Native：FP32 允许 tactic 使用 TF32；FP16 允许 tactic 使用 FP16 累加
// BaselinePlugin / OptPlugin：FP32/FP16 均 FP32 累加，KhKw<=128，FP32 C%4=0 / FP16 C%8=0
// FP16 权重在构建期打包；无 bias 时使用零常量，执行时无额外打包或分配
// stream 由调用方持有；建议传非默认流，避免 TRT 对默认流引入额外同步
class TensorrtForward
{
  public:
    TensorrtForward(const Conv2DParam &p, int n, bool fp16, const void *host_weights, const void *host_bias,
                    cudaStream_t stream = nullptr, TensorrtMode mode = TensorrtMode::Native);

    ~TensorrtForward();

    TensorrtForward(const TensorrtForward &) = delete;

    TensorrtForward &operator=(const TensorrtForward &) = delete;

    void bind(const void *input, void *output);

    void launch();

    std::string info() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
