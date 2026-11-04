#pragma once
#include "dwconv.cuh"
#include <memory>
#include <stdexcept>

enum class CudnnForwardMode
{
    Separate,
    Fused,
    Graph
};

enum class CudnnLayout
{
    NCHW,
    NHWC
};

struct CudnnUnsupported : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

struct CudnnForwardInfo
{
    size_t version;
    int algorithm;
    size_t workspace_bytes;
    int64_t engine;
};

// NCHW 或 NHWC，groups=in_ch=out_ch，dilation=1。
// 权重对应 KCRS / KRSC；每组 C=1，两种布局均为连续 [C,Kh,Kw]，不需要重排。
// p 的尺寸/派生字段须一致；fp16 选择所有数据指针的类型（half 或 float）。
// 构造时选择算法并分配 workspace，析构释放资源；均应放在 kernel 计时外。
// FP32 禁用 TF32，FP16 使用 FP32 累加；Separate 的 bias 可为空，融合模式要求 bias。
// Graph 使用 virtual FP32 中间张量；不支持时抛出 CudnnUnsupported，不自动回退。
// Graph 的数据指针须 16 字节对齐（cudaMalloc 满足）；输入、权重、bias 与输出不重叠。
// launch 异步覆盖输出；调用方在析构前同步 stream，同一实例不可并发使用。
class CudnnForward
{
  public:
    CudnnForward(const Conv2DParam &p, int n, bool fp16, cudaStream_t stream = nullptr,
                 CudnnForwardMode mode = CudnnForwardMode::Graph, CudnnLayout layout = CudnnLayout::NCHW);

    ~CudnnForward();

    CudnnForward(const CudnnForward &) = delete;

    CudnnForward &operator=(const CudnnForward &) = delete;

    void launch(const void *input, const void *weights, const void *bias, void *output);

    // Graph 首次绑定或指针改变时创建 variant pack；调用方放在计时外。
    void bind(const void *input, const void *weights, const void *bias, void *output);

    CudnnForwardInfo info() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
