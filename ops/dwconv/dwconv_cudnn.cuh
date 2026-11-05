#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include "common.h"

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

enum class CudnnHeuristicMode
{
    A,       // 快速启发式，查询开销低，按预期性能排序候选配置。
    B,       // 更侧重排序准确性，查询开销较高，部分情况内部回退到 A。
    Fallback // 提供功能性备选配置，不保证性能最优。
};

enum class CudnnPlanSelection
{
    Heuristic, // 选择首个符合条件的方案。
    Enumerate, // 构建全部候选，供调用方手动选择，不运行卷积。
    Autotune   // 构建全部候选；bind 后显式调用 autotune() 实测选择。
};

struct CudnnAutotuneOptions
{
    int warmup = 20;
    int iterations = 100;
    int repetitions = 3;
    int shortlist = 3; // 0：精测全部候选；正数：初筛后仅精测指定数量。
};

struct CudnnForwardOptions
{
    CudnnForwardMode mode = CudnnForwardMode::Graph;
    CudnnLayout layout = CudnnLayout::NCHW;
    CudnnPlanSelection selection = CudnnPlanSelection::Autotune;

    // 仅用于 Graph，默认收集 A/B/Fallback 的全部可用候选。
    std::vector<CudnnHeuristicMode> heuristics{CudnnHeuristicMode::A, CudnnHeuristicMode::B,
                                               CudnnHeuristicMode::Fallback};
    size_t workspace_limit_bytes = size_t(256) * 1024 * 1024; // 256MB
    bool deterministic = false;
    bool allow_tf32 = true; // 仅影响 FP32；关闭时限制为 FMA，FP16 Tensor Core 路径保持不变。
    CudnnAutotuneOptions tuning;
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
    size_t plan_workspace_bytes; // 当前方案需求；workspace_bytes 是实际分配量。
    size_t candidate_count;
    size_t selected_candidate;
};

struct CudnnForwardCandidate
{
    int algorithm = -1;  // legacy 算法编号；Graph 为 -1。
    int64_t engine = -1; // Graph engine 编号；legacy 为 -1。
    size_t workspace_bytes = 0;
    bool deterministic = false;
    int math_type = -1; // legacy 的 cudnnMathType_t 编号；Graph 为 -1。
};

// NCHW 或 NHWC，groups=in_ch=out_ch，dilation=1。
// 权重对应 KCRS / KRSC；每组 C=1，两种布局均为连续 [C,Kh,Kw]，不需要重排。
// FP32 默认允许 TF32，options.allow_tf32=false 时关闭；FP16 使用 FP32 累加，不受此选项影响。
// p.use_bias=false 时忽略 bias。
// 默认 Autotune：bind 后显式调用 autotune() 选优，再调用 launch()。
// Graph 支持 Conv 或 Conv+bias（virtual FP32 中间张量）；legacy Fused 要求启用 bias。
class CudnnForward
{
  public:
    CudnnForward(const Conv2DParam &p, int n, bool fp16, const CudnnForwardOptions &options,
                 cudaStream_t stream = nullptr);

    // 保留位置参数接口；默认启用 autotune，false 时使用 heuristic 首个可用方案。
    CudnnForward(const Conv2DParam &p, int n, bool fp16, cudaStream_t stream = nullptr,
                 CudnnForwardMode mode = CudnnForwardMode::Graph, CudnnLayout layout = CudnnLayout::NCHW,
                 bool autotune = true);

    ~CudnnForward();

    CudnnForward(const CudnnForward &) = delete;

    CudnnForward &operator=(const CudnnForward &) = delete;

    // 准备数据绑定，不执行卷积；Graph 首次绑定或指针改变时创建 variant pack，放在计时外。
    void bind(const void *input, const void *weights, const void *bias, void *output);

    // 计算入口：执行已绑定的数据，不创建描述符或分配 workspace。
    void launch();

    // 返回当前对象的候选；相同 engine 的不同 knob 配置保留为独立条目。
    std::vector<CudnnForwardCandidate> candidates() const;

    // 仅在准备阶段切换已有候选；不分配资源，索引仅在本对象内有效。
    void select_candidate(size_t index);

    // selection=Autotune 且已 bind 时调用，会执行并覆盖输出；计时参数来自 options.tuning。
    // 默认先预热 10 次、计时 20 次初筛，再精测前三个候选并取多轮中位数。
    void autotune();

    CudnnForwardInfo info() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
