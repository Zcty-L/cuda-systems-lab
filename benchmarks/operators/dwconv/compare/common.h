#pragma once

#include "cases.h"
#include <dwconv/common.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// DWConv 基准内部的后端准备与启动接口。
namespace dwconv_bench
{

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start);

void check(cudaError_t error);

struct Options
{
    std::string kernel = "baseline", dtype = "fp32", layout = "nchw";
    int case_index = -1, warmup = 20, iterations = 100, samples = 5;
    bool check_cpu = false, list = false, quick = false;
};

struct Stream
{
    cudaStream_t value{};

    Stream();
    ~Stream();
    Stream(const Stream &) = delete;
    Stream &operator=(const Stream &) = delete;
};

// 引用与数据指针由调用方管理，必须在 ForwardOperation 生命周期内保持有效。
struct ForwardCase
{
    const ShapeCase &shape;
    const Conv2DParam &param;
    const Options &options;
    const std::string &kernel;
    bool fp16;
    void *input, *weights, *bias, *output;
    const void *host_weights, *host_bias;
};

struct ForwardOperation
{
    cudaStream_t stream = nullptr;
    bool output_nhwc = false;
    double fp16_tolerance = 0;
    int algorithm = -1, engine = -1;
    // 未能从后端接口取得独立工作区或上下文内存时不填，基准 JSON 输出 null。
    std::optional<size_t> workspace_bytes = 0;

    virtual ~ForwardOperation() = default;
    virtual void launch() = 0;
};

class Unsupported : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

struct Backend
{
    std::string name;
    std::vector<std::string> kernels;
    bool fp16_packed_weights = false;
    bool host_weights = false;
    bool (*compatible)(const ShapeCase &, const std::string &, const std::string &, const Options &) = nullptr;
    std::unique_ptr<ForwardOperation> (*prepare)(const ForwardCase &) = nullptr;
};

Backend baseline_opt_backend();
Backend torch_backend();
Backend cudnn_backend();
Backend trt_backend();

int run_benchmarks(int argc, char **argv, const std::vector<Backend> &backends);

} // namespace dwconv_bench
