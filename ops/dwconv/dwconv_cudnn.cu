#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cudnn.h>
#include <cuda_lab/cuda_check.cuh>
#include "dwconv_cudnn.cuh"

namespace
{

bool unsupported(cudnnStatus_t status)
{
    switch(status)
    {
    case CUDNN_STATUS_NOT_SUPPORTED:
    case CUDNN_STATUS_ARCH_MISMATCH:
#if CUDNN_MAJOR >= 9
    case CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN:
    case CUDNN_STATUS_NOT_SUPPORTED_SHAPE:
    case CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE:
    case CUDNN_STATUS_NOT_SUPPORTED_LAYOUT:
    case CUDNN_STATUS_NOT_SUPPORTED_SHARED_MEMORY_INSUFFICIENT:
    case CUDNN_STATUS_NOT_SUPPORTED_PADDING:
#endif
        return true;
    default:
        return false;
    }
}

void check_cudnn(cudnnStatus_t status)
{
    if(unsupported(status))
        throw CudnnUnsupported(std::string("cuDNN: ") + cudnnGetErrorString(status));
    if(status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuDNN: ") + cudnnGetErrorString(status));
}

cudnnBackendHeurMode_t heuristic_mode(CudnnHeuristicMode mode)
{
    switch(mode)
    {
    case CudnnHeuristicMode::A:
        return CUDNN_HEUR_MODE_A;
    case CudnnHeuristicMode::B:
        return CUDNN_HEUR_MODE_B;
    case CudnnHeuristicMode::Fallback:
        return CUDNN_HEUR_MODE_FALLBACK;
    }
    throw std::invalid_argument("cuDNN heuristic 模式无效");
}

void validate_options(const Conv2DParam &p, int n, const CudnnForwardOptions &options)
{
    if(n <= 0 || p.in_ch <= 0 || p.in_ch != p.out_ch || p.in_h <= 0 || p.in_w <= 0 || p.Kh <= 0 || p.Kw <= 0 ||
       p.Sh <= 0 || p.Sw <= 0 || p.Ph < 0 || p.Pw < 0)
        throw std::invalid_argument("cuDNN DWConv 要求正尺寸、groups=in_ch=out_ch、正 stride 与非负 padding");
    const int64_t h = int64_t(p.in_h) + 2LL * p.Ph - p.Kh;
    const int64_t w = int64_t(p.in_w) + 2LL * p.Pw - p.Kw;
    if(h < 0 || w < 0 || p.out_h != h / p.Sh + 1 || p.out_w != w / p.Sw + 1)
        throw std::invalid_argument("cuDNN 输出尺寸与卷积参数不一致");
    if(options.mode != CudnnForwardMode::Separate && options.mode != CudnnForwardMode::Fused &&
       options.mode != CudnnForwardMode::Graph)
        throw std::invalid_argument("cuDNN 执行模式无效");
    if(options.layout != CudnnLayout::NCHW && options.layout != CudnnLayout::NHWC)
        throw std::invalid_argument("cuDNN 布局无效");
    if(options.selection != CudnnPlanSelection::Heuristic && options.selection != CudnnPlanSelection::Enumerate &&
       options.selection != CudnnPlanSelection::Autotune)
        throw std::invalid_argument("cuDNN 方案选择方式无效");
    if(options.mode == CudnnForwardMode::Fused && !p.use_bias)
        throw std::invalid_argument("cuDNN legacy Fused 要求 use_bias=true；无 bias 可选择 Graph 或 Separate");
    if(options.mode == CudnnForwardMode::Graph)
    {
        if(options.heuristics.empty())
            throw std::invalid_argument("cuDNN Graph 至少需要一种 heuristic");
        for(auto mode : options.heuristics)
            heuristic_mode(mode);
    }
    const auto &tuning = options.tuning;
    if(options.selection == CudnnPlanSelection::Autotune &&
       (tuning.warmup < 0 || tuning.iterations <= 0 || tuning.repetitions <= 0 || tuning.shortlist < 0))
        throw std::invalid_argument("cuDNN autotune 要求 warmup/shortlist>=0，iterations/repetitions>0");
}

CudnnForwardOptions compatibility_options(CudnnForwardMode mode, CudnnLayout layout, bool autotune)
{
    CudnnForwardOptions options;
    options.mode = mode;
    options.layout = layout;
    if(autotune)
    {
        options.selection = CudnnPlanSelection::Autotune;
        options.heuristics = {CudnnHeuristicMode::A, CudnnHeuristicMode::B, CudnnHeuristicMode::Fallback};
    }
    else
    {
        options.selection = CudnnPlanSelection::Heuristic;
        options.heuristics = {CudnnHeuristicMode::A, CudnnHeuristicMode::Fallback};
        options.workspace_limit_bytes = std::numeric_limits<size_t>::max();
    }
    return options;
}

struct PreparedCandidate
{
    CudnnForwardCandidate info;
    cudnnBackendDescriptor_t plan = nullptr;
};

// Backend Graph API: Conv -> bias ADD，virtual FP32 中间结果不分配用户显存。
// 属性定义：https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html
struct BackendDesc
{
    cudnnBackendDescriptor_t desc = nullptr;

    explicit BackendDesc(cudnnBackendDescriptorType_t type)
    {
        check_cudnn(cudnnBackendCreateDescriptor(type, &desc));
    }

    ~BackendDesc()
    {
        cudnnBackendDestroyDescriptor(desc);
    }

    BackendDesc(const BackendDesc &) = delete;
    BackendDesc &operator=(const BackendDesc &) = delete;
};

void attr(cudnnBackendDescriptor_t d, cudnnBackendAttributeName_t name, cudnnBackendAttributeType_t type, int64_t count,
          const void *value)
{
    check_cudnn(cudnnBackendSetAttribute(d, name, type, count, value));
}

struct GraphPlan
{
    bool nhwc = false;
    bool use_bias = true;
    std::vector<std::unique_ptr<BackendDesc>> descriptors;
    std::unique_ptr<BackendDesc> pack;
    std::array<void *, 4> pointers{};

    cudnnBackendDescriptor_t make(cudnnBackendDescriptorType_t type)
    {
        descriptors.push_back(std::make_unique<BackendDesc>(type));
        return descriptors.back()->desc;
    }

    cudnnBackendDescriptor_t tensor(int64_t id, cudnnDataType_t dtype, std::array<int64_t, 4> dims,
                                    bool virtual_tensor = false)
    {
        auto d = make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        const int64_t alignment = 16;
        std::array<int64_t, 4> strides =
            nhwc ? std::array<int64_t, 4>{dims[1] * dims[2] * dims[3], 1, dims[3] * dims[1], dims[1]}
                 : std::array<int64_t, 4>{dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
        attr(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &id);
        attr(d, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype);
        attr(d, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dims.data());
        attr(d, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides.data());
        attr(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment);
        attr(d, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &virtual_tensor);
        check_cudnn(cudnnBackendFinalize(d));
        return d;
    }

    void prepare(cudnnHandle_t handle, const Conv2DParam &p, int n, bool half, const CudnnForwardOptions &options,
                 std::vector<PreparedCandidate> &candidates)
    {
        nhwc = options.layout == CudnnLayout::NHWC;
        use_bias = p.use_bias;
        const auto dtype = half ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
        const auto compute = CUDNN_DATA_FLOAT;
        auto x = tensor(0, dtype, {n, p.in_ch, p.in_h, p.in_w});

        // Backend 通过 X.C / W.C 推导 groups，此处 W.C=1。
        auto w = tensor(1, dtype, {p.in_ch, 1, p.Kh, p.Kw});
        auto y = tensor(3, dtype, {n, p.in_ch, p.out_h, p.out_w});
        auto tmp = use_bias ? tensor(4, compute, {n, p.in_ch, p.out_h, p.out_w}, true) : y;
        auto conv = make(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        const auto mode = CUDNN_CROSS_CORRELATION;
        const int64_t ndims = 2, dilation[] = {1, 1}, stride[] = {p.Sh, p.Sw}, pad[] = {p.Ph, p.Pw};
        attr(conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &compute);
        attr(conv, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
        attr(conv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &ndims);
        attr(conv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, dilation);
        attr(conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, stride);
        attr(conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, pad);
        attr(conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, pad);
        check_cudnn(cudnnBackendFinalize(conv));

        auto convop = make(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
        const float one = 1, zero = 0;
        attr(convop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x);
        attr(convop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w);
        attr(convop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tmp);
        attr(convop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv);
        attr(convop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &one);
        attr(convop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &zero);
        check_cudnn(cudnnBackendFinalize(convop));

        std::vector<cudnnBackendDescriptor_t> ops{convop};
        if(use_bias)
        {
            auto b = tensor(2, dtype, {1, p.in_ch, 1, 1});
            auto pw = make(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
            const auto add = CUDNN_POINTWISE_ADD;
            attr(pw, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &add);
            attr(pw, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &compute);
            check_cudnn(cudnnBackendFinalize(pw));

            auto addop = make(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
            attr(addop, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &pw);
            attr(addop, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tmp);
            attr(addop, CUDNN_ATTR_OPERATION_POINTWISE_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &b);
            attr(addop, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y);
            attr(addop, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, CUDNN_TYPE_FLOAT, 1, &one);
            attr(addop, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2, CUDNN_TYPE_FLOAT, 1, &one);
            check_cudnn(cudnnBackendFinalize(addop));
            ops.push_back(addop);
        }

        auto graph = make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        attr(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
        attr(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, ops.size(), ops.data());
        check_cudnn(cudnnBackendFinalize(graph));

        // 保持布局与精度，不拆图；heuristic 顺序、候选收集和 workspace 预算由配置决定。
        const bool collect_all = options.selection != CudnnPlanSelection::Heuristic;
        for(auto mode : options.heuristics)
        {
            const auto heuristic = heuristic_mode(mode);
            auto heur = make(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
            attr(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph);
            attr(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &heuristic);
            auto status = cudnnBackendFinalize(heur);
            if(unsupported(status))
                continue;
            check_cudnn(status);
            int64_t count = 0;
            check_cudnn(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0,
                                                 &count, nullptr));
            std::vector<cudnnBackendDescriptor_t> configs;
            for(int64_t i = 0; i < count; ++i)
                configs.push_back(make(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR));
            if(count == 0)
                continue;
            check_cudnn(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                 count, &count, configs.data()));

            // 实际返回数可能小于查询容量，不访问未填充的描述符。
            configs.resize(count);
            for(auto config : configs)
            {
                auto engine = make(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
                int64_t found = 0;
                check_cudnn(cudnnBackendGetAttribute(config, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                     1, &found, &engine));
                cudnnBackendNumericalNote_t notes[CUDNN_NUMERICAL_NOTE_TYPE_COUNT]{};
                check_cudnn(cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                                     CUDNN_TYPE_NUMERICAL_NOTE, CUDNN_NUMERICAL_NOTE_TYPE_COUNT, &found,
                                                     notes));
                bool reject = false;
                bool deterministic = true;
                for(int64_t i = 0; i < found; ++i)
                {
                    // FP32 是否允许 Tensor Core / TF32 由配置决定；FP16 不受该开关影响。
                    // 保留 FP32 累加并排除额外输入降类型。
                    reject |= notes[i] == CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS ||
                              notes[i] == CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION ||
                              (!half && !options.allow_tf32 && notes[i] == CUDNN_NUMERICAL_NOTE_TENSOR_CORE);
                    deterministic &= notes[i] != CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC;
                }
                if(reject || (options.deterministic && !deterministic))
                    continue;
                auto candidate = make(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
                attr(candidate, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
                attr(candidate, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &config);
                status = cudnnBackendFinalize(candidate);
                if(unsupported(status))
                    continue;
                check_cudnn(status);
                int64_t bytes = 0, id = -1;
                check_cudnn(cudnnBackendGetAttribute(candidate, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                                     CUDNN_TYPE_INT64, 1, &found, &bytes));
                check_cudnn(
                    cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &found, &id));
                if(bytes < 0)
                    throw std::runtime_error("cuDNN 返回负 workspace 大小");
                // 同一 engine 的不同 knob 配置均保留，不按 engine ID 去重。
                if(uint64_t(bytes) > options.workspace_limit_bytes)
                    continue;
                candidates.push_back({{-1, id, size_t(bytes), deterministic}, candidate});
                if(!collect_all)
                    return;
            }
        }
        if(!candidates.empty())
            return;
        throw CudnnUnsupported("cuDNN Graph：当前布局、精度、确定性与 workspace 预算下无可用 plan");
    }

    void bind(const void *x, const void *w, const void *b, void *y, void *workspace)
    {
        std::array<void *, 4> next{const_cast<void *>(x), const_cast<void *>(w), const_cast<void *>(b), y};
        if(pack && pointers == next)
            return;
        auto next_pack = std::make_unique<BackendDesc>(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        const int64_t ids[] = {0, 1, 3, 2};
        const std::array<void *, 4> data{next[0], next[1], next[3], next[2]};
        const int64_t count = use_bias ? 4 : 3;
        attr(next_pack->desc, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, count, ids);
        attr(next_pack->desc, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, count, data.data());
        attr(next_pack->desc, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace);
        check_cudnn(cudnnBackendFinalize(next_pack->desc));
        pack = std::move(next_pack);
        pointers = next;
    }
};

} // namespace

// groups=C，权重为 [C,1,Kh,Kw]；资源创建及算法选择均在计时外。
struct CudnnForward::Impl
{
    CudnnForwardOptions options;
    bool use_bias = true;
    bool bound = false;
    std::array<void *, 4> pointers{};
    std::vector<PreparedCandidate> candidates;
    size_t selected_candidate = 0;
    std::unique_ptr<GraphPlan> graph;
    cudnnActivationDescriptor_t activation = nullptr;
    cudnnHandle_t handle = nullptr;
    cudnnTensorDescriptor_t x = nullptr, y = nullptr, bias = nullptr;
    cudnnFilterDescriptor_t weight = nullptr;
    cudnnConvolutionDescriptor_t conv = nullptr;
    void *workspace = nullptr;
    size_t workspace_bytes = 0;

    Impl() = default;
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;

    ~Impl()
    {
        graph.reset();
        if(activation)
            cudnnDestroyActivationDescriptor(activation);
        if(workspace)
            cudaFree(workspace);
        if(conv)
            cudnnDestroyConvolutionDescriptor(conv);
        if(weight)
            cudnnDestroyFilterDescriptor(weight);
        if(bias)
            cudnnDestroyTensorDescriptor(bias);
        if(y)
            cudnnDestroyTensorDescriptor(y);
        if(x)
            cudnnDestroyTensorDescriptor(x);
        if(handle)
            cudnnDestroy(handle);
    }

    void prepare(const Conv2DParam &p, int n, bool half, cudaStream_t stream)
    {
        use_bias = p.use_bias;
        check_cudnn(cudnnCreate(&handle));
        check_cudnn(cudnnSetStream(handle, stream));
        if(options.mode == CudnnForwardMode::Graph)
        {
            graph = std::make_unique<GraphPlan>();
            graph->prepare(handle, p, n, half, options, candidates);
        }
        else
            prepare_legacy(p, n, half);
        for(const auto &candidate : candidates)
            workspace_bytes = std::max(workspace_bytes, candidate.info.workspace_bytes);
        if(workspace_bytes)
            cuda_lab::check_cuda(cudaMalloc(&workspace, workspace_bytes), "cuDNN workspace");
        select_candidate(0);
    }

    void prepare_legacy(const Conv2DParam &p, int n, bool half)
    {
        check_cudnn(cudnnCreateTensorDescriptor(&x));
        check_cudnn(cudnnCreateTensorDescriptor(&y));
        check_cudnn(cudnnCreateTensorDescriptor(&bias));
        check_cudnn(cudnnCreateFilterDescriptor(&weight));
        check_cudnn(cudnnCreateConvolutionDescriptor(&conv));
        const auto dtype = half ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
        const auto format = options.layout == CudnnLayout::NHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
        check_cudnn(cudnnSetTensor4dDescriptor(x, format, dtype, n, p.in_ch, p.in_h, p.in_w));
        check_cudnn(cudnnSetTensor4dDescriptor(y, format, dtype, n, p.in_ch, p.out_h, p.out_w));
        check_cudnn(cudnnSetTensor4dDescriptor(bias, format, dtype, 1, p.in_ch, 1, 1));
        check_cudnn(cudnnSetFilter4dDescriptor(weight, dtype, format, p.in_ch, 1, p.Kh, p.Kw));
        check_cudnn(cudnnSetConvolution2dDescriptor(conv, p.Ph, p.Pw, p.Sh, p.Sw, 1, 1, CUDNN_CROSS_CORRELATION,
                                                    CUDNN_DATA_FLOAT));
        check_cudnn(cudnnSetConvolutionGroupCount(conv, p.in_ch));
        // FP32 DEFAULT_MATH 允许 TF32，FMA_MATH 禁用 TF32；FP16 保持 Tensor Core 与 FP32 累加。
        const auto math_type = half ? CUDNN_TENSOR_OP_MATH : (options.allow_tf32 ? CUDNN_DEFAULT_MATH : CUDNN_FMA_MATH);
        check_cudnn(cudnnSetConvolutionMathType(conv, math_type));

        if(options.mode == CudnnForwardMode::Fused)
        {
            // IDENTITY 融合仅支持此算法；前向 PRECOMP_GEMM 为确定性算法。
            // https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html#cudnnconvolutionbiasactivationforward
            const auto algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
            check_cudnn(cudnnCreateActivationDescriptor(&activation));
            check_cudnn(
                cudnnSetActivationDescriptor(activation, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0));
            size_t bytes = 0;
            check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(handle, x, weight, conv, y, algo, &bytes));
            if(bytes > options.workspace_limit_bytes)
                throw CudnnUnsupported("cuDNN legacy Fused 的 workspace 超出预算");
            candidates.push_back({{int(algo), -1, bytes, true, int(math_type)}});
            return;
        }

        int capacity = 0, count = 0;
        // 同一算法可能有多种数学模式，容量不能用 ALGO_COUNT 代替。
        check_cudnn(cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &capacity));
        std::vector<cudnnConvolutionFwdAlgoPerf_t> results(capacity);
        check_cudnn(
            cudnnGetConvolutionForwardAlgorithm_v7(handle, x, weight, conv, y, capacity, &count, results.data()));
        for(int i = 0; i < count; ++i)
        {
            const auto &result = results[i];
            if(result.status != CUDNN_STATUS_SUCCESS ||
               (!half && result.mathType == CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION) ||
               (!half && !options.allow_tf32 && result.mathType != CUDNN_FMA_MATH) ||
               (options.deterministic && result.determinism != CUDNN_DETERMINISTIC))
                continue;
            check_cudnn(cudnnSetConvolutionMathType(conv, result.mathType));
            size_t bytes = 0;
            const auto status =
                cudnnGetConvolutionForwardWorkspaceSize(handle, x, weight, conv, y, result.algo, &bytes);
            if(unsupported(status))
                continue;
            check_cudnn(status);
            if(bytes > options.workspace_limit_bytes)
                continue;
            candidates.push_back(
                {{int(result.algo), -1, bytes, result.determinism == CUDNN_DETERMINISTIC, int(result.mathType)}});
            if(options.selection == CudnnPlanSelection::Heuristic)
                break;
        }
        if(candidates.empty())
            throw CudnnUnsupported("cuDNN Separate: 当前布局、精度、确定性与 workspace 预算下无可用算法");
    }

    void select_candidate(size_t index)
    {
        if(index >= candidates.size())
            throw std::out_of_range("cuDNN 候选索引越界");
        if(!graph)
            check_cudnn(
                cudnnSetConvolutionMathType(conv, static_cast<cudnnMathType_t>(candidates[index].info.math_type)));
        selected_candidate = index;
    }

    void bind(const void *input, const void *weights, const void *b, void *output)
    {
        if(!input || !weights || !output || (use_bias && !b))
            throw std::invalid_argument("cuDNN 要求非空 input/weights/output；use_bias=true 时 bias 也须非空");
        if(!use_bias)
            b = nullptr;
        const std::array<void *, 4> next{const_cast<void *>(input), const_cast<void *>(weights), const_cast<void *>(b),
                                         output};
        if(graph)
        {
            for(auto pointer : next)
                if(pointer && reinterpret_cast<uintptr_t>(pointer) % 16 != 0)
                    throw std::invalid_argument("cuDNN Graph 数据指针必须满足描述符声明的 16 字节对齐");
            graph->bind(input, weights, b, output, workspace);
        }
        pointers = next;
        bound = true;
    }

    void launch()
    {
        if(!bound)
            throw std::invalid_argument("cuDNN launch 前须先 bind 数据指针");
        const float one = 1, zero = 0;
        const auto &candidate = candidates[selected_candidate];
        if(graph)
        {
            check_cudnn(cudnnBackendExecute(handle, candidate.plan, graph->pack->desc));
            return;
        }
        const auto algo = static_cast<cudnnConvolutionFwdAlgo_t>(candidate.info.algorithm);
        auto input = pointers[0], weights = pointers[1], b = pointers[2], output = pointers[3];
        if(options.mode == CudnnForwardMode::Fused)
        {
            check_cudnn(cudnnConvolutionBiasActivationForward(handle, &one, x, input, weight, weights, conv, algo,
                                                              workspace, workspace_bytes, &zero, y, output, bias, b,
                                                              activation, y, output));
            return;
        }
        check_cudnn(cudnnConvolutionForward(handle, &one, x, input, weight, weights, conv, algo, workspace,
                                            workspace_bytes, &zero, y, output));
        if(use_bias)
            check_cudnn(cudnnAddTensor(handle, &one, bias, b, &one, y, output));
    }
};

CudnnForward::CudnnForward(const Conv2DParam &p, int n, bool fp16, const CudnnForwardOptions &options,
                           cudaStream_t stream)
    : impl_(std::make_unique<Impl>())
{
    validate_options(p, n, options);
    impl_->options = options;
    impl_->prepare(p, n, fp16, stream);
}

CudnnForward::CudnnForward(const Conv2DParam &p, int n, bool fp16, cudaStream_t stream, CudnnForwardMode mode,
                           CudnnLayout layout, bool autotune)
    : CudnnForward(p, n, fp16, compatibility_options(mode, layout, autotune), stream)
{
}

CudnnForward::~CudnnForward() = default;

void CudnnForward::bind(const void *input, const void *weights, const void *bias, void *output)
{
    impl_->bind(input, weights, bias, output);
}

void CudnnForward::launch()
{
    impl_->launch();
}

CudnnForwardInfo CudnnForward::info() const
{
    const auto &candidate = impl_->candidates[impl_->selected_candidate].info;
    return {cudnnGetVersion(),         candidate.algorithm,      impl_->workspace_bytes,   candidate.engine,
            candidate.workspace_bytes, impl_->candidates.size(), impl_->selected_candidate};
}

std::vector<CudnnForwardCandidate> CudnnForward::candidates() const
{
    std::vector<CudnnForwardCandidate> result;
    result.reserve(impl_->candidates.size());
    for(const auto &candidate : impl_->candidates)
        result.push_back(candidate.info);
    return result;
}

void CudnnForward::select_candidate(size_t index)
{
    impl_->select_candidate(index);
}

void CudnnForward::autotune()
{
    if(impl_->options.selection != CudnnPlanSelection::Autotune || !impl_->bound)
        throw std::invalid_argument("autotune 要求 selection=Autotune 并先 bind");
    cudaStream_t stream;
    check_cudnn(cudnnGetStream(impl_->handle, &stream));

    struct Events
    {
        cudaEvent_t start = nullptr, stop = nullptr;

        ~Events()
        {
            if(start)
                cudaEventDestroy(start);
            if(stop)
                cudaEventDestroy(stop);
        }
    } events;

    cuda_lab::check_cuda(cudaEventCreate(&events.start), "autotune event");
    cuda_lab::check_cuda(cudaEventCreate(&events.stop), "autotune event");

    auto measure = [&](size_t index, int warmup, int iterations)
    {
        impl_->select_candidate(index);

        for(int j = 0; j < warmup; ++j)
            impl_->launch();

        cuda_lab::check_cuda(cudaEventRecord(events.start, stream), "autotune start");
        for(int j = 0; j < iterations; ++j)
            impl_->launch();
        cuda_lab::check_cuda(cudaEventRecord(events.stop, stream), "autotune stop");
        cuda_lab::check_cuda(cudaEventSynchronize(events.stop), "autotune sync");

        float ms;
        cuda_lab::check_cuda(cudaEventElapsedTime(&ms, events.start, events.stop), "autotune time");
        return ms * 1000 / iterations;
    };

    const size_t previous = impl_->selected_candidate;
    try
    {
        const auto &tuning = impl_->options.tuning;
        std::vector<std::pair<float, size_t>> shortlist;
        for(size_t i = 0; i < impl_->candidates.size(); ++i)
        {
            if(tuning.shortlist == 0)
            {
                shortlist.emplace_back(0, i);
                continue;
            }
            const auto &c = impl_->candidates[i].info;
            try
            {
                const float us = measure(i, 10, 20);
                if(std::isfinite(us))
                    shortlist.emplace_back(us, i);
                std::cout << "[autotune] screening candidate=" << i << " algo=" << c.algorithm << " engine=" << c.engine
                          << " us=" << us << "\n";
            }
            catch(const CudnnUnsupported &e)
            {
                std::cout << "[autotune] screening candidate=" << i << " skip=" << e.what() << "\n";
            }
        }
        if(tuning.shortlist > 0)
        {
            std::sort(shortlist.begin(), shortlist.end());
            shortlist.resize(std::min(size_t(tuning.shortlist), shortlist.size()));
        }
        float best = std::numeric_limits<float>::infinity();
        size_t selected = previous;

        for(const auto &entry : shortlist)
        {
            const size_t i = entry.second;
            const auto &c = impl_->candidates[i].info;
            try
            {
                std::vector<float> times(tuning.repetitions);

                for(auto &time : times)
                    time = measure(i, tuning.warmup, tuning.iterations);

                std::sort(times.begin(), times.end());
                const float median = (times[(times.size() - 1) / 2] + times[times.size() / 2]) * 0.5f;

                std::cout << "[autotune] candidate=" << i << " algo=" << c.algorithm << " engine=" << c.engine
                          << " workspace=" << c.workspace_bytes << " us=" << median << "\n";

                if(median < best)
                {
                    best = median;
                    selected = i;
                }
            }
            catch(const CudnnUnsupported &e)
            {
                std::cout << "[autotune] candidate=" << i << " skip=" << e.what() << "\n";
            }
        }

        if(!std::isfinite(best))
            throw CudnnUnsupported("autotune 无可执行方案");
        impl_->select_candidate(selected);

        const auto &c = impl_->candidates[selected].info;
        std::cout << "[autotune] selected_engine=" << c.engine << " candidates=" << impl_->candidates.size()
                  << " refined=" << shortlist.size() << " median_us=" << best << " selected_candidate=" << selected
                  << " algo=" << c.algorithm << "\n";
    }
    catch(...)
    {
        impl_->select_candidate(previous);
        throw;
    }
}
