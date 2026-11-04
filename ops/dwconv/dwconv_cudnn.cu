#include "dwconv_cudnn.cuh"
#include <cuda_lab/cuda_check.cuh>
#include <cudnn.h>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>

namespace
{

void check_cudnn(cudnnStatus_t status)
{
    if(status == CUDNN_STATUS_NOT_SUPPORTED || status == CUDNN_STATUS_ARCH_MISMATCH)
        throw CudnnUnsupported(std::string("cuDNN: ") + cudnnGetErrorString(status));
    if(status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuDNN: ") + cudnnGetErrorString(status));
}

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
    std::vector<std::unique_ptr<BackendDesc>> descriptors;
    std::unique_ptr<BackendDesc> pack;
    cudnnBackendDescriptor_t plan = nullptr;
    int64_t engine_id = -1, workspace_bytes = 0;
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

    void prepare(cudnnHandle_t handle, const Conv2DParam &p, int n, bool half, CudnnLayout layout)
    {
        nhwc = layout == CudnnLayout::NHWC;
        const auto dtype = half ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
        const auto compute = CUDNN_DATA_FLOAT;
        auto x = tensor(0, dtype, {n, p.in_ch, p.in_h, p.in_w});

        // Backend 通过 X.C / W.C 推导 groups，此处 W.C=1。
        auto w = tensor(1, dtype, {p.in_ch, 1, p.Kh, p.Kw});
        auto b = tensor(2, dtype, {1, p.in_ch, 1, 1});
        auto y = tensor(3, dtype, {n, p.in_ch, p.out_h, p.out_w});
        auto tmp = tensor(4, compute, {n, p.in_ch, p.out_h, p.out_w}, true);
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

        auto graph = make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        const cudnnBackendDescriptor_t ops[] = {convop, addop};
        attr(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
        attr(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 2, ops);
        check_cudnn(cudnnBackendFinalize(graph));

        // 保持调用方布局，不拆图回退；A 无可用方案时尝试 FALLBACK 融合 engine。
        for(auto heuristic : {CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_FALLBACK})
        {
            auto heur = make(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
            attr(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph);
            attr(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &heuristic);
            auto status = cudnnBackendFinalize(heur);
            if(status == CUDNN_STATUS_NOT_SUPPORTED)
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
                for(int64_t i = 0; i < found; ++i)
                    reject |= notes[i] == CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS ||
                              notes[i] == CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION ||
                              (!half && notes[i] == CUDNN_NUMERICAL_NOTE_TENSOR_CORE);
                if(reject)
                    continue;
                auto candidate = make(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
                attr(candidate, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
                attr(candidate, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &config);
                status = cudnnBackendFinalize(candidate);
                if(status == CUDNN_STATUS_NOT_SUPPORTED || status == CUDNN_STATUS_ARCH_MISMATCH)
                    continue;
                check_cudnn(status);
                plan = candidate;
                check_cudnn(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64,
                                                     1, &found, &workspace_bytes));
                check_cudnn(cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1,
                                                     &found, &engine_id));
                return;
            }
        }
        throw CudnnUnsupported("Graph Conv+bias：当前布局 / groups=C / 指定精度无可用融合 plan");
    }

    void bind(const void *x, const void *w, const void *b, void *y, void *workspace)
    {
        std::array<void *, 4> next{const_cast<void *>(x), const_cast<void *>(w), const_cast<void *>(b), y};
        if(pack && pointers == next)
            return;
        pack = std::make_unique<BackendDesc>(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        const int64_t ids[] = {0, 1, 2, 3};
        attr(pack->desc, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 4, ids);
        attr(pack->desc, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 4, next.data());
        attr(pack->desc, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace);
        check_cudnn(cudnnBackendFinalize(pack->desc));
        pointers = next;
    }
};

} // namespace

// groups=C，权重为 [C,1,Kh,Kw]；资源创建及算法选择均在计时外。
struct CudnnForward::Impl
{
    CudnnForwardMode mode = CudnnForwardMode::Graph;
    CudnnLayout layout = CudnnLayout::NCHW;
    std::unique_ptr<GraphPlan> graph;
    cudnnActivationDescriptor_t activation = nullptr;
    cudnnHandle_t handle = nullptr;
    cudnnTensorDescriptor_t x = nullptr, y = nullptr, bias = nullptr;
    cudnnFilterDescriptor_t weight = nullptr;
    cudnnConvolutionDescriptor_t conv = nullptr;
    cudnnConvolutionFwdAlgo_t algo{};
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
        check_cudnn(cudnnCreate(&handle));
        check_cudnn(cudnnSetStream(handle, stream));
        if(mode == CudnnForwardMode::Graph)
        {
            graph = std::make_unique<GraphPlan>();
            graph->prepare(handle, p, n, half, layout);
            workspace_bytes = graph->workspace_bytes;
            if(workspace_bytes)
                cuda_lab::check_cuda(cudaMalloc(&workspace, workspace_bytes), "cuDNN workspace");
            return;
        }
        check_cudnn(cudnnCreateTensorDescriptor(&x));
        check_cudnn(cudnnCreateTensorDescriptor(&y));
        check_cudnn(cudnnCreateTensorDescriptor(&bias));
        check_cudnn(cudnnCreateFilterDescriptor(&weight));
        check_cudnn(cudnnCreateConvolutionDescriptor(&conv));
        const auto dtype = half ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
        const auto format = layout == CudnnLayout::NHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
        check_cudnn(cudnnSetTensor4dDescriptor(x, format, dtype, n, p.in_ch, p.in_h, p.in_w));
        check_cudnn(cudnnSetTensor4dDescriptor(y, format, dtype, n, p.in_ch, p.out_h, p.out_w));
        check_cudnn(cudnnSetTensor4dDescriptor(bias, format, dtype, 1, p.in_ch, 1, 1));
        check_cudnn(cudnnSetFilter4dDescriptor(weight, dtype, format, p.in_ch, 1, p.Kh, p.Kw));
        check_cudnn(cudnnSetConvolution2dDescriptor(conv, p.Ph, p.Pw, p.Sh, p.Sw, 1, 1, CUDNN_CROSS_CORRELATION,
                                                    CUDNN_DATA_FLOAT));
        check_cudnn(cudnnSetConvolutionGroupCount(conv, p.in_ch));
        // FP32 禁用 TF32；FP16 允许 Tensor Core，计算类型仍为 FP32。
        check_cudnn(cudnnSetConvolutionMathType(conv, half ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH));

        if(mode == CudnnForwardMode::Fused)
        {
            algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
            check_cudnn(cudnnCreateActivationDescriptor(&activation));
            check_cudnn(
                cudnnSetActivationDescriptor(activation, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0));
        }
        else
        {
            cudnnConvolutionFwdAlgoPerf_t results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT]{};
            int count = 0;
            check_cudnn(cudnnGetConvolutionForwardAlgorithm_v7(handle, x, weight, conv, y,
                                                               CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &count, results));
            bool found = false;
            for(int i = 0; i < count; ++i)
            {
                if(results[i].status != CUDNN_STATUS_SUCCESS || (!half && results[i].mathType != CUDNN_FMA_MATH))
                    continue;
                algo = results[i].algo;
                check_cudnn(cudnnSetConvolutionMathType(conv, results[i].mathType));
                found = true;
                break;
            }
            if(!found)
                throw std::runtime_error("cuDNN 无兼容的前向算法");
        }
        check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(handle, x, weight, conv, y, algo, &workspace_bytes));
        if(workspace_bytes)
            cuda_lab::check_cuda(cudaMalloc(&workspace, workspace_bytes), "cuDNN workspace");
    }

    void launch(const void *input, const void *weights, const void *b, void *output)
    {
        const float one = 1, zero = 0;
        if(mode != CudnnForwardMode::Separate && !b)
            throw std::invalid_argument("cuDNN 融合模式要求 bias 指针");
        if(graph)
        {
            graph->bind(input, weights, b, output, workspace);
            check_cudnn(cudnnBackendExecute(handle, graph->plan, graph->pack->desc));
            return;
        }
        if(mode == CudnnForwardMode::Fused)
        {
            check_cudnn(cudnnConvolutionBiasActivationForward(handle, &one, x, input, weight, weights, conv, algo,
                                                              workspace, workspace_bytes, &zero, y, output, bias, b,
                                                              activation, y, output));
            return;
        }
        check_cudnn(cudnnConvolutionForward(handle, &one, x, input, weight, weights, conv, algo, workspace,
                                            workspace_bytes, &zero, y, output));
        if(b)
            check_cudnn(cudnnAddTensor(handle, &one, bias, b, &one, y, output));
    }
};

CudnnForward::CudnnForward(const Conv2DParam &p, int n, bool fp16, cudaStream_t stream, CudnnForwardMode mode,
                           CudnnLayout layout)
    : impl_(std::make_unique<Impl>())
{
    impl_->mode = mode;
    impl_->layout = layout;
    impl_->prepare(p, n, fp16, stream);
}

CudnnForward::~CudnnForward() = default;
void CudnnForward::launch(const void *input, const void *weights, const void *bias, void *output)
{
    impl_->launch(input, weights, bias, output);
}

CudnnForwardInfo CudnnForward::info() const
{
    return {cudnnGetVersion(), impl_->graph ? -1 : int(impl_->algo), impl_->workspace_bytes,
            impl_->graph ? impl_->graph->engine_id : -1};
}

void CudnnForward::bind(const void *input, const void *weights, const void *bias, void *output)
{
    if(impl_->mode != CudnnForwardMode::Separate && !bias)
        throw std::invalid_argument("cuDNN 融合模式要求 bias 指针");
    if(impl_->graph)
        impl_->graph->bind(input, weights, bias, output, impl_->workspace);
}
