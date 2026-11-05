#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <NvInfer.h>
#include <cuda_fp16.h>
#include "dwconv_tensorrt.cuh"
#include "dwconv_tensorrt_plugin.cuh"

namespace
{

constexpr size_t kWorkspaceLimit = 256ULL << 20;
constexpr int kBuilderOptimizationLevel = 5;
constexpr int kMaxAuxStreams = 0;

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *message) noexcept override
    {
        if(severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << message << "\n";
    }
};

void require(bool ok, const char *operation)
{
    if(!ok)
        throw std::runtime_error(std::string("TensorRT 失败：") + operation);
}

} // namespace

struct TensorrtForward::Impl
{
    Logger logger;

    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;
    TensorrtMode mode;
    bool bound = false;

    Impl(const Conv2DParam &p, int n, bool fp16, const void *weights, const void *bias, cudaStream_t s, TensorrtMode m)
        : stream(s), mode(m)
    {
        using namespace nvinfer1;
        require(n > 0 && p.in_ch > 0 && p.out_ch == p.in_ch && p.in_h > 0 && p.in_w > 0 && p.Kh > 0 && p.Kw > 0 &&
                    p.Sh > 0 && p.Sw > 0 && p.Ph >= 0 && p.Pw >= 0 && weights,
                "无效 DWConv 参数");
        require(mode == TensorrtMode::Native || mode == TensorrtMode::BaselinePlugin || mode == TensorrtMode::OptPlugin,
                "无效 TensorrtMode");
        require(!p.use_bias || bias, "use_bias=true 时 host_bias 不能为空");
        if(!p.use_bias)
            bias = nullptr;

        std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
        require(bool(builder), "createInferBuilder");

        std::unique_ptr<INetworkDefinition> network(
            builder->createNetworkV2(1U << uint32_t(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
        require(network && config, "createNetwork / createBuilderConfig");

        config->setFlag(BuilderFlag::kTF32);
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, kWorkspaceLimit);
        config->setBuilderOptimizationLevel(kBuilderOptimizationLevel);
        config->setMaxAuxStreams(kMaxAuxStreams);
        config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
        config->setProfileStream(stream);

        const DataType type = fp16 ? DataType::kHALF : DataType::kFLOAT;
        auto *input = network->addInput("input", type, Dims4{n, p.in_ch, p.in_h, p.in_w});
        require(input, "addInput");
        input->setAllowedFormats(1U << uint32_t(TensorFormat::kLINEAR));

        std::vector<__half2> packed_weights;
        std::vector<float> zero_bias_fp32;
        std::vector<__half> zero_bias_fp16;
        std::unique_ptr<IPluginV3> plugin;
        ITensor *output = nullptr;

        if(mode != TensorrtMode::Native)
        {
            require(p.in_ch % (fp16 ? 8 : 4) == 0 && p.KhKw > 0 && p.KhKw <= 128 && int64_t(p.Kh) * p.Kw == p.KhKw &&
                        n <= 65535 && p.in_ch / (fp16 ? 8 : 4) <= 65535,
                    "DWConv plugin 要求 FP32 C%4=0 / FP16 C%8=0 且 KhKw<=128");

            plugin = mode == TensorrtMode::OptPlugin ? make_dwconv_opt_plugin(p, n) : make_dwconv_baseline_plugin(p, n);

            if(fp16)
            {
                const auto *half_weights = static_cast<const __half *>(weights);
                packed_weights.resize(p.in_ch * p.KhKw / 2);
                for(int pair = 0; pair < p.in_ch / 2; ++pair)
                    for(int k = 0; k < p.KhKw; ++k)
                        packed_weights[pair * p.KhKw + k] = __halves2half2(half_weights[2 * pair * p.KhKw + k],
                                                                           half_weights[(2 * pair + 1) * p.KhKw + k]);
                weights = packed_weights.data();
            }

            if(!bias)
            {
                if(fp16)
                {
                    zero_bias_fp16.assign(p.in_ch, __float2half(0));
                    bias = zero_bias_fp16.data();
                }
                else
                {
                    zero_bias_fp32.assign(p.in_ch, 0);
                    bias = zero_bias_fp32.data();
                }
            }

            auto *w = network->addConstant(Dims2{p.in_ch, p.KhKw}, Weights{type, weights, int64_t(p.in_ch) * p.KhKw});
            auto *b = network->addConstant(Dims2{1, p.in_ch}, Weights{type, bias, p.in_ch});
            require(w && b, "DWConv plugin addConstant");

            ITensor *inputs[] = {input, w->getOutput(0), b->getOutput(0)};
            auto *layer = network->addPluginV3(inputs, 3, nullptr, 0, *plugin);
            require(layer, "addPluginV3");

            layer->setName(mode == TensorrtMode::OptPlugin ? "dwconv_opt" : "dwconv_baseline");
            output = layer->getOutput(0);
        }
        else
        {
            const Weights w{type, weights, int64_t(p.in_ch) * p.Kh * p.Kw};
            const Weights b{type, bias, bias ? p.in_ch : 0};
            auto *conv = network->addConvolutionNd(*input, p.out_ch, DimsHW{p.Kh, p.Kw}, w, b);
            require(conv, "addConvolutionNd");

            conv->setName("dwconv");
            conv->setNbGroups(p.in_ch);
            conv->setStrideNd(DimsHW{p.Sh, p.Sw});
            conv->setPaddingNd(DimsHW{p.Ph, p.Pw});
            output = conv->getOutput(0);
        }

        output->setName("output");
        network->markOutput(*output);
        output->setAllowedFormats(1U << uint32_t(TensorFormat::kLINEAR));
        const auto dims = output->getDimensions();
        require(dims.nbDims == 4 && dims.d[0] == n && dims.d[1] == p.out_ch && dims.d[2] == p.out_h &&
                    dims.d[3] == p.out_w,
                "输出尺寸与 Conv2DParam 不一致");

        std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
        require(bool(plan), "buildSerializedNetwork");

        runtime.reset(createInferRuntime(logger));
        require(bool(runtime), "createInferRuntime");

        engine.reset(runtime->deserializeCudaEngine(plan->data(), plan->size()));
        require(bool(engine), "deserializeCudaEngine");

        context.reset(engine->createExecutionContext());
        require(bool(context), "createExecutionContext");

        // 保留 inspector 的详细构建信息，但不在测速时生成详细 NVTX 标记。
        require(context->setNvtxVerbosity(ProfilingVerbosity::kNONE), "setNvtxVerbosity");
    }
};

TensorrtForward::TensorrtForward(const Conv2DParam &p, int n, bool fp16, const void *host_weights,
                                 const void *host_bias, cudaStream_t stream, TensorrtMode mode)
    : impl_(std::make_unique<Impl>(p, n, fp16, host_weights, host_bias, stream, mode))
{
}

TensorrtForward::~TensorrtForward() = default;

void TensorrtForward::bind(const void *input, void *output)
{
    require(input && output, "bind 空指针");
    impl_->bound = false;
    require(impl_->context->setInputTensorAddress("input", input) &&
                impl_->context->setOutputTensorAddress("output", output),
            "绑定输入输出");
    impl_->bound = true;
}

void TensorrtForward::launch()
{
    require(impl_->bound, "launch 前必须 bind");
    require(impl_->context->enqueueV3(impl_->stream), "enqueueV3");
}

std::string TensorrtForward::info() const
{
    std::ostringstream out;
    out << "[TensorRT] version=" << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH
        << " layers=" << impl_->engine->getNbLayers() << " context_memory=" << impl_->engine->getDeviceMemorySizeV2()
        << " bytes, workspace_limit=" << (kWorkspaceLimit >> 20)
        << " MiB, optimization_level=" << kBuilderOptimizationLevel << ", max_aux_streams=" << kMaxAuxStreams
        << ", NCHW, TF32=allowed, "
        << (impl_->mode == TensorrtMode::OptPlugin        ? "opt plugin, FP32 累加"
            : impl_->mode == TensorrtMode::BaselinePlugin ? "baseline plugin, FP32 累加"
                                                          : "原生计算，累加精度由 tactic 决定")
        << "\n";

    std::unique_ptr<nvinfer1::IEngineInspector> inspector(impl_->engine->createEngineInspector());
    require(bool(inspector), "createEngineInspector");

    const char *details = inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
    require(details, "getEngineInformation");

    out << "[TensorRT 引擎] " << details;
    return out.str();
}
