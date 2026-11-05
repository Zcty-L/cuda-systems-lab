#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "dwconv_fp32.cuh"
#include "dwconv_fp16.cuh"
#include "dwconv_opt.cuh"
#include "dwconv_tensorrt_plugin.cuh"

namespace
{

using namespace nvinfer1;

template <bool Optimized> constexpr const char *plugin_name()
{
    return Optimized ? "DWConvOpt" : "DWConvBaseline";
}

constexpr const char *kNamespace = "cuda_systems_lab";
constexpr int kMaxGridYZ = 65535;

// 与 dwconv_opt_fp32.cu / dwconv_opt_fp16.cu 的共享输入 tile、容量及 48 KiB 分派上限一致。
constexpr int kOptSharedTileSize = 16;
constexpr int kOptSharedChannels = 4;
constexpr int kOptSharedWeightElements = kOptSharedChannels * 128;
constexpr size_t kOptSharedLimit = 48 * 1024;

// N, C, H, W, Kh, Kw, Sh, Sw, Ph, Pw；按字段序列化，不依赖结构体内存布局。
using Shape = std::array<int32_t, 10>;

template <bool Optimized>
class DwconvPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
{
  public:
    explicit DwconvPlugin(const Shape &shape) : shape_(shape)
    {
        const auto &s = shape_;
        for(int i = 0; i < 8; ++i)
            if(s[i] <= 0)
                throw std::invalid_argument("DWConv plugin 的尺寸和 stride 必须为正");

        if(s[8] < 0 || s[9] < 0 || s[1] % 4 || int64_t(s[4]) * s[5] > 128)
            throw std::invalid_argument("DWConv plugin 要求 C % 4 == 0、KhKw <= 128、padding >= 0");
        if(s[0] > kMaxGridYZ)
            throw std::invalid_argument("DWConv plugin 的 batch 超出 CUDA grid.z 上限");

        auto checked = [](int64_t value)
        {
            if(value <= 0 || value > std::numeric_limits<int>::max())
                throw std::invalid_argument("DWConv plugin 尺寸超出 int 范围");
            return int(value);
        };

        p_.in_ch = p_.out_ch = s[1];
        p_.in_h = s[2];
        p_.in_w = s[3];
        p_.Kh = s[4];
        p_.Kw = s[5];
        p_.KhKw = s[4] * s[5];
        p_.Sh = s[6];
        p_.Sw = s[7];
        p_.Ph = s[8];
        p_.Pw = s[9];
        const int64_t h = int64_t(s[2]) + 2LL * s[8] - s[4];
        const int64_t w = int64_t(s[3]) + 2LL * s[9] - s[5];

        if(h < 0 || w < 0)
            throw std::invalid_argument("DWConv plugin 的卷积核超出输入范围");

        p_.out_h = checked(h / s[6] + 1);
        p_.out_w = checked(w / s[7] + 1);
        p_.inHW = checked(int64_t(s[2]) * s[3]);
        p_.outHW = checked(int64_t(p_.out_h) * p_.out_w);
        p_.inBatchNumel = checked(int64_t(s[1]) * p_.inHW);
        p_.outBatchNumel = checked(int64_t(s[1]) * p_.outHW);
        checked(int64_t(s[0]) * p_.inBatchNumel);
        checked(int64_t(s[0]) * p_.outBatchNumel);

        p_.inChKhKw = p_.kernelWeightsCount = checked(int64_t(s[1]) * p_.KhKw);
        p_.biasWeightsCount = s[1];

        if constexpr(Optimized)
        {
            // 原 opt helper 先用 int 算 tile 尺寸，再用 size_t 算字节数；提前排除其溢出范围。
            // FP32 的通道组和 FP16 的通道对组均为 4，每个元素均占 4 字节。
            const int64_t tile_h = int64_t(kOptSharedTileSize - 1) * p_.Sh + p_.Kh;
            const int64_t tile_stride = int64_t(kOptSharedTileSize - 1) * p_.Sw + p_.Kw + 1;

            constexpr size_t max_spatial_elements =
                (std::numeric_limits<size_t>::max() / sizeof(float) - kOptSharedWeightElements) / kOptSharedChannels;
            constexpr size_t fitting_spatial_elements =
                (kOptSharedLimit / sizeof(float) - kOptSharedWeightElements) / kOptSharedChannels;

            shared_input_safe_ = tile_h <= std::numeric_limits<int>::max() &&
                                 tile_stride <= std::numeric_limits<int>::max() &&
                                 uint64_t(tile_h) <= max_spatial_elements / uint64_t(tile_stride);
            shared_input_fits_ =
                shared_input_safe_ && uint64_t(tile_h) <= fitting_spatial_elements / uint64_t(tile_stride);
        }

        // Plugin 固定接收 bias 输入；无 bias 时由构建端提供零常量。
        p_.use_bias = true;
        field_ = PluginField{"shape", shape_.data(), PluginFieldType::kINT32, 10};
        fields_ = PluginFieldCollection{1, &field_};
    }

    IPluginCapability *getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        switch(type)
        {
        case PluginCapabilityType::kCORE:
            return static_cast<IPluginV3OneCore *>(this);
        case PluginCapabilityType::kBUILD:
            return static_cast<IPluginV3OneBuild *>(this);
        case PluginCapabilityType::kRUNTIME:
            return static_cast<IPluginV3OneRuntime *>(this);
        }
        return nullptr;
    }

    IPluginV3 *clone() noexcept override
    {
        try
        {
            return new DwconvPlugin(shape_);
        }
        catch(...)
        {
            return nullptr;
        }
    }

    const char *getPluginName() const noexcept override
    {
        return plugin_name<Optimized>();
    }

    const char *getPluginVersion() const noexcept override
    {
        return "1";
    }

    const char *getPluginNamespace() const noexcept override
    {
        return kNamespace;
    }

    int32_t getNbOutputs() const noexcept override
    {
        return 1;
    }

    int32_t getOutputDataTypes(DataType *out, int32_t no, const DataType *in, int32_t ni) const noexcept override
    {
        if(ni != 3 || no != 1)
            return -1;
        out[0] = in[0];
        return 0;
    }

    int32_t getOutputShapes(const DimsExprs *, int32_t ni, const DimsExprs *, int32_t ns, DimsExprs *out, int32_t no,
                            IExprBuilder &builder) noexcept override
    {
        if(ni != 3 || no != 1 || ns != 0)
            return -1;
        out[0].nbDims = 4;
        const int dims[] = {shape_[0], p_.out_ch, p_.out_h, p_.out_w};
        for(int i = 0; i < 4; ++i)
            out[0].d[i] = builder.constant(dims[i]);
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, const DynamicPluginTensorDesc *io, int32_t ni,
                                   int32_t no) noexcept override
    {
        if(ni != 3 || no != 1 || pos < 0 || pos >= 4)
            return false;
        const auto &d = io[pos].desc;
        return d.format == TensorFormat::kLINEAR && supports_type(d.type) && (pos == 0 || d.type == io[0].desc.type);
    }

    int32_t configurePlugin(const DynamicPluginTensorDesc *in, int32_t ni, const DynamicPluginTensorDesc *out,
                            int32_t no) noexcept override
    {
        if(ni != 3 || no != 1)
            return -1;
        PluginTensorDesc inputs[] = {in[0].desc, in[1].desc, in[2].desc};
        return onShapeChange(inputs, ni, &out[0].desc, no);
    }

    int32_t onShapeChange(const PluginTensorDesc *in, int32_t ni, const PluginTensorDesc *out,
                          int32_t no) noexcept override
    {
        if(ni != 3 || no != 1)
            return -1;
        const Dims expected[] = {Dims4{shape_[0], p_.in_ch, p_.in_h, p_.in_w}, Dims2{p_.in_ch, p_.KhKw},
                                 Dims2{1, p_.in_ch}, Dims4{shape_[0], p_.out_ch, p_.out_h, p_.out_w}};
        for(int i = 0; i < 4; ++i)
        {
            const auto &d = i < 3 ? in[i] : out[0];
            if(d.format != TensorFormat::kLINEAR || d.type != in[0].type || d.dims.nbDims != expected[i].nbDims)
                return -1;
            for(int j = 0; j < d.dims.nbDims; ++j)
                if(d.dims.d[j] != expected[i].d[j])
                    return -1;
        }
        return supports_type(in[0].type) ? 0 : -1;
    }

    int32_t enqueue(const PluginTensorDesc *in, const PluginTensorDesc *, const void *const *inputs,
                    void *const *outputs, void *, cudaStream_t stream) noexcept override
    {
        try
        {
            // 复用原始调度；不分配、不打包、不同步，严格使用 TRT 提供的 stream。
            if(in[0].type == DataType::kHALF)
            {
                const auto *x = static_cast<const __half *>(inputs[0]);
                const auto *w = static_cast<const __half2 *>(inputs[1]);
                const auto *bias = static_cast<const __half *>(inputs[2]);
                auto *y = static_cast<__half *>(outputs[0]);
                if constexpr(Optimized)
                    launch_dwconv_opt_fp16(x, w, bias, y, p_, shape_[0], stream);
                else
                    launch_dwconv_fp16(x, w, bias, y, p_, shape_[0], stream);
            }
            else
            {
                auto *x = static_cast<float *>(const_cast<void *>(inputs[0]));
                auto *w = static_cast<float *>(const_cast<void *>(inputs[1]));
                auto *bias = static_cast<float *>(const_cast<void *>(inputs[2]));
                auto *y = static_cast<float *>(outputs[0]);
                if constexpr(Optimized)
                    launch_dwconv_opt_fp32(x, w, bias, y, p_, shape_[0], stream);
                else
                    launch_dwconv_fp32(x, w, bias, y, p_, shape_[0], stream);
            }
            return cudaGetLastError() == cudaSuccess ? 0 : -1;
        }
        catch(...)
        {
            // 公共 launch 可能抛出参数异常，不能越过 TRT 的 noexcept 回调边界。
            return -1;
        }
    }

    IPluginV3 *attachToContext(IPluginResourceContext *) noexcept override
    {
        return clone();
    }

    const PluginFieldCollection *getFieldsToSerialize() noexcept override
    {
        return &fields_;
    }

  private:
    bool supports_type(DataType type) const noexcept
    {
        const bool square = p_.Kh == p_.Kw;
        if(type == DataType::kFLOAT)
        {
            if constexpr(Optimized)
                if(square && p_.Kh == 11 && !shared_input_safe_)
                    return false;
            // K7 single-channel 分支仅在 grouped_blocks<104 时使用，C 自动小于 grid.y 上限。
            return p_.in_ch / 4 <= kMaxGridYZ;
        }
        if(type != DataType::kHALF || p_.in_ch % 8 != 0)
            return false;
        int channel_group = 8;
        if constexpr(Optimized)
        {
            bool pair_kernel = square && (p_.Kh == 3 || p_.Kh == 5 || p_.Kh == 7);
            if(square && p_.Kh == 9)
            {
                const size_t pair_blocks = size_t((p_.outHW + 255) / 256) * (p_.in_ch / 2) * shape_[0];
                if(pair_blocks >= 140 && !shared_input_safe_)
                    return false;
                pair_kernel = pair_blocks < 140 || !shared_input_fits_;
            }
            if(square && p_.Kh == 11 && !shared_input_safe_)
                return false;
            if(pair_kernel)
                channel_group = 2;
        }
        return p_.in_ch / channel_group <= kMaxGridYZ;
    }

    Shape shape_;
    Conv2DParam p_{};
    bool shared_input_safe_ = true;
    bool shared_input_fits_ = false;
    PluginField field_{};
    PluginFieldCollection fields_{};
};

template <bool Optimized> class DwconvCreator : public IPluginCreatorV3One
{
  public:
    const char *getPluginName() const noexcept override
    {
        return plugin_name<Optimized>();
    }

    const char *getPluginVersion() const noexcept override
    {
        return "1";
    }

    const char *getPluginNamespace() const noexcept override
    {
        return kNamespace;
    }

    const PluginFieldCollection *getFieldNames() noexcept override
    {
        return &fields_;
    }

    IPluginV3 *createPlugin(const char *, const PluginFieldCollection *fc, TensorRTPhase) noexcept override
    {
        if(!fc || fc->nbFields != 1 || !fc->fields)
            return nullptr;
        const auto &field = fc->fields[0];
        if(!field.name || std::strcmp(field.name, "shape") || field.type != PluginFieldType::kINT32 ||
           field.length != 10 || !field.data)
            return nullptr;
        Shape shape;
        std::memcpy(shape.data(), field.data, sizeof(shape));
        try
        {
            return new DwconvPlugin<Optimized>(shape);
        }
        catch(...)
        {
            return nullptr;
        }
    }

  private:
    PluginField field_{"shape", nullptr, PluginFieldType::kINT32, 10};
    PluginFieldCollection fields_{1, &field_};
};

template <bool Optimized> std::unique_ptr<IPluginV3> make_plugin(const Conv2DParam &p, int n)
{
    static DwconvCreator<Optimized> creator;
    static const bool registered = getPluginRegistry()->registerCreator(creator, kNamespace);
    if(!registered)
        throw std::runtime_error("无法注册 DWConv plugin creator");
    return std::make_unique<DwconvPlugin<Optimized>>(
        Shape{n, p.in_ch, p.in_h, p.in_w, p.Kh, p.Kw, p.Sh, p.Sw, p.Ph, p.Pw});
}
} // namespace

std::unique_ptr<nvinfer1::IPluginV3> make_dwconv_baseline_plugin(const Conv2DParam &p, int n)
{
    return make_plugin<false>(p, n);
}

std::unique_ptr<nvinfer1::IPluginV3> make_dwconv_opt_plugin(const Conv2DParam &p, int n)
{
    return make_plugin<true>(p, n);
}
