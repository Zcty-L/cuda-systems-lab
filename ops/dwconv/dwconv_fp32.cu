#include <algorithm>
#include <stdexcept>
#include <cuda_lab/cuda_check.cuh>
#include <cuda_lab/ptx_utils.cuh>
#include "dwconv_fp32.cuh"

__device__ __forceinline__ void load_input(float *input_ptr, float (&input_frag)[4][4], const Conv2DParam &param,
                                           int posh_ori, int posw_ori, int k)
{
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        int tap = k + i;
        int curH = posh_ori + tap / param.Kw;
        int curW = posw_ori + tap % param.Kw;
        int in_offset = curH * param.in_w + curW;
        bool guard = curH >= 0 && curW >= 0 && curW < param.in_w && curH < param.in_h;

        if(guard)
        {
            input_frag[0][i] = input_ptr[in_offset + param.inHW * 0];
            input_frag[1][i] = input_ptr[in_offset + param.inHW * 1];
            input_frag[2][i] = input_ptr[in_offset + param.inHW * 2];
            input_frag[3][i] = input_ptr[in_offset + param.inHW * 3];
        }
        else
        {
            input_frag[0][i] = 0.0f;
            input_frag[1][i] = 0.0f;
            input_frag[2][i] = 0.0f;
            input_frag[3][i] = 0.0f;
        }
    }
}

template <int WeightStride>
__device__ __forceinline__ void load_weights_smem(float (&weight_frag)[16], uint32_t weights_lds_addr)
{
    ptx::lds128(weight_frag[0], weight_frag[1], weight_frag[2], weight_frag[3],
                weights_lds_addr + 0 * WeightStride * sizeof(float));
    ptx::lds128(weight_frag[4], weight_frag[5], weight_frag[6], weight_frag[7],
                weights_lds_addr + 1 * WeightStride * sizeof(float));
    ptx::lds128(weight_frag[8], weight_frag[9], weight_frag[10], weight_frag[11],
                weights_lds_addr + 2 * WeightStride * sizeof(float));
    ptx::lds128(weight_frag[12], weight_frag[13], weight_frag[14], weight_frag[15],
                weights_lds_addr + 3 * WeightStride * sizeof(float));
}

__device__ __forceinline__ void accumulate(float (&output_frag)[4], const float (&input_frag)[4][4],
                                           const float (&weight_frag)[16])
{
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        output_frag[0] += weight_frag[i + 0] * input_frag[0][i];
        output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
        output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
        output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
    }
}

namespace
{

template <int WeightStride, bool UseBias>
__global__ void dwconv_fp32_groups_kernel(float *inputs, float *weights, float *bias, float *outputs, Conv2DParam param)
{
    static_assert(WeightStride == 32 || WeightStride == 64 || WeightStride == 128, "WeightStride 仅支持 32、64、128");

    constexpr int kWeightsPerThread = WeightStride == 128 ? 2 : 1;
    constexpr int kThreadsPerChannel = WeightStride / kWeightsPerThread;

    __shared__ __align__(16) float smemweight[4 * WeightStride];

    int out_pos = blockIdx.x * 256 + threadIdx.x;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x * kWeightsPerThread);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const auto weight_offset = threadIdx.x % kThreadsPerChannel * kWeightsPerThread;
    const char *weight_ldg_ptr = reinterpret_cast<const char *>(
        weights + blockIdx.y * 4 * param.KhKw + threadIdx.x / kThreadsPerChannel * param.KhKw + weight_offset);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 4 * param.inHW;

    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        if constexpr(UseBias)
        {
            output_frag[i] = bias[blockIdx.y * 4 + i];
        }
        else
        {
            output_frag[i] = 0.0f;
        }
    }

    if constexpr(WeightStride == 128)
    {
        float weight_ldg_reg[2];
        ptx::ldg_nc_0(weight_ldg_reg[0], weight_ldg_ptr, weight_offset < param.KhKw);
        ptx::ldg_nc_0(weight_ldg_reg[1], weight_ldg_ptr + sizeof(float), weight_offset + 1 < param.KhKw);
        ptx::sts64(weight_ldg_reg[0], weight_ldg_reg[1], weights_sts_addr);
    }
    else if(WeightStride == 64 || threadIdx.x < 128)
    {
        float weight_ldg_reg;
        ptx::ldg_nc_0(weight_ldg_reg, weight_ldg_ptr, weight_offset < param.KhKw);
        ptx::sts32(weight_ldg_reg, weights_sts_addr);
    }
    __syncthreads();

    for(int k = 0; k < param.KhKw; k += 4)
    {
        load_input(input_ptr, input_frag, param, posh_ori, posw_ori, k);
        load_weights_smem<WeightStride>(weight_frag, weights_lds_addr);
        weights_lds_addr += 4 * sizeof(float);
        accumulate(output_frag, input_frag, weight_frag);
    }

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + out_pos;
    if(out_pos < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

template <int WeightStride>
void launch_dwconv_fp32_impl(float *inputs, float *weights, float *bias, float *outputs, const Conv2DParam &param,
                             dim3 grid, cudaStream_t stream)
{
    if(param.use_bias)
    {
        dwconv_fp32_groups_kernel<WeightStride, true><<<grid, 256, 0, stream>>>(inputs, weights, bias, outputs, param);
    }
    else
    {
        dwconv_fp32_groups_kernel<WeightStride, false><<<grid, 256, 0, stream>>>(inputs, weights, bias, outputs, param);
    }
}

// 权重一次加载, 交替预取下一输入片段
template <int WeightStride, bool UseBias>
__global__ void dwconv_fp32_groups_kernel_prefetch(float *inputs, float *weights, float *bias, float *outputs,
                                                   Conv2DParam param)
{
    static_assert(WeightStride == 32 || WeightStride == 64 || WeightStride == 128, "WeightStride 仅支持 32、64、128");

    constexpr int kWeightsPerThread = WeightStride == 128 ? 2 : 1;
    constexpr int kThreadsPerChannel = WeightStride / kWeightsPerThread;

    __shared__ __align__(16) float smemweight[4 * WeightStride];

    int out_pos = blockIdx.x * 256 + threadIdx.x;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x * kWeightsPerThread);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const auto weight_offset = threadIdx.x % kThreadsPerChannel * kWeightsPerThread;
    const char *weight_ldg_ptr = reinterpret_cast<const char *>(
        weights + blockIdx.y * 4 * param.KhKw + threadIdx.x / kThreadsPerChannel * param.KhKw + weight_offset);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 4 * param.inHW;

    float weight_frag[16];
    float input_a[4][4], input_b[4][4];
    float output_frag[4];

#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        if constexpr(UseBias)
        {
            output_frag[i] = bias[blockIdx.y * 4 + i];
        }
        else
        {
            output_frag[i] = 0.0f;
        }
    }

    if constexpr(WeightStride == 128)
    {
        float weight_ldg_reg[2];
        ptx::ldg_nc_0(weight_ldg_reg[0], weight_ldg_ptr, weight_offset < param.KhKw);
        ptx::ldg_nc_0(weight_ldg_reg[1], weight_ldg_ptr + sizeof(float), weight_offset + 1 < param.KhKw);
        ptx::sts64(weight_ldg_reg[0], weight_ldg_reg[1], weights_sts_addr);
    }
    else if(WeightStride == 64 || threadIdx.x < 128)
    {
        float weight_ldg_reg;
        ptx::ldg_nc_0(weight_ldg_reg, weight_ldg_ptr, weight_offset < param.KhKw);
        ptx::sts32(weight_ldg_reg, weights_sts_addr);
    }
    __syncthreads();

    load_input(input_ptr, input_a, param, posh_ori, posw_ori, 0);
    for(int k = 0; k < param.KhKw; k += 8)
    {
        if(k + 4 < param.KhKw)
        {
            load_input(input_ptr, input_b, param, posh_ori, posw_ori, k + 4);
        }

        load_weights_smem<WeightStride>(weight_frag, weights_lds_addr + k * sizeof(float));
        accumulate(output_frag, input_a, weight_frag);

        if(k + 4 < param.KhKw)
        {
            if(k + 8 < param.KhKw)
            {
                load_input(input_ptr, input_a, param, posh_ori, posw_ori, k + 8);
            }

            load_weights_smem<WeightStride>(weight_frag, weights_lds_addr + (k + 4) * sizeof(float));
            accumulate(output_frag, input_b, weight_frag);
        }
    }

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + out_pos;
    if(out_pos < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

template <int WeightStride>
void launch_dwconv_fp32_prefetch_impl(float *inputs, float *weights, float *bias, float *outputs,
                                      const Conv2DParam &param, dim3 grid, cudaStream_t stream)
{
    if(param.use_bias)
    {
        dwconv_fp32_groups_kernel_prefetch<WeightStride, true>
            <<<grid, 256, 0, stream>>>(inputs, weights, bias, outputs, param);
    }
    else
    {
        dwconv_fp32_groups_kernel_prefetch<WeightStride, false>
            <<<grid, 256, 0, stream>>>(inputs, weights, bias, outputs, param);
    }
}

// 基于 baseline 的独立 L2 权重实验：逐轮读取全局权重，不使用 shared 权重或 block 同步。
template <bool UseBias>
__global__ void dwconv_fp32_groups_kernel_weights_l2(float *inputs, float *weights, float *bias, float *outputs,
                                                     Conv2DParam param)
{
    int out_pos = blockIdx.x * 256 + threadIdx.x;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 4 * param.inHW;

    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        if constexpr(UseBias)
        {
            output_frag[i] = bias[blockIdx.y * 4 + i];
        }
        else
        {
            output_frag[i] = 0.0f;
        }
    }

    for(int k = 0; k < param.KhKw; k += 4)
    {
        load_input(input_ptr, input_frag, param, posh_ori, posw_ori, k);

#pragma unroll
        for(int ch = 0; ch < 4; ++ch)
        {
#pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                weight_frag[ch * 4 + i] =
                    k + i < param.KhKw ? __ldg(weights + (blockIdx.y * 4 + ch) * param.KhKw + k + i) : 0.0f;
            }
        }

        accumulate(output_frag, input_frag, weight_frag);
    }

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + out_pos;
    if(out_pos < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

int resolve_weight_stride(const Conv2DParam &param, const float *bias, int weight_stride)
{
    if(param.KhKw < 1 || param.KhKw > 128)
        throw std::invalid_argument("DWConv FP32 要求 1 <= KhKw <= 128");
    if(weight_stride == 0)
        weight_stride = param.KhKw <= 32 ? 32 : param.KhKw <= 64 ? 64 : 128;
    if((weight_stride != 32 && weight_stride != 64 && weight_stride != 128) || param.KhKw > weight_stride)
        throw std::invalid_argument("DWConv FP32 的 weight_stride 必须为 0、32、64、128，且容量不小于 KhKw");
    if(param.use_bias && bias == nullptr)
        throw std::invalid_argument("DWConv FP32 启用 bias 时指针不能为空");
    return weight_stride;
}

static cudaAccessPolicyWindow make_weights_l2_window(float *weights, const Conv2DParam &param, cudaStream_t stream)
{
    using cuda_lab::check_cuda;
    if(weights == nullptr)
        throw std::invalid_argument("DWConv FP32 的 L2 权重指针不能为空");

    int device = 0, max_window = 0, max_set_aside = 0;
    check_cuda(cudaGetDevice(&device), "读取 DWConv 当前设备");
    check_cuda(cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxAccessPolicyWindowSize, device), "读取 L2 窗口上限");
    check_cuda(cudaDeviceGetAttribute(&max_set_aside, cudaDevAttrMaxPersistingL2CacheSize, device), "读取 L2 保留上限");
    if(max_window <= 0 || max_set_aside <= 0)
        throw std::runtime_error("DWConv FP32 当前设备不支持 L2 优先保留窗口");

    const size_t weight_bytes = size_t(param.out_ch) * param.KhKw * sizeof(float);
    if(weight_bytes > size_t(max_window))
        throw std::invalid_argument("DWConv FP32 权重超过 L2 accessPolicyWindow 上限");

    // set-aside 属于当前 CUDA 上下文；只在容量不足时扩充，保留已有的更大配置。
    const size_t requested = std::min(size_t(max_set_aside), std::max(size_t(1024 * 1024), weight_bytes));
    size_t set_aside = 0;
    check_cuda(cudaDeviceGetLimit(&set_aside, cudaLimitPersistingL2CacheSize), "读取 L2 保留空间");
    if(set_aside < requested)
    {
        cudaStreamCaptureStatus capture_status;
        check_cuda(cudaStreamIsCapturing(stream, &capture_status), "读取 DWConv 捕获状态");
        if(capture_status == cudaStreamCaptureStatusNone)
        {
            check_cuda(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, requested), "设置 L2 保留空间");
            check_cuda(cudaDeviceGetLimit(&set_aside, cudaLimitPersistingL2CacheSize), "读取实际 L2 保留空间");
        }
        else if(set_aside == 0)
        {
            throw std::runtime_error("DWConv FP32 请在 CUDA Graph 捕获前先调用一次启用 L2 的 launch，以配置保留空间");
        }
    }
    if(set_aside == 0)
        throw std::runtime_error("DWConv FP32 未获得 L2 保留空间");

    cudaAccessPolicyWindow window{};
    window.base_ptr = weights;
    window.num_bytes = weight_bytes;
    window.hitRatio = std::min(1.0f, float(set_aside) / float(weight_bytes));
    window.hitProp = cudaAccessPropertyPersisting;
    window.missProp = cudaAccessPropertyNormal;
    return window;
}

template <bool UseBias>
void launch_dwconv_fp32_weights_l2_impl(float *inputs, float *weights, float *bias, float *outputs,
                                        const Conv2DParam &param, const cudaLaunchConfig_t &config)
{
    cuda_lab::check_cuda(cudaLaunchKernelEx(&config, dwconv_fp32_groups_kernel_weights_l2<UseBias>, inputs, weights,
                                            bias, outputs, param),
                         "启动 DWConv FP32 权重 L2 测试版本");
}

} // namespace

// TODO: block siwizle
void launch_dwconv_fp32(float *inputs, float *weights, float *bias, float *outputs, const Conv2DParam &param, int n,
                        cudaStream_t stream /* = nullptr */, int weight_stride /* = 0 */)
{
    weight_stride = resolve_weight_stride(param, bias, weight_stride);
    dim3 grid((param.outHW + 255) / 256, param.out_ch / 4, n);

    if(weight_stride == 32)
    {
        launch_dwconv_fp32_impl<32>(inputs, weights, bias, outputs, param, grid, stream);
    }
    else if(weight_stride == 64)
    {
        launch_dwconv_fp32_impl<64>(inputs, weights, bias, outputs, param, grid, stream);
    }
    else
    {
        launch_dwconv_fp32_impl<128>(inputs, weights, bias, outputs, param, grid, stream);
    }
}

// 测试 inputs-prefetch
void launch_dwconv_fp32_prefetch(float *inputs, float *weights, float *bias, float *outputs, const Conv2DParam &param,
                                 int n, cudaStream_t stream /* = nullptr */, int weight_stride /* = 0 */)
{
    weight_stride = resolve_weight_stride(param, bias, weight_stride);
    dim3 grid((param.outHW + 255) / 256, param.out_ch / 4, n);

    if(weight_stride == 32)
    {
        launch_dwconv_fp32_prefetch_impl<32>(inputs, weights, bias, outputs, param, grid, stream);
    }
    else if(weight_stride == 64)
    {
        launch_dwconv_fp32_prefetch_impl<64>(inputs, weights, bias, outputs, param, grid, stream);
    }
    else
    {
        launch_dwconv_fp32_prefetch_impl<128>(inputs, weights, bias, outputs, param, grid, stream);
    }
}

// TODO: block siwizle
// 尝试 weights 放到 L2
void launch_dwconv_fp32_weights_l2(float *inputs, float *weights, float *bias, float *outputs, const Conv2DParam &param,
                                   int n, cudaStream_t stream /* = nullptr */, bool use_l2_access_policy /* = false */)
{
    resolve_weight_stride(param, bias, 128);

    // 窗口绑定本次 launch，捕获时成为 kernel node 属性；不修改 stream 的持久属性。
    cudaLaunchAttribute policy{};
    policy.id = cudaLaunchAttributeAccessPolicyWindow;
    if(use_l2_access_policy)
        policy.val.accessPolicyWindow = make_weights_l2_window(weights, param, stream);

    cudaLaunchConfig_t config{};
    config.gridDim = dim3((param.outHW + 255) / 256, param.out_ch / 4, n);
    config.blockDim = dim3(256);
    config.stream = stream;
    config.attrs = use_l2_access_policy ? &policy : nullptr;
    config.numAttrs = use_l2_access_policy ? 1 : 0;

    if(param.use_bias)
        launch_dwconv_fp32_weights_l2_impl<true>(inputs, weights, bias, outputs, param, config);
    else
        launch_dwconv_fp32_weights_l2_impl<false>(inputs, weights, bias, outputs, param, config);
}
