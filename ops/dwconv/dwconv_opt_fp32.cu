#include "dwconv_opt.cuh"
#include <cuda_lab/ptx_utils.cuh>

namespace
{
constexpr int kSharedTileH = 16;
constexpr int kSharedTileW = 16;
constexpr int kSharedBlockSize = kSharedTileH * kSharedTileW;
constexpr int kSharedChannelGroup = 4;
constexpr int kSharedKernelSize = 11;
constexpr int kSharedKernelElements = kSharedKernelSize * kSharedKernelSize;
constexpr int kSharedWeightStride = 128;
constexpr int kSharedWeightElements = kSharedChannelGroup * kSharedWeightStride;

__device__ __forceinline__ int swizzle_shared_input_x(int input_x)
{
    return input_x ^ ((input_x >> 3) & 1);
}

template <int KernelSize, bool ShuffleBias = false>
__global__ void conv2d_4x128x256_groups_kernel(float *inputs, float *weights, float *bias, float *outputs,
                                               Conv2DParam param)
{
    constexpr bool specialized_kernel = KernelSize > 0;
    constexpr int specialized_kernel_elements = KernelSize * KernelSize;
    int kernel_width = specialized_kernel ? KernelSize : param.Kw;
    int kernel_elements = specialized_kernel ? specialized_kernel_elements : param.KhKw;

    __shared__ __align__(2 * 1024) char smem[4 * 128 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem);

    int out_pos = blockIdx.x * 256 + threadIdx.x;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x * 2);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = reinterpret_cast<const char *>(
        weights + blockIdx.y * 4 * kernel_elements + threadIdx.x / 64 * kernel_elements + threadIdx.x % 64 * 2);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 4 * param.inHW;

    float weight_ldg_reg[2];
    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

    // 编译期开关仅用于 bias 加载对照；默认保留直接读取。
    if constexpr(ShuffleBias)
    {
        int lane = threadIdx.x & 31;
        float value = 0.0f;
        if(lane < 4)
            value = bias[blockIdx.y * 4 + lane];
#pragma unroll
        for(int i = 0; i < 4; ++i)
            output_frag[i] = __shfl_sync(0xffffffff, value, i);
    }
    else
    {
#pragma unroll
        for(int i = 0; i < 4; ++i)
            output_frag[i] = bias[blockIdx.y * 4 + i];
    }

    ptx::ldg_nc_0(weight_ldg_reg[0], weight_ldg_ptr, threadIdx.x % 64 * 2 < kernel_elements);
    ptx::ldg_nc_0(weight_ldg_reg[1], weight_ldg_ptr + sizeof(float), threadIdx.x % 64 * 2 + 1 < kernel_elements);

    ptx::sts64(weight_ldg_reg[0], weight_ldg_reg[1], weights_sts_addr);
    __syncthreads();

    for(int k = 0; k < kernel_elements; k += 4)
    {
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            int tap = k + i;
            int curH = posh_ori + tap / kernel_width;
            int curW = posw_ori + tap % kernel_width;
            int in_offset = curH * param.in_w + curW;
            bool guard = tap < kernel_elements && curH >= 0 && curW >= 0 && curW < param.in_w && curH < param.in_h;

            if constexpr(KernelSize == 3)
            {
                if(guard)
                {
                    ptx::ldg_nc_0(input_frag[0][i], input_ptr + in_offset, true);
                    ptx::ldg_nc_0(input_frag[1][i], input_ptr + in_offset + param.inHW, true);
                    ptx::ldg_nc_0(input_frag[2][i], input_ptr + in_offset + param.inHW * 2, true);
                    ptx::ldg_nc_0(input_frag[3][i], input_ptr + in_offset + param.inHW * 3, true);
                }
                else
                {
                    input_frag[0][i] = 0.0f;
                    input_frag[1][i] = 0.0f;
                    input_frag[2][i] = 0.0f;
                    input_frag[3][i] = 0.0f;
                }
            }
            else if constexpr(specialized_kernel)
            {
                ptx::ldg_nc_0(input_frag[0][i], input_ptr + in_offset, guard);
                ptx::ldg_nc_0(input_frag[1][i], input_ptr + in_offset + param.inHW, guard);
                ptx::ldg_nc_0(input_frag[2][i], input_ptr + in_offset + param.inHW * 2, guard);
                ptx::ldg_nc_0(input_frag[3][i], input_ptr + in_offset + param.inHW * 3, guard);
            }
            else
            {
                if(guard)
                {
                    input_frag[0][i] = input_ptr[in_offset];
                    input_frag[1][i] = input_ptr[in_offset + param.inHW];
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

        ptx::lds128(weight_frag[0], weight_frag[1], weight_frag[2], weight_frag[3], weights_lds_addr);
        ptx::lds128(weight_frag[4], weight_frag[5], weight_frag[6], weight_frag[7],
                    weights_lds_addr + 128 * sizeof(float));
        ptx::lds128(weight_frag[8], weight_frag[9], weight_frag[10], weight_frag[11],
                    weights_lds_addr + 256 * sizeof(float));
        ptx::lds128(weight_frag[12], weight_frag[13], weight_frag[14], weight_frag[15],
                    weights_lds_addr + 384 * sizeof(float));
        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
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

static void launch_custom(void *inputs, void *weights, void *bias, void *outputs, Conv2DParam param, int n,
                          cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((param.outHW + 255) / 256, param.out_ch / 4, n);
    conv2d_4x128x256_groups_kernel<0>
        <<<grid, block, 0, stream>>>(static_cast<float *>(inputs), static_cast<float *>(weights),
                                     static_cast<float *>(bias), static_cast<float *>(outputs), param);
}

template <int KernelSize>
static void launch_specialized(void *inputs, void *weights, void *bias, void *outputs, Conv2DParam param, int n,
                               cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((param.outHW + 255) / 256, param.out_ch / 4, n);
    conv2d_4x128x256_groups_kernel<KernelSize>
        <<<grid, block, 0, stream>>>(static_cast<float *>(inputs), static_cast<float *>(weights),
                                     static_cast<float *>(bias), static_cast<float *>(outputs), param);
}

__global__ void conv2d_1x128x256_groups_k7_kernel(const float *inputs, const float *weights, const float *bias,
                                                  float *outputs, Conv2DParam param)
{
    constexpr int kernel_size = 7;
    constexpr int kernel_elements = kernel_size * kernel_size;
    __shared__ float shared_weights[kernel_elements];

    if(threadIdx.x < kernel_elements)
    {
        shared_weights[threadIdx.x] = weights[blockIdx.y * kernel_elements + threadIdx.x];
    }
    __syncthreads();

    int out_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int posh_origin = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_origin = (out_pos % param.out_w) * param.Sw - param.Pw;
    const float *input_channel = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * param.inHW;
    float accumulator = bias[blockIdx.y];

    for(int tap_base = 0; tap_base < kernel_elements; tap_base += 4)
    {
        float input_values[4];
        float weight_values[4];
#pragma unroll
        for(int index = 0; index < 4; ++index)
        {
            int tap = tap_base + index;
            int kernel_h = tap / kernel_size;
            int kernel_w = tap % kernel_size;
            int input_h = posh_origin + kernel_h;
            int input_w = posw_origin + kernel_w;
            bool valid =
                tap < kernel_elements && input_h >= 0 && input_w >= 0 && input_h < param.in_h && input_w < param.in_w;
            int input_offset = input_h * param.in_w + input_w;
            ptx::ldg_nc_0(input_values[index], input_channel + input_offset, valid);
            weight_values[index] = tap < kernel_elements ? shared_weights[tap] : 0.0f;
        }
#pragma unroll
        for(int index = 0; index < 4; ++index)
        {
            accumulator += weight_values[index] * input_values[index];
        }
    }

    if(out_pos < param.outHW)
    {
        int output_offset = blockIdx.z * param.outBatchNumel + blockIdx.y * param.outHW + out_pos;
        outputs[output_offset] = accumulator;
    }
}

static void launch_k7_single_channel(void *inputs, void *weights, void *bias, void *outputs, Conv2DParam param, int n,
                                     cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((param.outHW + 255) / 256, param.out_ch, n);
    conv2d_1x128x256_groups_k7_kernel<<<grid, block, 0, stream>>>(
        static_cast<const float *>(inputs), static_cast<const float *>(weights), static_cast<const float *>(bias),
        static_cast<float *>(outputs), param);
}

__global__ void conv2d_4x16x16_shared_input_kernel(const float *inputs, const float *weights, const float *bias,
                                                   float *outputs, Conv2DParam param)
{
    extern __shared__ __align__(16) float shared[];
    float *shared_weights = shared;
    float *shared_inputs = shared_weights + kSharedWeightElements;

    int input_tile_h = (kSharedTileH - 1) * param.Sh + kSharedKernelSize;
    int input_tile_w = (kSharedTileW - 1) * param.Sw + kSharedKernelSize;
    int shared_input_stride = input_tile_w + 1;
    int input_tile_hw = input_tile_h * input_tile_w;
    int tiles_w = (param.out_w + kSharedTileW - 1) / kSharedTileW;
    int tile_h = blockIdx.x / tiles_w;
    int tile_w = blockIdx.x % tiles_w;
    int output_h_origin = tile_h * kSharedTileH;
    int output_w_origin = tile_w * kSharedTileW;
    int input_h_origin = output_h_origin * param.Sh - param.Ph;
    int input_w_origin = output_w_origin * param.Sw - param.Pw;

    for(int index = threadIdx.x; index < kSharedWeightElements; index += blockDim.x)
    {
        int channel = index / kSharedWeightStride;
        int tap = index % kSharedWeightStride;
        float value = 0.0f;
        if(tap < kSharedKernelElements)
        {
            value = weights[blockIdx.y * kSharedChannelGroup * kSharedKernelElements + channel * kSharedKernelElements +
                            tap];
        }
        shared_weights[tap * kSharedChannelGroup + channel] = value;
    }

    const float *input_group = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * kSharedChannelGroup * param.inHW;
    for(int spatial = threadIdx.x; spatial < input_tile_hw; spatial += blockDim.x)
    {
        int tile_input_h = spatial / input_tile_w;
        int tile_input_w = spatial % input_tile_w;
        int input_h = input_h_origin + tile_input_h;
        int input_w = input_w_origin + tile_input_w;
        bool valid = input_h >= 0 && input_w >= 0 && input_h < param.in_h && input_w < param.in_w;
        int input_offset = input_h * param.in_w + input_w;
        float input_values[kSharedChannelGroup];
#pragma unroll
        for(int channel = 0; channel < kSharedChannelGroup; ++channel)
        {
            const float *input_address = input_group + channel * param.inHW + input_offset;
            ptx::ldg_nc_0(input_values[channel], input_address, valid);
        }
        int shared_input_x = swizzle_shared_input_x(tile_input_w);
        int shared_input_offset = tile_input_h * shared_input_stride + shared_input_x;
        uint32_t shared_input_address = ptx::smem_u32addr(shared_inputs + shared_input_offset * kSharedChannelGroup);
        ptx::sts128(input_values[0], input_values[1], input_values[2], input_values[3], shared_input_address);
    }
    __syncthreads();

    int local_output_h = threadIdx.x / kSharedTileW;
    int local_output_w = threadIdx.x % kSharedTileW;
    int output_h = output_h_origin + local_output_h;
    int output_w = output_w_origin + local_output_w;
    if(output_h >= param.out_h || output_w >= param.out_w)
    {
        return;
    }

    float accumulators[kSharedChannelGroup];
#pragma unroll
    for(int channel = 0; channel < kSharedChannelGroup; ++channel)
        accumulators[channel] = bias[blockIdx.y * kSharedChannelGroup + channel];

    for(int kernel_h = 0; kernel_h < kSharedKernelSize; ++kernel_h)
    {
        int shared_input_h = local_output_h * param.Sh + kernel_h;
        for(int kernel_w = 0; kernel_w < kSharedKernelSize; ++kernel_w)
        {
            int shared_input_w = local_output_w * param.Sw + kernel_w;
            int shared_input_x = swizzle_shared_input_x(shared_input_w);
            int input_offset = shared_input_h * shared_input_stride + shared_input_x;
            int tap = kernel_h * kSharedKernelSize + kernel_w;
            float input_values[kSharedChannelGroup];
            float weight_values[kSharedChannelGroup];
            uint32_t input_address = ptx::smem_u32addr(shared_inputs + input_offset * kSharedChannelGroup);
            uint32_t weight_address = ptx::smem_u32addr(shared_weights + tap * kSharedChannelGroup);
            ptx::lds128(input_values[0], input_values[1], input_values[2], input_values[3], input_address);
            ptx::lds128(weight_values[0], weight_values[1], weight_values[2], weight_values[3], weight_address);
#pragma unroll
            for(int channel = 0; channel < kSharedChannelGroup; ++channel)
            {
                accumulators[channel] += weight_values[channel] * input_values[channel];
            }
        }
    }

    int output_position = output_h * param.out_w + output_w;
    int output_base =
        blockIdx.z * param.outBatchNumel + blockIdx.y * kSharedChannelGroup * param.outHW + output_position;
#pragma unroll
    for(int channel = 0; channel < kSharedChannelGroup; ++channel)
    {
        outputs[output_base + channel * param.outHW] = accumulators[channel];
    }
}

static size_t shared_input_bytes(const Conv2DParam &param)
{
    int input_tile_h = (kSharedTileH - 1) * param.Sh + param.Kh;
    int input_tile_w = (kSharedTileW - 1) * param.Sw + param.Kw;
    int shared_input_stride = input_tile_w + 1;
    size_t input_elements = static_cast<size_t>(kSharedChannelGroup) * input_tile_h * shared_input_stride;
    return (kSharedWeightElements + input_elements) * sizeof(float);
}

static bool supports_shared_input(const Conv2DParam &param)
{
    constexpr size_t maximum_shared_bytes = 48 * 1024;
    return param.out_ch % kSharedChannelGroup == 0 && param.Kh == 11 && param.Kw == 11 &&
           param.KhKw <= kSharedWeightStride && shared_input_bytes(param) <= maximum_shared_bytes;
}

static void launch_target(void *inputs, void *weights, void *bias, void *outputs, Conv2DParam param, int n,
                          cudaStream_t stream)
{
    if(!supports_shared_input(param))
    {
        if(param.Kh == 3 && param.Kw == 3)
        {
            launch_specialized<3>(inputs, weights, bias, outputs, param, n, stream);
        }
        else if(param.Kh == 5 && param.Kw == 5)
        {
            launch_specialized<5>(inputs, weights, bias, outputs, param, n, stream);
        }
        else if(param.Kh == 7 && param.Kw == 7)
        {
            size_t grouped_blocks = static_cast<size_t>((param.outHW + 255) / 256) * (param.out_ch / 4) * n;
            if(grouped_blocks < 104)
            {
                launch_k7_single_channel(inputs, weights, bias, outputs, param, n, stream);
            }
            else
            {
                launch_specialized<7>(inputs, weights, bias, outputs, param, n, stream);
            }
        }
        else
        {
            launch_custom(inputs, weights, bias, outputs, param, n, stream);
        }
        return;
    }

    int tiles_h = (param.out_h + kSharedTileH - 1) / kSharedTileH;
    int tiles_w = (param.out_w + kSharedTileW - 1) / kSharedTileW;
    dim3 block(kSharedBlockSize);
    dim3 grid(tiles_h * tiles_w, param.out_ch / kSharedChannelGroup, n);
    conv2d_4x16x16_shared_input_kernel<<<grid, block, shared_input_bytes(param), stream>>>(
        static_cast<const float *>(inputs), static_cast<const float *>(weights), static_cast<const float *>(bias),
        static_cast<float *>(outputs), param);
}

} // namespace

void launch_opt_fp32(float *inputs, float *weights, float *bias, float *outputs, Conv2DParam param, int n,
                     cudaStream_t stream)
{
    launch_target(inputs, weights, bias, outputs, param, n, stream);
}
