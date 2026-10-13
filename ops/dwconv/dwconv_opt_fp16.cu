#include "dwconv_opt.cuh"
#include <cuda_lab/ptx_utils.cuh>

namespace
{
// FP16 存储，两个通道独立使用 FP32 FMA 累加。
__device__ __forceinline__ float2 accumulate_fp32(__half2 weight, __half2 input, float2 sum)
{
    float2 w = __half22float2(weight);
    float2 x = __half22float2(input);
    return make_float2(fmaf(w.x, x.x, sum.x), fmaf(w.y, x.y, sum.y));
}

constexpr int kKernelCapacity = 128;
constexpr int kSharedTileH = 16;
constexpr int kSharedTileW = 16;
constexpr int kSharedBlockSize = kSharedTileH * kSharedTileW;
constexpr int kSharedChannelPairs = 4;
constexpr int kSharedWeightStride = 128;
constexpr int kSharedWeightElements = kSharedChannelPairs * kSharedWeightStride;
template <int KernelSize>
__global__ void conv2d_4x128x256_fp16_groups_kernel(const __half *inputs, const __half2 *packed_weights,
                                                    const __half *bias, __half *outputs, Conv2DParam param)
{
    constexpr bool specialized = KernelSize > 0;
    constexpr int specialized_elements = KernelSize * KernelSize;
    constexpr int specialized_capacity = (specialized_elements + 3) / 4 * 4;
    constexpr int shared_capacity = specialized ? specialized_capacity : kKernelCapacity;
    int kernel_width = specialized ? KernelSize : param.Kw;
    int kernel_elements = specialized ? specialized_elements : param.KhKw;
    __shared__ __half2 smem_weights[4 * shared_capacity];

    int tid = threadIdx.x;
    int out_pos = blockIdx.x * blockDim.x + tid;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    for(int index = tid; index < 4 * shared_capacity; index += blockDim.x)
    {
        int channel_pair = index / shared_capacity;
        int kernel_pos = index % shared_capacity;
        __half2 value = __float2half2_rn(0.0f);
        if(kernel_pos < kernel_elements)
        {
            int weight_offset = blockIdx.y * 4 * kernel_elements + channel_pair * kernel_elements + kernel_pos;
            value = packed_weights[weight_offset];
        }
        smem_weights[index] = value;
    }
    __syncthreads();

    const __half *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 8 * param.inHW;
    float2 output_frag[4];

#pragma unroll
    for(int channel_pair = 0; channel_pair < 4; ++channel_pair)
    {
        int channel = blockIdx.y * 8 + channel_pair * 2;
        output_frag[channel_pair] = make_float2(__half2float(bias[channel]), __half2float(bias[channel + 1]));
    }

    for(int k = 0; k < kernel_elements; k += 4)
    {
        __half2 input_frag[4][4];

#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            int kernel_pos = k + i;
            int cur_h = posh_ori + kernel_pos / kernel_width;
            int cur_w = posw_ori + kernel_pos % kernel_width;
            bool valid =
                kernel_pos < kernel_elements && cur_h >= 0 && cur_h < param.in_h && cur_w >= 0 && cur_w < param.in_w;

#pragma unroll
            for(int channel_pair = 0; channel_pair < 4; ++channel_pair)
            {
                if(valid)
                {
                    int input_offset = cur_h * param.in_w + cur_w + channel_pair * 2 * param.inHW;
                    input_frag[channel_pair][i] =
                        __halves2half2(input_ptr[input_offset], input_ptr[input_offset + param.inHW]);
                }
                else
                {
                    input_frag[channel_pair][i] = __float2half2_rn(0.0f);
                }
            }
        }

#pragma unroll
        for(int channel_pair = 0; channel_pair < 4; ++channel_pair)
        {
#pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                __half2 weight = smem_weights[channel_pair * shared_capacity + k + i];
                output_frag[channel_pair] =
                    accumulate_fp32(weight, input_frag[channel_pair][i], output_frag[channel_pair]);
            }
        }
    }

    if(out_pos < param.outHW)
    {
        int out_offset = blockIdx.z * param.outBatchNumel + blockIdx.y * 8 * param.outHW + out_pos;
#pragma unroll
        for(int channel_pair = 0; channel_pair < 4; ++channel_pair)
        {
            outputs[out_offset + channel_pair * 2 * param.outHW] = __float2half_rn(output_frag[channel_pair].x);
            outputs[out_offset + (channel_pair * 2 + 1) * param.outHW] = __float2half_rn(output_frag[channel_pair].y);
        }
    }
}

template <int KernelSize>
static void launch_specialized(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                               const Conv2DParam &param, int n)
{
    dim3 block(256);
    dim3 grid((param.outHW + block.x - 1) / block.x, param.out_ch / 8, n);
    conv2d_4x128x256_fp16_groups_kernel<KernelSize><<<grid, block>>>(inputs, packed_weights, bias, outputs, param);
}

template <int KernelSize>
__global__ void conv2d_1x128x256_fp16_groups_kernel(const __half *inputs, const __half2 *packed_weights,
                                                    const __half *bias, __half *outputs, Conv2DParam param)
{
    constexpr int kernel_elements = KernelSize * KernelSize;
    constexpr int shared_capacity = (kernel_elements + 3) / 4 * 4;
    __shared__ __half2 shared_weights[shared_capacity];

    int tid = threadIdx.x;
    if(tid < shared_capacity)
    {
        shared_weights[tid] =
            tid < kernel_elements ? packed_weights[blockIdx.y * kernel_elements + tid] : __float2half2_rn(0.0f);
    }
    __syncthreads();

    int out_pos = blockIdx.x * blockDim.x + tid;
    int posh_origin = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_origin = (out_pos % param.out_w) * param.Sw - param.Pw;
    const __half *input_pair = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 2 * param.inHW;
    int channel = blockIdx.y * 2;
    float2 output_value = make_float2(__half2float(bias[channel]), __half2float(bias[channel + 1]));

    for(int tap_base = 0; tap_base < kernel_elements; tap_base += 4)
    {
        __half2 input_values[4];
#pragma unroll
        for(int index = 0; index < 4; ++index)
        {
            int tap = tap_base + index;
            int input_h = posh_origin + tap / KernelSize;
            int input_w = posw_origin + tap % KernelSize;
            bool valid =
                tap < kernel_elements && input_h >= 0 && input_h < param.in_h && input_w >= 0 && input_w < param.in_w;
            if(valid)
            {
                int input_offset = input_h * param.in_w + input_w;
                input_values[index] = __halves2half2(input_pair[input_offset], input_pair[input_offset + param.inHW]);
            }
            else
            {
                input_values[index] = __float2half2_rn(0.0f);
            }
        }

#pragma unroll
        for(int index = 0; index < 4; ++index)
        {
            output_value = accumulate_fp32(shared_weights[tap_base + index], input_values[index], output_value);
        }
    }

    if(out_pos < param.outHW)
    {
        int output_offset = blockIdx.z * param.outBatchNumel + blockIdx.y * 2 * param.outHW + out_pos;
        outputs[output_offset] = __float2half_rn(output_value.x);
        outputs[output_offset + param.outHW] = __float2half_rn(output_value.y);
    }
}

template <int KernelSize>
static void launch_pair(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                        const Conv2DParam &param, int n)
{
    dim3 block(256);
    dim3 grid((param.outHW + block.x - 1) / block.x, param.out_ch / 2, n);
    conv2d_1x128x256_fp16_groups_kernel<KernelSize><<<grid, block>>>(inputs, packed_weights, bias, outputs, param);
}

__device__ __forceinline__ int swizzle_shared_input_x(int input_x)
{
    return input_x ^ ((input_x >> 3) & 1);
}

template <int KernelSize>
__global__ void conv2d_8x16x16_fp16_shared_input_kernel(const __half *inputs, const __half2 *packed_weights,
                                                        const __half *bias, __half *outputs, Conv2DParam param)
{
    constexpr int kernel_elements = KernelSize * KernelSize;
    extern __shared__ __align__(16) __half2 shared[];
    __half2 *shared_weights = shared;
    __half2 *shared_inputs = shared_weights + kSharedWeightElements;

    int input_tile_h = (kSharedTileH - 1) * param.Sh + KernelSize;
    int input_tile_w = (kSharedTileW - 1) * param.Sw + KernelSize;
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
        int tap = index / kSharedChannelPairs;
        int channel_pair = index % kSharedChannelPairs;
        __half2 value = __float2half2_rn(0.0f);
        if(tap < kernel_elements)
        {
            value = packed_weights[blockIdx.y * kSharedChannelPairs * kernel_elements + channel_pair * kernel_elements +
                                   tap];
        }
        shared_weights[index] = value;
    }

    const __half *input_group = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 8 * param.inHW;
    for(int spatial = threadIdx.x; spatial < input_tile_hw; spatial += blockDim.x)
    {
        int tile_input_h = spatial / input_tile_w;
        int tile_input_w = spatial % input_tile_w;
        int input_h = input_h_origin + tile_input_h;
        int input_w = input_w_origin + tile_input_w;
        bool valid = input_h >= 0 && input_w >= 0 && input_h < param.in_h && input_w < param.in_w;
        int input_offset = input_h * param.in_w + input_w;
        __half2 input_values[kSharedChannelPairs];
#pragma unroll
        for(int channel_pair = 0; channel_pair < kSharedChannelPairs; ++channel_pair)
        {
            if(valid)
            {
                const __half *pair_input = input_group + channel_pair * 2 * param.inHW + input_offset;
                input_values[channel_pair] = __halves2half2(pair_input[0], pair_input[param.inHW]);
            }
            else
            {
                input_values[channel_pair] = __float2half2_rn(0.0f);
            }
        }
        int shared_input_x = swizzle_shared_input_x(tile_input_w);
        int shared_input_offset = tile_input_h * shared_input_stride + shared_input_x;
        uint32_t shared_input_address = ptx::smem_u32addr(shared_inputs + shared_input_offset * kSharedChannelPairs);
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

    float2 accumulators[kSharedChannelPairs];
#pragma unroll
    for(int channel_pair = 0; channel_pair < kSharedChannelPairs; ++channel_pair)
    {
        int channel = blockIdx.y * 8 + channel_pair * 2;
        accumulators[channel_pair] = make_float2(__half2float(bias[channel]), __half2float(bias[channel + 1]));
    }
    for(int kernel_h = 0; kernel_h < KernelSize; ++kernel_h)
    {
        int shared_input_h = local_output_h * param.Sh + kernel_h;
        for(int kernel_w = 0; kernel_w < KernelSize; ++kernel_w)
        {
            int shared_input_w = local_output_w * param.Sw + kernel_w;
            int shared_input_x = swizzle_shared_input_x(shared_input_w);
            int input_offset = shared_input_h * shared_input_stride + shared_input_x;
            int tap = kernel_h * KernelSize + kernel_w;
            __half2 input_values[kSharedChannelPairs];
            __half2 weight_values[kSharedChannelPairs];
            uint32_t input_address = ptx::smem_u32addr(shared_inputs + input_offset * kSharedChannelPairs);
            uint32_t weight_address = ptx::smem_u32addr(shared_weights + tap * kSharedChannelPairs);
            ptx::lds128(input_values[0], input_values[1], input_values[2], input_values[3], input_address);
            ptx::lds128(weight_values[0], weight_values[1], weight_values[2], weight_values[3], weight_address);
#pragma unroll
            for(int channel_pair = 0; channel_pair < kSharedChannelPairs; ++channel_pair)
            {
                accumulators[channel_pair] = accumulate_fp32(weight_values[channel_pair], input_values[channel_pair],
                                                             accumulators[channel_pair]);
            }
        }
    }

    int output_position = output_h * param.out_w + output_w;
    int output_base = blockIdx.z * param.outBatchNumel + blockIdx.y * 8 * param.outHW + output_position;
#pragma unroll
    for(int channel_pair = 0; channel_pair < kSharedChannelPairs; ++channel_pair)
    {
        outputs[output_base + channel_pair * 2 * param.outHW] = __float2half_rn(accumulators[channel_pair].x);
        outputs[output_base + (channel_pair * 2 + 1) * param.outHW] = __float2half_rn(accumulators[channel_pair].y);
    }
}

static size_t shared_input_bytes(const Conv2DParam &param)
{
    int input_tile_h = (kSharedTileH - 1) * param.Sh + param.Kh;
    int input_tile_w = (kSharedTileW - 1) * param.Sw + param.Kw;
    int shared_input_stride = input_tile_w + 1;
    size_t input_elements = static_cast<size_t>(kSharedChannelPairs) * input_tile_h * shared_input_stride;
    return (kSharedWeightElements + input_elements) * sizeof(__half2);
}

static bool supports_shared_input(const Conv2DParam &param)
{
    constexpr size_t maximum_shared_bytes = 48 * 1024;
    return param.out_ch % 8 == 0 && param.Kh == param.Kw && (param.Kh == 9 || param.Kh == 11) &&
           shared_input_bytes(param) <= maximum_shared_bytes;
}

template <int KernelSize>
static void launch_shared_input(const __half *inputs, const __half2 *packed_weights, const __half *bias,
                                __half *outputs, const Conv2DParam &param, int n)
{
    int tiles_h = (param.out_h + kSharedTileH - 1) / kSharedTileH;
    int tiles_w = (param.out_w + kSharedTileW - 1) / kSharedTileW;
    dim3 block(kSharedBlockSize);
    dim3 grid(tiles_h * tiles_w, param.out_ch / 8, n);
    conv2d_8x16x16_fp16_shared_input_kernel<KernelSize>
        <<<grid, block, shared_input_bytes(param)>>>(inputs, packed_weights, bias, outputs, param);
}

static void launch_baseline(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                            const Conv2DParam &param, int n)
{
    if(param.Kh == 3 && param.Kw == 3)
    {
        launch_pair<3>(inputs, packed_weights, bias, outputs, param, n);
    }
    else if(param.Kh == 5 && param.Kw == 5)
    {
        launch_pair<5>(inputs, packed_weights, bias, outputs, param, n);
    }
    else if(param.Kh == 7 && param.Kw == 7)
    {
        launch_pair<7>(inputs, packed_weights, bias, outputs, param, n);
    }
    else if(param.Kh == 9 && param.Kw == 9)
    {
        size_t pair_blocks = static_cast<size_t>((param.outHW + 255) / 256) * (param.out_ch / 2) * n;
        if(pair_blocks >= 140 && supports_shared_input(param))
        {
            launch_shared_input<9>(inputs, packed_weights, bias, outputs, param, n);
        }
        else
        {
            launch_pair<9>(inputs, packed_weights, bias, outputs, param, n);
        }
    }
    else if(param.Kh == 11 && param.Kw == 11)
    {
        if(supports_shared_input(param))
        {
            launch_shared_input<11>(inputs, packed_weights, bias, outputs, param, n);
        }
        else
        {
            launch_specialized<11>(inputs, packed_weights, bias, outputs, param, n);
        }
    }
    else
    {
        launch_specialized<0>(inputs, packed_weights, bias, outputs, param, n);
    }
}

} // namespace

void launch_opt_fp16(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                     const Conv2DParam &param, int n)
{
    launch_baseline(inputs, packed_weights, bias, outputs, param, n);
}
