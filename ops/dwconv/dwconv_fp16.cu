#include "dwconv_fp16.cuh"

namespace
{
template <int Capacity>
__global__ void conv2d_fp16_baseline_kernel(const __half *inputs, const __half2 *packed_weights, const __half *bias,
                                            __half *outputs, Conv2DParam param)
{
    __shared__ __half2 smem_weights[4 * Capacity];

    int tid = threadIdx.x;
    int out_pos = blockIdx.x * blockDim.x + tid;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    for(int index = tid; index < 4 * Capacity; index += blockDim.x)
    {
        int channel_pair = index / Capacity;
        int kernel_pos = index % Capacity;
        __half2 value = __float2half2_rn(0.0f);
        if(kernel_pos < param.KhKw)
        {
            int weight_offset = blockIdx.y * 4 * param.KhKw + channel_pair * param.KhKw + kernel_pos;
            value = packed_weights[weight_offset];
        }
        smem_weights[index] = value;
    }
    // 权重合作写入后同步；后续 shared 只读，循环内无需 block 同步。
    __syncthreads();

    const __half *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 8 * param.inHW;
    __half2 output_frag[4];

#pragma unroll
    for(int channel_pair = 0; channel_pair < 4; ++channel_pair)
    {
        int channel = blockIdx.y * 8 + channel_pair * 2;
        output_frag[channel_pair] = __halves2half2(bias[channel], bias[channel + 1]);
    }

    for(int k = 0; k < param.KhKw; k += 4)
    {
        __half2 input_frag[4][4];

#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            int kernel_pos = k + i;
            int cur_h = posh_ori + kernel_pos / param.Kw;
            int cur_w = posw_ori + kernel_pos % param.Kw;
            bool valid =
                kernel_pos < param.KhKw && cur_h >= 0 && cur_h < param.in_h && cur_w >= 0 && cur_w < param.in_w;

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
                __half2 weight = smem_weights[channel_pair * Capacity + k + i];
                output_frag[channel_pair] = __hfma2(weight, input_frag[channel_pair][i], output_frag[channel_pair]);
            }
        }
    }

    if(out_pos < param.outHW)
    {
        int out_offset = blockIdx.z * param.outBatchNumel + blockIdx.y * 8 * param.outHW + out_pos;
#pragma unroll
        for(int channel_pair = 0; channel_pair < 4; ++channel_pair)
        {
            outputs[out_offset + channel_pair * 2 * param.outHW] = output_frag[channel_pair].x;
            outputs[out_offset + (channel_pair * 2 + 1) * param.outHW] = output_frag[channel_pair].y;
        }
    }
}

__device__ __forceinline__ void load_input_fp16(const __half *input_ptr, __half2 (&input_frag)[4][4],
                                                const Conv2DParam &param, int posh_ori, int posw_ori, int k)
{

#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        int kernel_pos = k + i;
        int cur_h = posh_ori + kernel_pos / param.Kw;
        int cur_w = posw_ori + kernel_pos % param.Kw;
        bool valid = kernel_pos < param.KhKw && cur_h >= 0 && cur_h < param.in_h && cur_w >= 0 && cur_w < param.in_w;

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
}

template <int Capacity>
__device__ __forceinline__ void accumulate_fp16(const __half2 *smem_weights, const __half2 (&input_frag)[4][4],
                                                __half2 (&output_frag)[4], int k)
{
#pragma unroll
    for(int channel_pair = 0; channel_pair < 4; ++channel_pair)
    {
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            __half2 weight = smem_weights[channel_pair * Capacity + k + i];
            output_frag[channel_pair] = __hfma2(weight, input_frag[channel_pair][i], output_frag[channel_pair]);
        }
    }
}

template <int Capacity>
__global__ void conv2d_fp16_prefetch_kernel(const __half *inputs, const __half2 *packed_weights, const __half *bias,
                                            __half *outputs, Conv2DParam param)
{
    __shared__ __half2 smem_weights[4 * Capacity];

    int tid = threadIdx.x;
    int out_pos = blockIdx.x * blockDim.x + tid;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    for(int index = tid; index < 4 * Capacity; index += blockDim.x)
    {
        int channel_pair = index / Capacity;
        int kernel_pos = index % Capacity;
        __half2 value = __float2half2_rn(0.0f);
        if(kernel_pos < param.KhKw)
        {
            int weight_offset = blockIdx.y * 4 * param.KhKw + channel_pair * param.KhKw + kernel_pos;
            value = packed_weights[weight_offset];
        }
        smem_weights[index] = value;
    }
    // 权重合作写入后同步；后续 shared 只读，循环内无需 block 同步。
    __syncthreads();

    const __half *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 8 * param.inHW;
    __half2 output_frag[4];

#pragma unroll
    for(int channel_pair = 0; channel_pair < 4; ++channel_pair)
    {
        int channel = blockIdx.y * 8 + channel_pair * 2;
        output_frag[channel_pair] = __halves2half2(bias[channel], bias[channel + 1]);
    }

    // 权重一次加载；A/B 两份输入片段交替预取，保持每通道的 tap 累加顺序。
    __half2 input_a[4][4], input_b[4][4];
    load_input_fp16(input_ptr, input_a, param, posh_ori, posw_ori, 0);
    for(int k = 0; k < param.KhKw; k += 8)
    {
        if(k + 4 < param.KhKw)
        {
            load_input_fp16(input_ptr, input_b, param, posh_ori, posw_ori, k + 4);
        }

        accumulate_fp16<Capacity>(smem_weights, input_a, output_frag, k);

        if(k + 8 < param.KhKw)
        {
            load_input_fp16(input_ptr, input_a, param, posh_ori, posw_ori, k + 8);
        }

        if(k + 4 < param.KhKw)
        {
            accumulate_fp16<Capacity>(smem_weights, input_b, output_frag, k + 4);
        }
    }

    if(out_pos < param.outHW)
    {
        int out_offset = blockIdx.z * param.outBatchNumel + blockIdx.y * 8 * param.outHW + out_pos;
#pragma unroll
        for(int channel_pair = 0; channel_pair < 4; ++channel_pair)
        {
            outputs[out_offset + channel_pair * 2 * param.outHW] = output_frag[channel_pair].x;
            outputs[out_offset + (channel_pair * 2 + 1) * param.outHW] = output_frag[channel_pair].y;
        }
    }
}

} // namespace

void launch_baseline_fp16_k128(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                               const Conv2DParam &param, int n, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((param.outHW + block.x - 1) / block.x, param.out_ch / 8, n);
    conv2d_fp16_baseline_kernel<128><<<grid, block, 0, stream>>>(inputs, packed_weights, bias, outputs, param);
}

void launch_baseline_fp16_k64(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                              const Conv2DParam &param, int n, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((param.outHW + block.x - 1) / block.x, param.out_ch / 8, n);
    conv2d_fp16_baseline_kernel<64><<<grid, block, 0, stream>>>(inputs, packed_weights, bias, outputs, param);
}

void launch_baseline_fp16_k32(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                              const Conv2DParam &param, int n, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((param.outHW + block.x - 1) / block.x, param.out_ch / 8, n);
    conv2d_fp16_baseline_kernel<32><<<grid, block, 0, stream>>>(inputs, packed_weights, bias, outputs, param);
}

void launch_prefetch_fp16_k128(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                               const Conv2DParam &param, int n, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((param.outHW + block.x - 1) / block.x, param.out_ch / 8, n);
    conv2d_fp16_prefetch_kernel<128><<<grid, block, 0, stream>>>(inputs, packed_weights, bias, outputs, param);
}

void launch_baseline_fp16(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                          const Conv2DParam &param, int n, cudaStream_t stream)
{
    if(param.KhKw <= 32)
    {
        launch_baseline_fp16_k32(inputs, packed_weights, bias, outputs, param, n, stream);
    }
    else if(param.KhKw <= 64)
    {
        launch_baseline_fp16_k64(inputs, packed_weights, bias, outputs, param, n, stream);
    }
    else
    {
        launch_baseline_fp16_k128(inputs, packed_weights, bias, outputs, param, n, stream);
    }
}