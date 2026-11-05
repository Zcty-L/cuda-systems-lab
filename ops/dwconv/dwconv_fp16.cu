#include <stdexcept>
#include <type_traits>
#include "dwconv_fp16.cuh"

namespace
{

template <bool AccumulateFP32> using Accumulator = std::conditional_t<AccumulateFP32, float2, __half2>;

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

template <int WeightStride, bool AccumulateFP32 = true>
__device__ __forceinline__ void accumulate_fp16(const __half2 *smem_weights, const __half2 (&input_frag)[4][4],
                                                Accumulator<AccumulateFP32> (&output_frag)[4], int k)
{
#pragma unroll
    for(int channel_pair = 0; channel_pair < 4; ++channel_pair)
    {
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            __half2 weight = smem_weights[channel_pair * WeightStride + k + i];
            if constexpr(AccumulateFP32)
            {
                float2 w = __half22float2(weight);
                float2 x = __half22float2(input_frag[channel_pair][i]);
                output_frag[channel_pair].x = fmaf(w.x, x.x, output_frag[channel_pair].x);
                output_frag[channel_pair].y = fmaf(w.y, x.y, output_frag[channel_pair].y);
            }
            else
            {
                output_frag[channel_pair] = __hfma2(weight, input_frag[channel_pair][i], output_frag[channel_pair]);
            }
        }
    }
}

template <int WeightStride, bool UseBias, bool AccumulateFP32 = true>
__global__ void dwconv_fp16_groups_kernel(const __half *inputs, const __half2 *packed_weights, const __half *bias,
                                          __half *outputs, Conv2DParam param)
{
    static_assert(WeightStride == 32 || WeightStride == 64 || WeightStride == 128, "WeightStride 仅支持 32、64、128");

    __shared__ __align__(16) __half2 smem_weights[4 * WeightStride];

    int tid = threadIdx.x;
    int out_pos = blockIdx.x * blockDim.x + tid;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    for(int index = tid; index < 4 * WeightStride; index += blockDim.x)
    {
        int channel_pair = index / WeightStride;
        int kernel_pos = index % WeightStride;
        __half2 value = __float2half2_rn(0.0f);
        if(kernel_pos < param.KhKw)
        {
            int weight_offset = blockIdx.y * 4 * param.KhKw + channel_pair * param.KhKw + kernel_pos;
            value = packed_weights[weight_offset];
        }
        smem_weights[index] = value;
    }
    __syncthreads();

    const __half *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 8 * param.inHW;

    Accumulator<AccumulateFP32> output_frag[4];

#pragma unroll
    for(int channel_pair = 0; channel_pair < 4; ++channel_pair)
    {
        if constexpr(UseBias)
        {
            int channel = blockIdx.y * 8 + channel_pair * 2;
            if constexpr(AccumulateFP32)
                output_frag[channel_pair] = make_float2(__half2float(bias[channel]), __half2float(bias[channel + 1]));
            else
                output_frag[channel_pair] = __halves2half2(bias[channel], bias[channel + 1]);
        }
        else
        {
            if constexpr(AccumulateFP32)
                output_frag[channel_pair] = make_float2(0.0f, 0.0f);
            else
                output_frag[channel_pair] = __float2half2_rn(0.0f);
        }
    }

    for(int k = 0; k < param.KhKw; k += 4)
    {
        __half2 input_frag[4][4];
        load_input_fp16(input_ptr, input_frag, param, posh_ori, posw_ori, k);
        accumulate_fp16<WeightStride, AccumulateFP32>(smem_weights, input_frag, output_frag, k);
    }

    if(out_pos < param.outHW)
    {
        int out_offset = blockIdx.z * param.outBatchNumel + blockIdx.y * 8 * param.outHW + out_pos;
#pragma unroll
        for(int channel_pair = 0; channel_pair < 4; ++channel_pair)
        {
            if constexpr(AccumulateFP32)
            {
                outputs[out_offset + channel_pair * 2 * param.outHW] = __float2half_rn(output_frag[channel_pair].x);
                outputs[out_offset + (channel_pair * 2 + 1) * param.outHW] =
                    __float2half_rn(output_frag[channel_pair].y);
            }
            else
            {
                outputs[out_offset + channel_pair * 2 * param.outHW] = __low2half(output_frag[channel_pair]);
                outputs[out_offset + (channel_pair * 2 + 1) * param.outHW] = __high2half(output_frag[channel_pair]);
            }
        }
    }
}

template <int WeightStride, bool AccumulateFP32>
void launch_dwconv_fp16_impl(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                             const Conv2DParam &param, dim3 grid, cudaStream_t stream)
{
    if(param.use_bias)
    {
        dwconv_fp16_groups_kernel<WeightStride, true, AccumulateFP32>
            <<<grid, 256, 0, stream>>>(inputs, packed_weights, bias, outputs, param);
    }
    else
    {
        dwconv_fp16_groups_kernel<WeightStride, false, AccumulateFP32>
            <<<grid, 256, 0, stream>>>(inputs, packed_weights, bias, outputs, param);
    }
}

template <bool AccumulateFP32>
void launch_dwconv_fp16_dispatch(const __half *inputs, const __half2 *packed_weights, const __half *bias,
                                 __half *outputs, const Conv2DParam &param, dim3 grid, cudaStream_t stream,
                                 int weight_stride)
{
    if(weight_stride == 32)
    {
        launch_dwconv_fp16_impl<32, AccumulateFP32>(inputs, packed_weights, bias, outputs, param, grid, stream);
    }
    else if(weight_stride == 64)
    {
        launch_dwconv_fp16_impl<64, AccumulateFP32>(inputs, packed_weights, bias, outputs, param, grid, stream);
    }
    else
    {
        launch_dwconv_fp16_impl<128, AccumulateFP32>(inputs, packed_weights, bias, outputs, param, grid, stream);
    }
}

int resolve_weight_stride(const Conv2DParam &param, const __half *bias, int weight_stride)
{
    if(param.KhKw < 1 || param.KhKw > 128)
        throw std::invalid_argument("DWConv FP16 要求 1 <= KhKw <= 128");
    if(weight_stride == 0)
        weight_stride = param.KhKw <= 32 ? 32 : param.KhKw <= 64 ? 64 : 128;
    if((weight_stride != 32 && weight_stride != 64 && weight_stride != 128) || param.KhKw > weight_stride)
        throw std::invalid_argument("DWConv FP16 的 weight_stride 必须为 0、32、64、128，且容量不小于 KhKw");
    if(param.use_bias && bias == nullptr)
        throw std::invalid_argument("DWConv FP16 启用 bias 时指针不能为空");
    return weight_stride;
}

} // namespace

void launch_dwconv_fp16(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                        const Conv2DParam &param, int n, cudaStream_t stream /* = nullptr */,
                        int weight_stride /* = 0 */, bool accumulate_fp32 /* = true */)
{
    weight_stride = resolve_weight_stride(param, bias, weight_stride);
    dim3 grid((param.outHW + 255) / 256, param.out_ch / 8, n);

    if(accumulate_fp32)
    {
        launch_dwconv_fp16_dispatch<true>(inputs, packed_weights, bias, outputs, param, grid, stream, weight_stride);
    }
    else
    {
        launch_dwconv_fp16_dispatch<false>(inputs, packed_weights, bias, outputs, param, grid, stream, weight_stride);
    }
}
