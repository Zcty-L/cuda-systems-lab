// 抽离自 PyTorch 9a3eac3b8ffba6e54e61df7ebecfb0afcc5e5a74。
// 下文来源路径和行号均对应此固定提交，行号从 1 开始。
// 原始 CUDA kernel、分派及辅助函数来源见 README.md；版权许可见 PYTORCH-LICENSE。
#include "dwconv_pytorch.cuh"
#include <cuda_lab/cuda_check.cuh>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace
{

// 来源：aten/src/ATen/cuda/detail/KernelUtils.h:23。
constexpr int CUDA_NUM_THREADS = 1024;

// 来源：torch/headeronly/macros/Macros.h:415，非 ROCm 的 C10_WARP_SIZE=32。
constexpr int C10_WARP_SIZE = 32;

// 本接口仅实例化 float / __half，对应上游 acc_type<scalar_t, true>。
// 本地类型适配；来源：aten/src/ATen/AccumulateType.h:68-77、128、131、136。
template <typename T> using AccType = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, __half>, float>;

// 来源：aten/src/ATen/cuda/DeviceUtils.cuh:69-77，仅抽离 NVIDIA CUDA 分支。
template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = warpSize,
                                            unsigned int mask = 0xffffffff)
{
    return __shfl_down_sync(mask, value, delta, width);
}

namespace cuda_utils
{

// 来源：aten/src/ATen/native/cuda/block_reduce.cuh:29-36。
template <typename T> __inline__ __device__ T WarpReduceSum(T val)
{
#pragma unroll
    for(int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1)
    {
        val += WARP_SHFL_DOWN(val, offset);
    }
    return val;
}

// 来源：aten/src/ATen/native/cuda/block_reduce.cuh:51-57。
struct Block1D
{
    static __forceinline__ __device__ int Tid()
    {
        return threadIdx.x;
    }

    static __forceinline__ __device__ int Warps()
    {
        return blockDim.x / C10_WARP_SIZE;
    }
};

// 来源：aten/src/ATen/native/cuda/block_reduce.cuh:76-92。
template <typename T, typename B = Block1D> __inline__ __device__ T BlockReduceSum(T val, T *shared)
{
    const int tid = B::Tid();
    const int lid = tid % C10_WARP_SIZE;
    const int wid = tid / C10_WARP_SIZE;
    val = WarpReduceSum(val);
    __syncthreads(); // prevent races when BlockReduces are called in a row.
    if(lid == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();
    val = (tid < B::Warps()) ? shared[lid] : T(0);
    if(wid == 0)
    {
        val = WarpReduceSum(val);
    }
    return val;
}

} // namespace cuda_utils

// 沿用上游的 64 位循环游标，避免最后一次 grid-stride 递增溢出。
// 来源：aten/src/ATen/cuda/detail/KernelUtils.h:15-17。
#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                                                                        \
    int64_t _i_n_d_e_x = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;                                             \
    for(index_type i = _i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x += blockDim.x * gridDim.x, i = _i_n_d_e_x)

// 除 accessor 改为裸指针、累加类型和 CUDA 宏适配外，以下四个 kernel 保留上游计算体。
// launch_bounds 宏定义来源：torch/headeronly/macros/Macros.h:319-320。
// 来源：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:32-138。
template <typename scalar_t, typename index_t>
__global__ void __launch_bounds__(CUDA_NUM_THREADS)
    conv_depthwise2d_forward_kernel_generic(const scalar_t *input, scalar_t *output, const scalar_t *weight,
                                            const scalar_t *bias, bool biasEnabled, index_t totalElements,
                                            const int outputChannels, const int depthwiseMultiplier,
                                            const int inputWidth, const int inputHeight, const int outputWidth,
                                            const int outputHeight, const int kernelWidth, const int kernelHeight,
                                            const int strideWidth, const int strideHeight, const int padWidth,
                                            const int padHeight, const int dilationWidth, const int dilationHeight)
{
    using acc_t = AccType<scalar_t>;

    CUDA_KERNEL_LOOP_TYPE(linearIndex, totalElements, index_t)
    {
        // calculate n,c,h,w indices, replacing modulos by divide and multiply add,
        // result is same as would be in the code below
        // const int n = linearIndex / batchStride; //batchStride = outputChannels * outputHeight * outputWidth
        // const int c = (linearIndex / channelStride) % outputChannels; //channelStride = outputHeight * outputWidth
        // const int h = (linearIndex / outputWidth) % outputHeight;
        // const int w = linearIndex % outputWidth;

        int indtmp1 = linearIndex / outputWidth;
        const int w = linearIndex - indtmp1 * outputWidth;
        int indtmp2 = indtmp1 / outputHeight;
        const int h = indtmp1 - indtmp2 * outputHeight;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1 / outputChannels;
        const int c = indtmp1 - indtmp2 * outputChannels;
        const int n = indtmp2;

        int inputChannel = c;
        int inputChannels = outputChannels;
        if(depthwiseMultiplier != 1)
        {
            inputChannel /= depthwiseMultiplier;
            inputChannels /= depthwiseMultiplier;
        }

        int weightOffset = c * kernelHeight * kernelWidth;

        // By precisely computing the filtering boundaries, we avoid repeating several
        // expensive edge condition checks for every fetched item. If the input element is
        // resident in L1, then the extra branches and comparisons would have been
        // comparable in terms of cycles with the actual data fetch. Therefore computing
        // boundaries ahead of the loop showed significant performance boost.

        int kHmin = 0, kHmax = kernelHeight, kWmin = 0, kWmax = kernelWidth;

        // Top
        int h_in_min = -padHeight + h * strideHeight;
        if(h_in_min < 0)
        {
            kHmin = -h_in_min / dilationHeight;
            if((-h_in_min) % dilationHeight > 0)
            {
                kHmin++;
            }
        }

        // Bottom
        int h_in_max = h_in_min + (kernelHeight - 1) * dilationHeight - inputHeight + 1;
        if(h_in_max >= 0)
        {
            kHmax = kernelHeight - h_in_max / dilationHeight;
            if(h_in_max % dilationHeight > 0)
            {
                kHmax--;
            }
        }

        // Left
        int w_in_min = -padWidth + w * strideWidth;
        if(w_in_min < 0)
        {
            kWmin = -w_in_min / dilationWidth;
            if((-w_in_min) % dilationWidth > 0)
            {
                kWmin++;
            }
        }

        // Right
        int w_in_max = w_in_min + (kernelWidth - 1) * dilationWidth - inputWidth + 1;
        if(w_in_max >= 0)
        {
            kWmax = kernelWidth - w_in_max / dilationWidth;
            if(w_in_max % dilationWidth > 0)
            {
                kWmax--;
            }
        }

        acc_t value = biasEnabled ? static_cast<acc_t>(bias[c]) : acc_t(0);
        const index_t offset0 = (n * inputChannels + inputChannel) * inputHeight * inputWidth;

        for(int kH = kHmin; kH < kHmax; ++kH)
        {
            const int h_in = -padHeight + h * strideHeight + kH * dilationHeight;
            for(int kW = kWmin; kW < kWmax; ++kW)
            {
                const int w_in = -padWidth + w * strideWidth + kW * dilationWidth;
                const index_t offset = offset0 + h_in * inputWidth + w_in;
                value += (static_cast<acc_t>(weight[weightOffset + kH * kernelWidth + kW]) *
                          static_cast<acc_t>(input[offset]));
            }
        }
        output[linearIndex] = static_cast<scalar_t>(value);
    }
}

// 来源：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:140-213。
template <int kSize, typename scalar_t, typename index_t>
__global__ void __launch_bounds__(CUDA_NUM_THREADS)
    conv_depthwise2d_forward_kernel(const scalar_t *input, scalar_t *output, const scalar_t *weight,
                                    const scalar_t *bias, bool biasEnabled, index_t totalElements,
                                    const int outputChannels, const int depthwiseMultiplier, const int inputWidth,
                                    const int inputHeight, const int outputWidth, const int outputHeight,
                                    const int kernelWidth, const int kernelHeight, const int strideWidth,
                                    const int strideHeight, const int padWidth, const int padHeight,
                                    const int dilationWidth, const int dilationHeight)
{
    using acc_t = AccType<scalar_t>;
    const int KW_LIMIT = (kSize != 0) ? kSize : kernelWidth;
    const int KH_LIMIT = (kSize != 0) ? kSize : kernelHeight;

    CUDA_KERNEL_LOOP_TYPE(linearIndex, totalElements, index_t)
    {
        // calculate n,c,h,w indices, replacing modulos by divide and multiply add,
        // result is same as would be in the code below
        // const int n = linearIndex / batchStride; //batchStride = outputChannels * outputHeight * outputWidth
        // const int c = (linearIndex / channelStride) % outputChannels; //channelStride = outputHeight * outputWidth
        // const int h = (linearIndex / outputWidth) % outputHeight;
        // const int w = linearIndex % outputWidth;

        int indtmp1 = linearIndex / outputWidth;
        const int w = linearIndex - indtmp1 * outputWidth;
        int indtmp2 = indtmp1 / outputHeight;
        const int h = indtmp1 - indtmp2 * outputHeight;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1 / outputChannels;
        const int c = indtmp1 - indtmp2 * outputChannels;
        const int n = indtmp2;

        int inputChannel = c;
        int inputChannels = outputChannels;
        if(depthwiseMultiplier != 1)
        {
            inputChannel /= depthwiseMultiplier;
            inputChannels /= depthwiseMultiplier;
        }

        int weightOffset = c * kernelHeight * kernelWidth;

        acc_t value = biasEnabled ? static_cast<acc_t>(bias[c]) : acc_t(0);
        const index_t offset0 = (n * inputChannels + inputChannel) * inputHeight * inputWidth;
#pragma unroll
        for(int kH = 0; kH < KH_LIMIT; ++kH)
        {
#pragma unroll
            for(int kW = 0; kW < KW_LIMIT; ++kW)
            {
                const int h_in = -padHeight + h * strideHeight + kH * dilationHeight;
                const int w_in = -padWidth + w * strideWidth + kW * dilationWidth;

                if((h_in >= 0) && (h_in < inputHeight) && (w_in >= 0) && (w_in < inputWidth))
                {
                    const index_t offset = offset0 + h_in * inputWidth + w_in;
                    value += (static_cast<acc_t>(weight[weightOffset]) * static_cast<acc_t>(input[offset]));
                }
                ++weightOffset;
            }
        }
        output[linearIndex] = static_cast<scalar_t>(value);
    }
}

// 来源：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:215-280。
template <int kSize, int stride, typename scalar_t, typename index_t>
__launch_bounds__(CUDA_NUM_THREADS) __global__
    void conv_depthwise2d_backward_kernel(const scalar_t *grad_output, scalar_t *grad_input, const scalar_t *weight,
                                          index_t totalElements, const int inputChannels, const int depthwiseMultiplier,
                                          const int outputChannels, const int inputWidth, const int inputHeight,
                                          const int outputWidth, const int outputHeight, const int kernelWidth,
                                          const int kernelHeight, const int strideWidth, const int strideHeight,
                                          const int padWidth, const int padHeight, const int dilationWidth,
                                          const int dilationHeight)
{
    using acc_t = AccType<scalar_t>;
    const int KW_LIMIT = (kSize != 0) ? kSize : kernelWidth;
    const int KH_LIMIT = (kSize != 0) ? kSize : kernelHeight;
    const int strideW = (stride != 0) ? stride : strideWidth;
    const int strideH = (stride != 0) ? stride : strideHeight;

    CUDA_KERNEL_LOOP_TYPE(linearIndex, totalElements, index_t)
    {
        int indtmp1 = linearIndex / inputWidth;
        const int w = linearIndex - indtmp1 * inputWidth;
        int indtmp2 = indtmp1 / inputHeight;
        const int h = indtmp1 - indtmp2 * inputHeight;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1 / inputChannels;
        const int c = indtmp1 - indtmp2 * inputChannels;
        const int n = indtmp2;

        acc_t value(0);

        for(int multiplier = 0; multiplier < depthwiseMultiplier; ++multiplier)
        {
            int och = (c * depthwiseMultiplier) + multiplier;
            int weightOffset = och * kernelHeight * kernelWidth;
            for(int kh = 0; kh < KH_LIMIT; ++kh)
            {
#pragma unroll
                for(int kw = 0; kw < KW_LIMIT; ++kw)
                {
                    int h_out = h + padHeight - kh * dilationHeight;
                    int w_out = w + padWidth - kw * dilationWidth;
                    if((h_out % strideH == 0) && (w_out % strideW == 0))
                    {
                        h_out = h_out / strideH;
                        w_out = w_out / strideW;

                        if((h_out >= 0) && (h_out < outputHeight) && (w_out >= 0) && (w_out < outputWidth))
                        {

                            const int offset =
                                ((n * outputChannels + och) * outputHeight + h_out) * outputWidth + w_out;
                            value +=
                                (static_cast<acc_t>(weight[weightOffset]) * static_cast<acc_t>(grad_output[offset]));
                        }
                    }
                    ++weightOffset;
                }
            }
        }
        grad_input[linearIndex] = static_cast<scalar_t>(value);
    }
}

// 来源：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:282-358。
template <typename scalar_t, typename index_t = unsigned>
__global__ void conv_depthwise2d_grad_weight_kernel(const scalar_t *grad_output, const scalar_t *input,
                                                    scalar_t *grad_weight, const int batchSize, const int inputChannels,
                                                    const int kernelChannels, const int depthwiseMultiplier,
                                                    const int inputWidth, const int inputHeight, const int outputWidth,
                                                    const int outputHeight, const int kernelWidth,
                                                    const int kernelHeight, const int strideWidth,
                                                    const int strideHeight, const int padWidth, const int padHeight,
                                                    const int dilationWidth, const int dilationHeight)
{
    using acc_t = AccType<scalar_t>;
    const int channelStride = kernelWidth * kernelHeight;

    // Each Block is responsible for accumulating over a permutation of
    // (channels x kH x kW), use blockIdx to determine which one
    int bidx = blockIdx.x;
    int kW = bidx % kernelWidth;
    int kH = (bidx / kernelWidth) % kernelHeight;
    int ch = (bidx / channelStride);

    // Need to calculate which input channel is associated with this filter
    // channel
    int inputCh = ch / depthwiseMultiplier;

    acc_t grad(0);

    const int laneId = threadIdx.x % C10_WARP_SIZE;
    const int batch = threadIdx.x / C10_WARP_SIZE;
    const int nwarps = blockDim.x / C10_WARP_SIZE;
    const int imageElements = outputWidth * outputHeight;
    // Use warp per item.  In the original kernel, a threadblock was used to sum over NHW.
    // Here, we use a warp to sum values over HW dimension, and if batchSize is larger than the
    // number of warps, a warp would loop over remaining batch items (e.g. if there are 8 warps,
    // warp 0 would go over 0-8-16 etc image, warp 1 over 1-9-17 etc). Later in blockReduce,
    // all the warps will be reduced anyway, thus the full reduction will be over NHW, like it
    // should be. That allows to get rid of one modulo operation inside the loop (because n/batchIdx
    // now does not have to be computed through modulo, you are just looping over it), and
    // bring a nice speed-up.
    for(int batchIdx = batch; batchIdx < batchSize; batchIdx += nwarps)
    {
        // Warp-stride loop over elements in a batch item
        for(index_t idx = laneId; idx < imageElements; idx += C10_WARP_SIZE)
        {
            // Need to calculate the following: batch position, and offset into the grad_output
            // in height, and width. We can intuit the corresponding position in the input from
            // the other parameters we have
            int go_w_offset = idx % outputWidth;
            int go_h_offset = (idx / outputWidth);

            int i_w_offset = (go_w_offset * strideWidth) + (kW * dilationWidth) - padWidth;
            int i_h_offset = (go_h_offset * strideHeight) + (kH * dilationHeight) - padHeight;

            if(i_w_offset >= 0 && i_h_offset >= 0 && i_w_offset < inputWidth && i_h_offset < inputHeight)
            {
                int inputOffset =
                    ((batchIdx * inputChannels + inputCh) * inputHeight + i_h_offset) * inputWidth + i_w_offset;
                int outputOffset = ((batchIdx * kernelChannels + ch) * outputHeight) * outputWidth + idx;
                grad += (static_cast<acc_t>(input[inputOffset]) * static_cast<acc_t>(grad_output[outputOffset]));
            }
        }
    }

    // At this point each thread in the block has a local gradient, which we need to
    // accumulate prior to writing the global value
    extern __shared__ char smem[];
    acc_t *buf = reinterpret_cast<acc_t *>(smem);
    acc_t tval = cuda_utils::BlockReduceSum(grad, buf);

    // After reduction, first thread in the block has the gradient, so its responsible
    // for writing it to grad_weight
    if(threadIdx.x == 0)
    {
        int weightOffset = kW + (kernelWidth * kH) + (kernelWidth * kernelHeight * ch);
        grad_weight[weightOffset] = static_cast<scalar_t>(tval);
    }
}

#undef CUDA_KERNEL_LOOP_TYPE

// 来源：aten/src/ATen/cuda/detail/KernelUtils.h:26-35；断言适配为标准异常。
inline int GET_BLOCKS(const int64_t N, const int64_t max_threads_per_block = CUDA_NUM_THREADS)
{
    if(N <= 0)
        throw std::invalid_argument("PyTorch DWConv 的元素数必须为正");
    constexpr int64_t max_int = std::numeric_limits<int>::max();
    auto block_num = (N - 1) / max_threads_per_block + 1;
    if(block_num > max_int)
        throw std::invalid_argument("PyTorch DWConv 的 block 数超出 int 范围");
    return static_cast<int>(block_num);
}

// 本地元数据适配；对应上游 32 位索引检查：
// aten/src/ATen/native/cuda/DepthwiseConv2d.cu:407、494-495、622-623。
int32_t checked_numel(int batch_size, int batch_numel)
{
    const int64_t count = int64_t(batch_size) * batch_numel;
    if(batch_size <= 0 || batch_numel <= 0 || count > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("PyTorch DWConv 要求正 batch、正元素数和 32 位索引");
    return static_cast<int32_t>(count);
}

// 本地错误检查适配；替代上游 C10_CUDA_KERNEL_LAUNCH_CHECK 的调用，例如：
// aten/src/ATen/native/cuda/DepthwiseConv2d.cu:430。
// 宏定义来源：c10/cuda/CUDAException.h:85。
void check_launch()
{
    cuda_lab::check_cuda(cudaGetLastError(), "PyTorch DWConv kernel launch");
}

// 对应上游 getGradParamsNumThreads，最多八个 warp，一个 warp 负责一个 batch。
// 来源：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:577-583。
// 主机端 warp_size() 来源：aten/src/ATen/cuda/CUDAContext.cpp:38-40，此处适配为 NVIDIA CUDA 的 32。
int getGradParamsNumThreads(int batchSize)
{
    // warp per item in a batch, up to a maximum
    constexpr int MAX_BLOCK_SIZE = 256;
    return std::min(batchSize * C10_WARP_SIZE, MAX_BLOCK_SIZE);
}

// 本地参数/stream 适配；原始前向启动函数：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:360-451。
template <typename scalar_t>
void launch_forward(const scalar_t *input, const scalar_t *weight, const scalar_t *bias, scalar_t *output,
                    const Conv2DParam &p, int batch_size, cudaStream_t stream)
{
    const int kW = p.Kw, kH = p.Kh;
    const int dW = p.Sw, dH = p.Sh;
    const int padW = p.Pw, padH = p.Ph;
    // 本地接口限定 dilation=1、通道倍增=1；上游原函数从 Tensor/参数中获取。
    const int dilationW = 1, dilationH = 1;
    const int width = p.in_w, height = p.in_h;
    const int outputWidth = p.out_w, outputHeight = p.out_h;
    const int outputChannels = p.out_ch, depthwiseMultiplier = 1;
    const bool has_bias = bias != nullptr;
    checked_numel(batch_size, p.inBatchNumel);
    const int32_t n = checked_numel(batch_size, p.outBatchNumel);
    // grid/block 来源：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:406-411。
    dim3 grid(GET_BLOCKS(n));
    dim3 block(CUDA_NUM_THREADS);

    // 核尺寸分派来源：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:425-449。
    if(kW == 5 && kH == 5)
    {
        conv_depthwise2d_forward_kernel<5><<<grid, block, 0, stream>>>(
            input, output, weight, bias, has_bias, n, outputChannels, depthwiseMultiplier, width, height, outputWidth,
            outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
        check_launch();
    }
    else if(kW == 3 && kH == 3)
    {
        conv_depthwise2d_forward_kernel<3><<<grid, block, 0, stream>>>(
            input, output, weight, bias, has_bias, n, outputChannels, depthwiseMultiplier, width, height, outputWidth,
            outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
        check_launch();
    }
    else if(kW == 1 && kH == 1)
    {
        conv_depthwise2d_forward_kernel<1><<<grid, block, 0, stream>>>(
            input, output, weight, bias, has_bias, n, outputChannels, depthwiseMultiplier, width, height, outputWidth,
            outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
        check_launch();
    }
    else
    {
        conv_depthwise2d_forward_kernel_generic<<<grid, block, 0, stream>>>(
            input, output, weight, bias, has_bias, n, outputChannels, depthwiseMultiplier, width, height, outputWidth,
            outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
        check_launch();
    }
}

// 本地参数/stream 适配；原始 dX 启动函数：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:453-575。
template <typename scalar_t>
void launch_input_grad(const scalar_t *weight, const scalar_t *grad_output, scalar_t *grad_input, const Conv2DParam &p,
                       int batch_size, cudaStream_t stream)
{
    const int kW = p.Kw, kH = p.Kh;
    const int dW = p.Sw, dH = p.Sh;
    const int padW = p.Pw, padH = p.Ph;
    const int dilationW = 1, dilationH = 1;
    const int width = p.in_w, height = p.in_h;
    const int outputWidth = p.out_w, outputHeight = p.out_h;
    const int inputChannels = p.in_ch, outputChannels = p.out_ch, depthwiseMultiplier = 1;
    checked_numel(batch_size, p.outBatchNumel);
    const int32_t n = checked_numel(batch_size, p.inBatchNumel);
    // grid/block 来源：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:493-499。
    dim3 grid(GET_BLOCKS(n));
    dim3 block(CUDA_NUM_THREADS);

    // 核尺寸/stride 十二路分派来源：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:507-573。
    if(kW == 5 && kH == 5)
    {
        if(dW == 1 && dH == 1)
        {
            conv_depthwise2d_backward_kernel<5, 1><<<grid, block, 0, stream>>>(
                grad_output, grad_input, weight, n, inputChannels, depthwiseMultiplier, outputChannels, width, height,
                outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
            check_launch();
        }
        else if(dW == 2 && dH == 2)
        {
            conv_depthwise2d_backward_kernel<5, 2><<<grid, block, 0, stream>>>(
                grad_output, grad_input, weight, n, inputChannels, depthwiseMultiplier, outputChannels, width, height,
                outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
            check_launch();
        }
        else
        {
            conv_depthwise2d_backward_kernel<5, 0><<<grid, block, 0, stream>>>(
                grad_output, grad_input, weight, n, inputChannels, depthwiseMultiplier, outputChannels, width, height,
                outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
            check_launch();
        }
    }
    else if(kW == 3 && kH == 3)
    {
        if(dW == 1 && dH == 1)
        {
            conv_depthwise2d_backward_kernel<3, 1><<<grid, block, 0, stream>>>(
                grad_output, grad_input, weight, n, inputChannels, depthwiseMultiplier, outputChannels, width, height,
                outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
            check_launch();
        }
        else if(dW == 2 && dH == 2)
        {
            conv_depthwise2d_backward_kernel<3, 2><<<grid, block, 0, stream>>>(
                grad_output, grad_input, weight, n, inputChannels, depthwiseMultiplier, outputChannels, width, height,
                outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
            check_launch();
        }
        else
        {
            conv_depthwise2d_backward_kernel<3, 0><<<grid, block, 0, stream>>>(
                grad_output, grad_input, weight, n, inputChannels, depthwiseMultiplier, outputChannels, width, height,
                outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
            check_launch();
        }
    }
    else if(kW == 1 && kH == 1)
    {
        if(dW == 1 && dH == 1)
        {
            conv_depthwise2d_backward_kernel<1, 1><<<grid, block, 0, stream>>>(
                grad_output, grad_input, weight, n, inputChannels, depthwiseMultiplier, outputChannels, width, height,
                outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
            check_launch();
        }
        else if(dW == 2 && dH == 2)
        {
            conv_depthwise2d_backward_kernel<1, 2><<<grid, block, 0, stream>>>(
                grad_output, grad_input, weight, n, inputChannels, depthwiseMultiplier, outputChannels, width, height,
                outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
            check_launch();
        }
        else
        {
            conv_depthwise2d_backward_kernel<1, 0><<<grid, block, 0, stream>>>(
                grad_output, grad_input, weight, n, inputChannels, depthwiseMultiplier, outputChannels, width, height,
                outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
            check_launch();
        }
    }
    else if(dW == 1 && dH == 1)
    {
        conv_depthwise2d_backward_kernel<0, 1><<<grid, block, 0, stream>>>(
            grad_output, grad_input, weight, n, inputChannels, depthwiseMultiplier, outputChannels, width, height,
            outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
        check_launch();
    }
    else if(dW == 2 && dH == 2)
    {
        conv_depthwise2d_backward_kernel<0, 2><<<grid, block, 0, stream>>>(
            grad_output, grad_input, weight, n, inputChannels, depthwiseMultiplier, outputChannels, width, height,
            outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
        check_launch();
    }
    else
    {
        conv_depthwise2d_backward_kernel<0, 0><<<grid, block, 0, stream>>>(
            grad_output, grad_input, weight, n, inputChannels, depthwiseMultiplier, outputChannels, width, height,
            outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
        check_launch();
    }
}

// 本地参数/stream 适配；原始 dWeight 启动函数：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:585-646。
template <typename scalar_t>
void launch_weight_grad(const scalar_t *input, const scalar_t *grad_output, scalar_t *grad_weight, const Conv2DParam &p,
                        int batch_size, cudaStream_t stream)
{
    const int kW = p.Kw, kH = p.Kh;
    const int dW = p.Sw, dH = p.Sh;
    const int padW = p.Pw, padH = p.Ph;
    const int dilationW = 1, dilationH = 1;
    const int width = p.in_w, height = p.in_h;
    const int outputWidth = p.out_w, outputHeight = p.out_h;
    const int inputChannels = p.in_ch, outputChannels = p.out_ch, depthwiseMultiplier = 1;
    // 对应上游 canUse32BitIndexMath(input / grad_output) 的元素数约束。
    checked_numel(batch_size, p.inBatchNumel);
    checked_numel(batch_size, p.outBatchNumel);
    // grid/block 来源：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:621-629。
    int blocks = outputChannels * kH * kW;
    dim3 grid(blocks);
    dim3 block(getGradParamsNumThreads(batch_size));
    using acc_t = AccType<scalar_t>;
    // 动态共享内存与启动参数来源：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:637-644。
    int warp_size = C10_WARP_SIZE;
    int smem = (block.x / warp_size) * sizeof(acc_t);
    conv_depthwise2d_grad_weight_kernel<<<grid, block, smem, stream>>>(
        grad_output, input, grad_weight, batch_size, inputChannels, outputChannels, depthwiseMultiplier, width, height,
        outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    check_launch();
}

// 本地兼容实现：上游 DepthwiseConv2d.cu 只计算 dX / dWeight，框架另用 sum 计算 dBias。
// 此 kernel 不作为上游原生 depthwise kernel，也不与 dWeight 融合。
// 框架 dBias 归约语义参照：aten/src/ATen/native/Convolution.cpp:2380-2384。
template <typename scalar_t>
__global__ void bias_grad_compat_kernel(const scalar_t *grad_output, scalar_t *grad_bias, Conv2DParam p, int n)
{
    const int c = blockIdx.x;
    const int lane = threadIdx.x % C10_WARP_SIZE;
    const int warp = threadIdx.x / C10_WARP_SIZE;
    float sum = 0;
    for(int b = warp; b < n; b += blockDim.x / C10_WARP_SIZE)
        for(unsigned pos = lane; pos < p.outHW; pos += C10_WARP_SIZE)
            sum += static_cast<float>(grad_output[(b * p.in_ch + c) * p.outHW + pos]);
    __shared__ float partial[256 / C10_WARP_SIZE];
    sum = cuda_utils::BlockReduceSum(sum, partial);
    if(threadIdx.x == 0)
        grad_bias[c] = static_cast<scalar_t>(sum);
}

// 本地适配；上游反向组织与可选输出：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:698-730。
template <typename scalar_t>
void launch_backward(const scalar_t *input, const scalar_t *weight, const scalar_t *grad_output, scalar_t *grad_input,
                     scalar_t *grad_weight, scalar_t *grad_bias, const Conv2DParam &p, int n, cudaStream_t stream)
{
    // 沿用上游 backward_cuda_out 的顺序：dWeight 在前，dX 在后。
    // 顺序来源：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:710-727。
    if(grad_weight)
        launch_weight_grad(input, grad_output, grad_weight, p, n, stream);
    if(grad_input)
        launch_input_grad(weight, grad_output, grad_input, p, n, stream);
    if(grad_bias)
    {
        checked_numel(n, p.outBatchNumel);
        bias_grad_compat_kernel<<<p.in_ch, 256, 0, stream>>>(grad_output, grad_bias, p, n);
        check_launch();
    }
}

} // namespace

// 本地 FP32 外部接口；上游前向类型分派：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:414-415。
void launch_pytorch_forward_fp32(const float *input, const float *weight, const float *bias, float *output,
                                 const Conv2DParam &p, int n, cudaStream_t stream)
{
    launch_forward(input, weight, bias, output, p, n, stream);
}

// 本地 FP32 外部接口；上游 dX/dWeight 类型分派：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:501-502、632-633。
void launch_pytorch_backward_fp32(const float *input, const float *weight, const float *grad_output, float *grad_input,
                                  float *grad_weight, float *grad_bias, const Conv2DParam &p, int n,
                                  cudaStream_t stream)
{
    launch_backward(input, weight, grad_output, grad_input, grad_weight, grad_bias, p, n, stream);
}

// 本地 FP16 外部接口；上游前向类型分派：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:414-415。
void launch_pytorch_forward_fp16(const __half *input, const __half *weight, const __half *bias, __half *output,
                                 const Conv2DParam &p, int n, cudaStream_t stream)
{
    launch_forward(input, weight, bias, output, p, n, stream);
}

// 本地 FP16 外部接口；上游 dX/dWeight 类型分派：aten/src/ATen/native/cuda/DepthwiseConv2d.cu:501-502、632-633。
void launch_pytorch_backward_fp16(const __half *input, const __half *weight, const __half *grad_output,
                                  __half *grad_input, __half *grad_weight, __half *grad_bias, const Conv2DParam &p,
                                  int n, cudaStream_t stream)
{
    launch_backward(input, weight, grad_output, grad_input, grad_weight, grad_bias, p, n, stream);
}
