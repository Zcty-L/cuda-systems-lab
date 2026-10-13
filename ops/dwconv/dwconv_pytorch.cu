// 改编自 PyTorch aten/src/ATen/native/cuda/DepthwiseConv2d.cu。
// 来源提交与版权许可见 PYTORCH-LICENSE 和 README.md。
#include "dwconv_pytorch.cuh"

namespace
{

// 与上游 acc_type<Half, true> 一致：FP16 存储、FP32 乘加与归约，写回时舍入。
constexpr int threads = 256;

// 保留上游 generic 前向的有效核边界裁剪；固定 multiplier=1、dilation=1。
template <class T>
__global__ void forward_kernel(const T *input, const T *weight, const T *bias, T *output, Conv2DParam p, int count)
{
    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;)
    {
        int q = index % p.out_w;
        int y = index / p.out_w % p.out_h;
        int nc = index / p.outHW;
        int c = nc % p.in_ch;
        int ih = y * p.Sh - p.Ph;
        int iw = q * p.Sw - p.Pw;
        int r0 = max(0, -ih), r1 = min(p.Kh, p.in_h - ih);
        int s0 = max(0, -iw), s1 = min(p.Kw, p.in_w - iw);
        float sum = bias ? float(bias[c]) : 0.0f;
        for(int r = r0; r < r1; ++r)
            for(int s = s0; s < s1; ++s)
                sum +=
                    float(weight[c * p.KhKw + r * p.Kw + s]) * float(input[nc * p.inHW + (ih + r) * p.in_w + iw + s]);
        output[index] = sum;
        if(count - index <= int(blockDim.x * gridDim.x))
            break;
        index += blockDim.x * gridDim.x;
    }
}

// 上游 backward kernel 的通用 gather 路径，一个线程独占一个 dX。
template <class T> __global__ void input_grad_kernel(const T *weight, const T *dy, T *dx, Conv2DParam p, int count)
{
    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;)
    {
        int w = index % p.in_w, h = index / p.in_w % p.in_h;
        int nc = index / p.inHW, c = nc % p.in_ch;
        float sum = 0;
        for(int r = 0; r < p.Kh; ++r)
            for(int s = 0; s < p.Kw; ++s)
            {
                int y = h + p.Ph - r, x = w + p.Pw - s;
                if(y % p.Sh || x % p.Sw)
                    continue;
                y /= p.Sh;
                x /= p.Sw;
                if(y >= 0 && y < p.out_h && x >= 0 && x < p.out_w)
                    sum += float(weight[c * p.KhKw + r * p.Kw + s]) * float(dy[nc * p.outHW + y * p.out_w + x]);
            }
        dx[index] = sum;
        if(count - index <= int(blockDim.x * gridDim.x))
            break;
        index += blockDim.x * gridDim.x;
    }
}

__device__ float warp_sum(float value)
{
    for(int offset = 16; offset; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

// 保留上游 dK 的 warp 按 batch 分工方式；用简单 shuffle 归约替换 ATen 辅助函数。
// Bias=true 是本地补充的 db 路径，上游该文件未提供 db kernel。
template <bool Bias, class T>
__global__ void parameter_grad_kernel(const T *input, const T *dy, T *grad, Conv2DParam p, int n)
{
    int k = blockIdx.x;
    int c = Bias ? k : k / p.KhKw;
    int r = Bias ? 0 : k / p.Kw % p.Kh;
    int s = Bias ? 0 : k % p.Kw;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float sum = 0;
    for(int b = warp; b < n; b += threads / 32)
        for(int pos = lane; pos < p.outHW; pos += 32)
        {
            int out = (b * p.in_ch + c) * p.outHW + pos;
            if constexpr(Bias)
                sum += float(dy[out]);
            else
            {
                int ih = pos / p.out_w * p.Sh - p.Ph + r;
                int iw = pos % p.out_w * p.Sw - p.Pw + s;
                if(ih >= 0 && ih < p.in_h && iw >= 0 && iw < p.in_w)
                    sum += float(input[(b * p.in_ch + c) * p.inHW + ih * p.in_w + iw]) * float(dy[out]);
            }
        }
    __shared__ float partial[threads / 32];
    sum = warp_sum(sum);
    if(!lane)
        partial[warp] = sum;
    __syncthreads();
    if(!warp)
    {
        sum = warp_sum(lane < threads / 32 ? partial[lane] : 0.0f);
        if(!lane)
            grad[k] = sum;
    }
}

int blocks(int count)
{
    return min(65535, (count - 1) / threads + 1);
}

} // namespace

void launch_pytorch_forward_fp32(const float *input, const float *weight, const float *bias, float *output,
                                 const Conv2DParam &p, int n, cudaStream_t stream)
{
    int count = n * p.outBatchNumel;
    forward_kernel<<<blocks(count), threads, 0, stream>>>(input, weight, bias, output, p, count);
}

void launch_pytorch_backward_fp32(const float *input, const float *weight, const float *dy, float *dx, float *dk,
                                  float *db, const Conv2DParam &p, int n, cudaStream_t stream)
{
    if(dx)
    {
        int count = n * p.inBatchNumel;
        input_grad_kernel<<<blocks(count), threads, 0, stream>>>(weight, dy, dx, p, count);
    }
    if(dk)
        parameter_grad_kernel<false><<<p.in_ch * p.KhKw, threads, 0, stream>>>(input, dy, dk, p, n);
    if(db)
        parameter_grad_kernel<true><<<p.in_ch, threads, 0, stream>>>(input, dy, db, p, n);
}

void launch_pytorch_forward_fp16(const __half *input, const __half *weight, const __half *bias, __half *output,
                                 const Conv2DParam &p, int n, cudaStream_t stream)
{
    int count = n * p.outBatchNumel;
    forward_kernel<<<blocks(count), threads, 0, stream>>>(input, weight, bias, output, p, count);
}

void launch_pytorch_backward_fp16(const __half *input, const __half *weight, const __half *dy, __half *dx, __half *dk,
                                  __half *db, const Conv2DParam &p, int n, cudaStream_t stream)
{
    if(dx)
    {
        int count = n * p.inBatchNumel;
        input_grad_kernel<<<blocks(count), threads, 0, stream>>>(weight, dy, dx, p, count);
    }
    if(dk)
        parameter_grad_kernel<false><<<p.in_ch * p.KhKw, threads, 0, stream>>>(input, dy, dk, p, n);
    if(db)
        parameter_grad_kernel<true><<<p.in_ch, threads, 0, stream>>>(input, dy, db, p, n);
}
