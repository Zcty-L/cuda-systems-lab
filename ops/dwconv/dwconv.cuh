#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

typedef struct
{
    int in_h;
    int in_w;
    int in_ch;
    int inHW;
    int inChKhKw;
    int inBatchNumel;
    int out_ch;
    int out_h;
    int out_w;
    int outHW;
    int outBatchNumel;
    int Kh;
    int Kw;
    int KhKw;
    int Sh;
    int Sw;
    int Ph;
    int Pw;

    int k_tiles;
    int first_k_tile;

    int kernelWeightsCount;
    int biasWeightsCount;
} Conv2DParam;

// FP32 baseline 分派：KhKw <= 32 / 64 / 128 时分别使用 k32 / k64 / k128。
void launch_baseline_fp32(float *inputs, float *weights, float *bias, float *outputs, const Conv2DParam &param, int n,
                          cudaStream_t stream = nullptr);

// FP32 NCHW depthwise 卷积；groups = in_ch = out_ch，通道数为 4 的倍数。
// Kh * Kw <= 128，dilation = 1；block = 256，grid = (ceil(outHW/256), out_ch/4, N)。
__global__ void conv2d_4x128x256_groups_kernel(float *inputs, float *weights, float *bias, float *outputs,
                                               Conv2DParam param);

// 权重一次加载到 shared，输入使用两套寄存器片段预取；调用约束同 k128 baseline。
__global__ void conv2d_4x128x256_groups_kernel_prefetch(float *inputs, float *weights, float *bias, float *outputs,
                                                        Conv2DParam param);

__global__ void conv2d_4x64x256_groups_kernel(float *inputs, float *weights, float *bias, float *outputs,
                                              Conv2DParam param);

__global__ void conv2d_4x32x256_groups_kernel(float *inputs, float *weights, float *bias, float *outputs,
                                              Conv2DParam param);

// 权重存储实验：直接读取只读全局权重，无 shared。
__global__ void conv2d_4x128x256_groups_kernel_gmem(float *inputs, float *weights, float *bias, float *outputs,
                                                    Conv2DParam param);

// 与 gmem 共用计算路径；启动方须在 stream 或 graph kernel node 上设置仅覆盖 weights 的
// accessPolicyWindow（hitProp=Persisting），并配置 cudaLimitPersistingL2CacheSize。
__global__ void conv2d_4x128x256_groups_kernel_gmem_l2(float *inputs, float *weights, float *bias, float *outputs,
                                                       Conv2DParam param);
