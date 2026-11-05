#pragma once


#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include "common.h"

// DWConv FP32 baseline, NCHW，groups = in_ch = out_ch，通道数为 4 的倍数，dilation = 1
// weight_stride=0 时按 KhKw 自动选择 32 / 64 / 128；指定容量时要求 KhKw <= weight_stride
// param.use_bias=false 时不读取 bias，bias 可传 nullptr；true 时 bias 必须非空
void launch_dwconv_fp32(float *inputs, float *weights, float *bias, float *outputs, const Conv2DParam &param, int n,
                        cudaStream_t stream = nullptr, int weight_stride = 0);

// FP32 input 寄存器预取版本
// 权重一次加载到 shared，输入使用两套寄存器片段交替预取
void launch_dwconv_fp32_prefetch(float *inputs, float *weights, float *bias, float *outputs, const Conv2DParam &param,
                                 int n, cudaStream_t stream = nullptr, int weight_stride = 0);

// FP32 weight L2 测试
// 无 shared weight
// use_l2_access_policy=false 使用默认缓存策略；true 在内部配置仅覆盖 weights 的优先保留窗口
void launch_dwconv_fp32_weights_l2(float *inputs, float *weights, float *bias, float *outputs, const Conv2DParam &param,
                                   int n, cudaStream_t stream = nullptr, bool use_l2_access_policy = false);
