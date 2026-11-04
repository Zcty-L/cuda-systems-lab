#pragma once
#include "dwconv.cuh"
#include <cuda_fp16.h>

// PyTorch 原生算法的 FP32/FP16 简化版：连续 NCHW，通道倍增=1，dilation=1。
// 任意正通道数；尺寸/派生字段一致且所有索引及坐标中间值不超过 INT_MAX。
// n、尺寸、stride 为正，padding 非负，输出尺寸遵循前向公式。
// 指针不重叠；异步启动，调用方检查 CUDA 错误及同步。输出覆盖写入，无需清零。
// bias 可为空；反向的 grad_input / grad_weight / grad_bias 可分别为空以跳过计算。
void launch_pytorch_forward_fp32(const float *input, const float *weight, const float *bias, float *output,
                                 const Conv2DParam &p, int n, cudaStream_t stream = nullptr);

void launch_pytorch_backward_fp32(const float *input, const float *weight, const float *grad_output, float *grad_input,
                                  float *grad_weight, float *grad_bias, const Conv2DParam &p, int n,
                                  cudaStream_t stream = nullptr);

// FP16 使用普通 [C, Kh, Kw] 权重，无需 half2 配对；所有乘加与归约使用 FP32。
void launch_pytorch_forward_fp16(const __half *input, const __half *weight, const __half *bias, __half *output,
                                 const Conv2DParam &p, int n, cudaStream_t stream = nullptr);

void launch_pytorch_backward_fp16(const __half *input, const __half *weight, const __half *grad_output,
                                  __half *grad_input, __half *grad_weight, __half *grad_bias, const Conv2DParam &p,
                                  int n, cudaStream_t stream = nullptr);
