#pragma once
#include "dwconv.cuh"
#include <cuda_fp16.h>

// FP16 baseline 分派：KhKw <= 32 / 64 / 128 时分别使用 k32 / k64 / k128。
void launch_baseline_fp16(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                          const Conv2DParam &param, int n, cudaStream_t stream = nullptr);

// k128 输入寄存器预取：权重一次加载，A/B 输入片段交替预取；约束与 k128 baseline 相同。
void launch_prefetch_fp16_k128(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                               const Conv2DParam &param, int n, cudaStream_t stream = nullptr);

// FP16 NCHW depthwise 卷积，half2 FMA 累加；C 为 8 的倍数，dilation = 1。
// packed_weights[pair * KhKw + k] = half2(weights[2*pair, k], weights[2*pair+1, k])。
// 三个入口分别要求 1 <= KhKw <= 128 / 64 / 32，派生尺寸由调用者正确填写。
void launch_baseline_fp16_k128(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                               const Conv2DParam &param, int n, cudaStream_t stream = nullptr);

void launch_baseline_fp16_k64(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                              const Conv2DParam &param, int n, cudaStream_t stream = nullptr);

void launch_baseline_fp16_k32(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                              const Conv2DParam &param, int n, cudaStream_t stream = nullptr);
