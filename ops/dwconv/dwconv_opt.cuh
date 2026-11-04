#pragma once
#include "dwconv.cuh"
#include <cuda_fp16.h>

// FP32 与 FP16 优化实现的公共接口，保留导入时的分派策略，异步启动；FP32 支持指定流，FP16 使用默认流。
// 连续 NCHW depthwise，groups = in_ch = out_ch，dilation = 1，1 <= KhKw <= 128。
// FP32 要求 C % 4 == 0；FP16 要求 C % 8 == 0，权重配对布局与 FP16 baseline 相同。
// 调用方保证参数及派生尺寸有效，并检查启动错误与执行完成状态。
void launch_opt_fp32(float *inputs, float *weights, float *bias, float *outputs, Conv2DParam param, int n,
                     cudaStream_t stream = nullptr);

void launch_opt_fp16(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                     const Conv2DParam &param, int n);
