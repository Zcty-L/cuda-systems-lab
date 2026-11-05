#pragma once


#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "common.h"

// DWConv FP16 baseline，NCHW，groups = in_ch = out_ch，通道数为 8 的倍数，dilation = 1。
// 输入、bias、输出为 half；accumulate_fp32=true（默认）使用 FP32 FMA，写回时舍入到 FP16。
// accumulate_fp32=false 使用 half2 FMA，每个 tap 的累加结果舍入到 FP16。
// packed_weights[pair * KhKw + k] = half2(weights[2*pair, k], weights[2*pair+1, k])。
// param.use_bias=false 时不读取 bias，bias 可传 nullptr；true 时 bias 必须非空。
void launch_dwconv_fp16(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                        const Conv2DParam &param, int n, cudaStream_t stream = nullptr, int weight_stride = 0,
                        bool accumulate_fp32 = true);
