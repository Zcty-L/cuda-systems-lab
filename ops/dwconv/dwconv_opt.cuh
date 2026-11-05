#pragma once
#include "common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// DWConv opt FP32，连续 NCHW，groups = in_ch = out_ch，dilation = 1，1 <= KhKw <= 128，C % 4 == 0。
// 按原 opt 策略选择尺寸特化、单通道或共享输入 kernel。
// param.use_bias=false 时不读取 bias，bias 可传 nullptr；true 时 bias 必须非空。
void launch_dwconv_opt_fp32(float *inputs, float *weights, float *bias, float *outputs, const Conv2DParam &param, int n,
                            cudaStream_t stream = nullptr);

// DWConv opt FP16，连续 NCHW，groups = in_ch = out_ch，dilation = 1，1 <= KhKw <= 128，C % 8 == 0。
// packed_weights 布局与 FP16 baseline 相同；accumulate_fp32=true（默认）使用 FP32 FMA，写回时舍入到 FP16。
// accumulate_fp32=false 使用 half2 FMA，每个 tap 的累加结果舍入到 FP16。
// param.use_bias=false 时不读取 bias，bias 可传 nullptr；true 时 bias 必须非空。
void launch_dwconv_opt_fp16(const __half *inputs, const __half2 *packed_weights, const __half *bias, __half *outputs,
                            const Conv2DParam &param, int n, cudaStream_t stream = nullptr,
                            bool accumulate_fp32 = true);
