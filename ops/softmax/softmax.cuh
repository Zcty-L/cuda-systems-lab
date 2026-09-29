#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace cuda_lab
{
inline constexpr int kSoftmaxWarpSize = 32;
inline constexpr int kSoftmaxBlockThreads = 256;
inline constexpr int kSoftmaxMaxColumnsPerThread = 32;
inline constexpr int kSoftmaxBlockMaxColumns = kSoftmaxBlockThreads * kSoftmaxMaxColumnsPerThread;
inline constexpr int kSoftmaxOnlineVectorSize = 4;
inline constexpr int kSoftmaxInt8MaxColumns = 1024;
inline constexpr float kSoftmaxInt8OutputScale = 1.0f / 256.0f;
inline constexpr int32_t kSoftmaxInt8OutputZeroPoint = -128;

// 行主序连续 FP32 矩阵，rows、cols 均为正数；输入有限，输入输出为不重叠设备缓冲区。
// cols <= 1024 使用 warp；1025..8192 使用 block；cols > 8192 使用 online。
// 后两种路径要求 rows <= INT_MAX；调用者负责分配内存和同步。
cudaError_t launch_softmax(const float *source, float *destination, int64_t rows, int64_t cols,
                           cudaStream_t stream = nullptr);

// INT8 支持 1 <= cols <= 1024。输入按 (q - input_zero_point) * input_scale 反量化。
cudaError_t launch_softmax_int8_to_float(const int8_t *source, float *destination, int64_t rows, int64_t cols,
                                         float input_scale, int32_t input_zero_point, cudaStream_t stream = nullptr);

// INT8 输出固定使用 scale=1/256、zero_point=-128，按最近偶数舍入并饱和。
cudaError_t launch_softmax_int8_to_int8(const int8_t *source, int8_t *destination, int64_t rows, int64_t cols,
                                        float input_scale, int32_t input_zero_point, cudaStream_t stream = nullptr);
} // namespace cuda_lab
