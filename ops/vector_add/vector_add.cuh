#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda_lab
{
// a、b、out 均为设备指针；out[i] = a[i] + b[i]。
void vector_add(const float *a, const float *b, float *out, std::size_t count, cudaStream_t stream = nullptr);
} // namespace cuda_lab
