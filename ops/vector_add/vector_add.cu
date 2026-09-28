#include "vector_add.cuh"

#include <cuda_lab/cuda_check.cuh>

namespace cuda_lab
{

namespace
{

__global__ void vector_add_kernel(const float *a, const float *b, float *out, std::size_t count)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index < count)
    {
        out[index] = a[index] + b[index];
    }
}

} // namespace

void vector_add(const float *a, const float *b, float *out, std::size_t count, cudaStream_t stream)
{
    if(count == 0)
    {
        return;
    }

    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    vector_add_kernel<<<blocks, threads, 0, stream>>>(a, b, out, count);
    check_cuda(cudaGetLastError(), "vector_add_kernel launch");
}

} // namespace cuda_lab
