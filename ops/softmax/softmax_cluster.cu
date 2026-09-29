#include "softmax_experimental.cuh"

#include <cooperative_groups.h>
#include <math_constants.h>

namespace
{
// 共享内存仅交换归约结果；寄存器保存局部指数。
// DSM 生命周期与同步依据：
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#distributed-shared-memory
template <bool MAXIMUM> __device__ float block_reduce(float value, float *scratch)
{
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for(int offset = 16; offset > 0; offset /= 2)
    {
        const float other = __shfl_down_sync(0xffffffffU, value, offset);
        value = MAXIMUM ? fmaxf(value, other) : value + other;
    }
    if(lane == 0)
        scratch[warp] = value;
    __syncthreads();
    if(warp == 0)
    {
        value = lane < 8 ? scratch[lane] : (MAXIMUM ? -CUDART_INF_F : 0.0f);
        for(int offset = 16; offset > 0; offset /= 2)
        {
            const float other = __shfl_down_sync(0xffffffffU, value, offset);
            value = MAXIMUM ? fmaxf(value, other) : value + other;
        }
        if(lane == 0)
            scratch[0] = value;
    }
    __syncthreads();
    const float result = scratch[0];
    __syncthreads(); // 允许下一次归约复用 scratch。
    return result;
}

template <int BLOCKS, int ITEMS>
__global__ void softmax_cluster_kernel(const float *__restrict__ source, float *__restrict__ destination, int64_t cols)
{
    namespace cg = cooperative_groups;
    const cg::cluster_group cluster = cg::this_cluster();
    const int rank = cluster.block_rank();
    const int64_t row = blockIdx.x / BLOCKS;
    const int64_t segment = (cols + BLOCKS - 1) / BLOCKS;
    const int64_t base = rank * segment;
    __shared__ float scratch[8];
    __shared__ float stats[2];
    __shared__ float normalization;

    float values[ITEMS];
    float maximum = -CUDART_INF_F;
#pragma unroll
    for(int i = 0; i < ITEMS; ++i)
    {
        const int offset = threadIdx.x + i * 256;
        const bool valid = offset < segment && base + offset < cols;
        values[i] = valid ? source[row * cols + base + offset] : -CUDART_INF_F;
        maximum = fmaxf(maximum, values[i]);
    }
    maximum = block_reduce<true>(maximum, scratch);
    float sum = 0.0f;
#pragma unroll
    for(int i = 0; i < ITEMS; ++i)
    {
        const int offset = threadIdx.x + i * 256;
        const bool valid = offset < segment && base + offset < cols;
        values[i] = valid ? expf(values[i] - maximum) : 0.0f;
        sum += values[i];
    }
    sum = block_reduce<false>(sum, scratch);
    if(threadIdx.x == 0)
    {
        stats[0] = maximum;
        stats[1] = sum;
    }
    cluster.sync(); // 所有 blocks 已启动，局部统计量对 DSM 读取可见。
    if(threadIdx.x == 0)
    {
        float maxima[BLOCKS], sums[BLOCKS];
        float global_max = -CUDART_INF_F;
#pragma unroll
        for(int b = 0; b < BLOCKS; ++b)
        {
            const float *remote = cluster.map_shared_rank(stats, b);
            maxima[b] = remote[0];
            sums[b] = remote[1];
            global_max = fmaxf(global_max, maxima[b]);
        }
        float global_sum = 0.0f;
#pragma unroll
        for(int b = 0; b < BLOCKS; ++b)
            global_sum += sums[b] * expf(maxima[b] - global_max);
        normalization = expf(maximum - global_max) / global_sum;
    }
    // 同时广播本 block 的系数，保证所有远端读取完成后才允许 block 退出。
    cluster.sync();
    const float scale = normalization;
#pragma unroll
    for(int i = 0; i < ITEMS; ++i)
    {
        const int offset = threadIdx.x + i * 256;
        if(offset < segment && base + offset < cols)
            destination[row * cols + base + offset] = values[i] * scale;
    }
}

const void *select_kernel(int blocks, int64_t cols)
{
    if(blocks == 2)
        return cols <= 8192 ? reinterpret_cast<const void *>(softmax_cluster_kernel<2, 16>)
                            : reinterpret_cast<const void *>(softmax_cluster_kernel<2, 32>);
    return cols <= 16384 ? reinterpret_cast<const void *>(softmax_cluster_kernel<4, 16>)
                         : reinterpret_cast<const void *>(softmax_cluster_kernel<4, 32>);
}

cudaLaunchConfig_t make_config(int blocks, int64_t rows, cudaStream_t stream, cudaLaunchAttribute &attribute)
{
    attribute = {};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim = {static_cast<unsigned int>(blocks), 1, 1};
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(static_cast<unsigned int>(rows * blocks));
    config.blockDim = dim3(256);
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    return config;
}
} // namespace

namespace cuda_lab::detail
{
cudaError_t query_softmax_cluster(int blocks, int64_t cols, SoftmaxVariantInfo &info)
{
    int device = 0, supported = 0;
    cudaError_t status = cudaGetDevice(&device);
    if(status != cudaSuccess)
        return status;
    status = cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, device);
    if(status != cudaSuccess)
        return status;
    if(!supported)
        return cudaErrorNotSupported;
    const void *kernel = select_kernel(blocks, cols);
    cudaFuncAttributes attributes{};
    status = cudaFuncGetAttributes(&attributes, kernel);
    if(status != cudaSuccess)
        return status;
    info.block_threads = 256;
    info.cluster_blocks = blocks;
    info.registers_per_thread = attributes.numRegs;
    info.local_bytes_per_thread = attributes.localSizeBytes;
    cudaLaunchAttribute attribute{};
    auto config = make_config(blocks, 1, nullptr, attribute);
    int max_cluster = 0;
    status = cudaOccupancyMaxPotentialClusterSize(&max_cluster, kernel, &config);
    if(status != cudaSuccess)
        return status;
    if(max_cluster < blocks)
        return cudaErrorNotSupported;
    status = cudaOccupancyMaxActiveClusters(&info.active_units, kernel, &config);
    if(status != cudaSuccess)
        return status;
    return info.active_units > 0 ? cudaSuccess : cudaErrorLaunchOutOfResources;
}

cudaError_t launch_softmax_cluster(const float *source, float *destination, int64_t rows, int64_t cols, int blocks,
                                   cudaStream_t stream)
{
    cudaLaunchAttribute attribute{};
    const auto config = make_config(blocks, rows, stream, attribute);
    void *arguments[] = {&source, &destination, &cols};
    return cudaLaunchKernelExC(&config, select_kernel(blocks, cols), arguments);
}
} // namespace cuda_lab::detail
