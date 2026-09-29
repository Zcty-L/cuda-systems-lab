#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cuda_lab
{
// 显式对照入口，不改变 launch_softmax 的自动分派。
enum class SoftmaxVariant
{
    Online,
    Block512,
    Block1024,
    Cluster2,
    Cluster4
};

struct SoftmaxVariantInfo
{
    int block_threads = 0;
    int cluster_blocks = 1;
    int registers_per_thread = 0;
    size_t local_bytes_per_thread = 0;
    // 整个设备的驻留容量：普通 kernel 为 blocks，cluster kernel 为 clusters。
    int active_units = 0;
};

// 在当前设备查询；cluster 需要构建支持及设备支持，否则返回 cudaErrorNotSupported。
cudaError_t query_softmax_variant(SoftmaxVariant variant, int64_t cols, SoftmaxVariantInfo &info);

// FP32 连续、不重叠设备缓冲区；有限输入；rows、cols > 0。
// Block512/Cluster2 最多 16384 列，Block1024/Cluster4 最多 32768 列。
// grid.x = rows * cluster_blocks，必须 <= INT_MAX。支持调用者指定 stream。
cudaError_t launch_softmax_variant(const float *source, float *destination, int64_t rows, int64_t cols,
                                   SoftmaxVariant variant, cudaStream_t stream = nullptr);
} // namespace cuda_lab
