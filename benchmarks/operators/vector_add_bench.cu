#include <cuda_lab/cuda_check.cuh>

#include <vector_add.cuh>

#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

int main()
{
    try
    {
        constexpr std::size_t count = 1 << 20;
        constexpr int warmup = 20;
        constexpr int iterations = 100;
        const std::size_t bytes = count * sizeof(float);

        cudaDeviceProp properties{};
        cuda_lab::check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::cout << "[设备] " << properties.name << " SM " << properties.major << "." << properties.minor << "\n";
        std::cout << "[配置] count=" << count << " warmup=" << warmup << " iterations=" << iterations << "\n";

        std::vector<float> input(count, 1.0F);
        float *device_a = nullptr;
        float *device_b = nullptr;
        float *device_out = nullptr;
        cuda_lab::check_cuda(cudaMalloc(&device_a, bytes), "cudaMalloc(a)");
        cuda_lab::check_cuda(cudaMalloc(&device_b, bytes), "cudaMalloc(b)");
        cuda_lab::check_cuda(cudaMalloc(&device_out, bytes), "cudaMalloc(out)");
        cuda_lab::check_cuda(cudaMemcpy(device_a, input.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(a)");
        cuda_lab::check_cuda(cudaMemcpy(device_b, input.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(b)");

        std::cout << "\n[阶段] 预热\n";
        for(int index = 0; index < warmup; ++index)
        {
            cuda_lab::vector_add(device_a, device_b, device_out, count);
        }
        cuda_lab::check_cuda(cudaDeviceSynchronize(), "warmup synchronize");

        cudaEvent_t start{};
        cudaEvent_t stop{};
        cuda_lab::check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
        cuda_lab::check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

        std::cout << "\n[阶段] 计时 kernel，不含内存分配和传输\n";
        cuda_lab::check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
        for(int index = 0; index < iterations; ++index)
        {
            cuda_lab::vector_add(device_a, device_b, device_out, count);
        }
        cuda_lab::check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
        cuda_lab::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");

        float total_ms = 0.0F;
        cuda_lab::check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "cudaEventElapsedTime");
        const double latency_us = 1000.0 * total_ms / iterations;
        const double bandwidth_gbs = 3.0 * static_cast<double>(bytes) * iterations / (total_ms * 1.0e6);
        std::cout << std::fixed << std::setprecision(3) << "[结果] latency=" << latency_us << " us"
                  << " effective_bandwidth=" << bandwidth_gbs << " GB/s\n";

        cuda_lab::check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
        cuda_lab::check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
        cuda_lab::check_cuda(cudaFree(device_a), "cudaFree(a)");
        cuda_lab::check_cuda(cudaFree(device_b), "cudaFree(b)");
        cuda_lab::check_cuda(cudaFree(device_out), "cudaFree(out)");

        std::cout << "\n[SUCCESS] vector_add_bench\n";
        return 0;
    }
    catch(const std::exception &error)
    {
        std::cout << "\n[失败] " << error.what() << "\n";
        return 1;
    }
}
