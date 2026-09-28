#include <cuda_lab/cuda_check.cuh>

#include <vector_add.cuh>

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

namespace
{
void run_case(std::size_t count)
{
    std::cout << "\n[配置] count=" << count << "\n";

    std::vector<float> a(count);
    std::vector<float> b(count);
    std::vector<float> result(count);
    for(std::size_t index = 0; index < count; ++index)
    {
        a[index] = static_cast<float>(index) * 0.25F;
        b[index] = static_cast<float>(index % 7) * 0.5F;
    }

    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_out = nullptr;
    const std::size_t bytes = count * sizeof(float);
    cuda_lab::check_cuda(cudaMalloc(&device_a, bytes), "cudaMalloc(a)");
    cuda_lab::check_cuda(cudaMalloc(&device_b, bytes), "cudaMalloc(b)");
    cuda_lab::check_cuda(cudaMalloc(&device_out, bytes), "cudaMalloc(out)");

    std::cout << "[阶段] 上传输入并执行 kernel\n";
    cuda_lab::check_cuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(a)");
    cuda_lab::check_cuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(b)");
    cuda_lab::vector_add(device_a, device_b, device_out, count);
    cuda_lab::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cuda_lab::check_cuda(cudaMemcpy(result.data(), device_out, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(out)");

    std::cout << "[阶段] 验证结果\n";
    for(std::size_t index = 0; index < count; ++index)
    {
        const float expected = a[index] + b[index];
        if(std::abs(result[index] - expected) > 1e-6F)
        {
            std::cout << "[失败] index=" << index << " expected=" << expected << " actual=" << result[index] << "\n";
            throw std::runtime_error("vector_add 结果不匹配");
        }
    }
    std::cout << "[结果] 已验证 " << count << " 个元素\n";

    cuda_lab::check_cuda(cudaFree(device_a), "cudaFree(a)");
    cuda_lab::check_cuda(cudaFree(device_b), "cudaFree(b)");
    cuda_lab::check_cuda(cudaFree(device_out), "cudaFree(out)");
}
} // namespace

int main()
{
    try
    {
        cudaDeviceProp properties{};
        cuda_lab::check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::cout << "[设备] " << properties.name << " SM " << properties.major << "." << properties.minor << "\n";

        run_case(1);
        run_case(1003);
        run_case(4096);

        std::cout << "\n[SUCCESS] vector_add_test\n";
        return 0;
    }
    catch(const std::exception &error)
    {
        std::cout << "\n[失败] " << error.what() << "\n";
        return 1;
    }
}
