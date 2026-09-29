#include <cuda_runtime.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

#include "cuda_lab/cuda_check.cuh"

namespace
{
void section(const char *title)
{
    std::cout << "\n--- " << title << " ---\n";
}

template <typename T> void value(const char *key, const T &data)
{
    std::cout << "  " << std::left << std::setw(40) << key << data << '\n';
}

void bytes(const char *key, size_t count)
{
    const char *units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double scaled = static_cast<double>(count);
    int unit = 0;
    while(scaled >= 1024.0 && unit < 4)
    {
        scaled /= 1024.0;
        ++unit;
    }
    std::cout << "  " << std::left << std::setw(40) << key << scaled << ' ' << units[unit] << " (" << count << " B)\n";
}

int clock_rate(cudaDeviceAttr attribute, int device, const char *key)
{
    int rate = 0;
    const auto status = cudaDeviceGetAttribute(&rate, attribute, device);
    if(status != cudaSuccess)
    {
        value(key, std::string("N/A: ") + cudaGetErrorString(status));
        return 0;
    }
    value(key, rate / 1000.0);
    return rate;
}

void dimensions(const char *key, const int data[3])
{
    value(key, std::to_string(data[0]) + ", " + std::to_string(data[1]) + ", " + std::to_string(data[2]));
}

void occupancy(const cudaDeviceProp &p)
{
    section("驻留线程上限参考（忽略寄存器和共享内存约束）");
    std::cout << "  Block threads | Blocks/SM | Warps/SM | Threads/SM\n";
    for(int threads = 64; threads <= p.maxThreadsPerBlock; threads += 64)
    {
        const int warps_per_block = (threads + p.warpSize - 1) / p.warpSize;
        const int blocks =
            std::min(p.maxBlocksPerMultiProcessor, (p.maxThreadsPerMultiProcessor / p.warpSize) / warps_per_block);
        std::cout << "  " << std::right << std::setw(13) << threads << " | " << std::setw(9) << blocks << " | "
                  << std::setw(8) << blocks * warps_per_block << " | " << std::setw(10) << blocks * threads << '\n';
    }
}

void device_info(int device)
{
    cudaDeviceProp p{};
    cuda_lab::check_cuda(cudaGetDeviceProperties(&p, device), "cudaGetDeviceProperties");
    section("设备标识");
    value("Device", device);
    value("Name", p.name);
    value("Compute capability", std::to_string(p.major) + "." + std::to_string(p.minor));
    value("SM architecture", "sm_" + std::to_string(p.major) + std::to_string(p.minor));
    value("Multiprocessors (SM)", p.multiProcessorCount);
    value("Warp size", p.warpSize);

    section("存储层次");
    bytes("Global memory", p.totalGlobalMem);
    bytes("L2 cache", p.l2CacheSize);
    bytes("Shared memory / block", p.sharedMemPerBlock);
    bytes("Shared memory / block (opt-in)", p.sharedMemPerBlockOptin);
    bytes("Shared memory / SM", p.sharedMemPerMultiprocessor);
    bytes("Constant memory", p.totalConstMem);
    value("Registers / block", p.regsPerBlock);
    value("Registers / SM", p.regsPerMultiprocessor);

    section("时钟与理论显存带宽（非实测）");
    clock_rate(cudaDevAttrClockRate, device, "GPU clock (MHz)");
    const int memory_clock = clock_rate(cudaDevAttrMemoryClockRate, device, "Memory clock (MHz)");
    value("Memory bus width (bit)", p.memoryBusWidth);
    if(memory_clock > 0 && p.memoryBusWidth > 0)
    {
        value("Estimated DDR bandwidth (GB/s)", 2.0 * memory_clock * 1000.0 * p.memoryBusWidth / 8.0 / 1e9);
    }
    else
    {
        value("Estimated DDR bandwidth (GB/s)", "N/A");
    }

    section("执行与驻留限制");
    value("Max threads / block", p.maxThreadsPerBlock);
    value("Max threads / SM", p.maxThreadsPerMultiProcessor);
    value("Max blocks / SM", p.maxBlocksPerMultiProcessor);
    value("Max warps / SM", p.maxThreadsPerMultiProcessor / p.warpSize);
    dimensions("Max block dimensions", p.maxThreadsDim);
    dimensions("Max grid dimensions", p.maxGridSize);
    value("Async engines", p.asyncEngineCount);
    occupancy(p);

    section("运行时报告的设备特性（1=是，0=否）");
    value("Unified addressing", p.unifiedAddressing);
    value("Concurrent kernels", p.concurrentKernels);
    value("Managed memory", p.managedMemory);
    value("Concurrent managed access", p.concurrentManagedAccess);
    value("Pageable memory access", p.pageableMemoryAccess);
    value("ECC enabled", p.ECCEnabled);
    value("Can map host memory", p.canMapHostMemory);
    value("Cooperative launch", p.cooperativeLaunch);
}
} // namespace

int main()
{
    try
    {
        std::cout << std::fixed << std::setprecision(2);
        section("配置：查询所有 CUDA 可见设备");
        int count = 0;
        cuda_lab::check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        value("Device count", count);
        if(count == 0)
        {
            std::cout << "\n[SKIP] 未发现 CUDA 可见设备。\n";
            return 0;
        }
        int driver = 0;
        int runtime = 0;
        cuda_lab::check_cuda(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
        cuda_lab::check_cuda(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
        value("CUDA driver API version", std::to_string(driver / 1000) + "." + std::to_string(driver % 1000 / 10));
        value("CUDA runtime version", std::to_string(runtime / 1000) + "." + std::to_string(runtime % 1000 / 10));
        for(int device = 0; device < count; ++device)
        {
            device_info(device);
        }
        std::cout << "\n[SUCCESS] 设备查询完成。\n";
        return 0;
    }
    catch(const std::exception &error)
    {
        std::cout << "\n[ERROR] " << error.what() << '\n';
        return 1;
    }
}
