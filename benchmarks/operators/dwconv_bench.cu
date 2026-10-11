#include <dwconv.cuh>
#include <cuda_lab/cuda_check.cuh>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using cuda_lab::check_cuda;
using Kernel = void (*)(float *, float *, float *, float *, Conv2DParam);
constexpr int iterations = 100;
constexpr int rounds = 9;
constexpr int variants = 2;
Kernel kernels[] = {conv2d_4x128x256_groups_kernel, conv2d_4x128x256_groups_kernel_prefetch};
const char *names[] = {"baseline", "prefetch"};
struct Buffer
{
    float *ptr = nullptr;
    explicit Buffer(size_t count)
    {
        check_cuda(cudaMalloc(&ptr, count * sizeof(float)), "分配");
        std::vector<float> data(count);
        for(size_t i = 0; i < count; ++i)
            data[i] = (int(i % 29) - 14) / 32.0f;
        check_cuda(cudaMemcpy(ptr, data.data(), count * sizeof(float), cudaMemcpyHostToDevice), "上传");
    }
    ~Buffer()
    {
        cudaFree(ptr);
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
};
void run(int n, int c, int h, int k, int stride, const cudaDeviceProp &device, bool profile = false)
{
    Conv2DParam p{};
    p.in_h = p.in_w = h;
    p.in_ch = p.out_ch = c;
    p.Kh = p.Kw = k;
    p.KhKw = k * k;
    p.Sh = p.Sw = stride;
    p.Ph = p.Pw = k / 2;
    p.inHW = h * h;
    p.inBatchNumel = c * p.inHW;
    p.out_h = p.out_w = (h + 2 * p.Ph - k) / stride + 1;
    p.outHW = p.out_h * p.out_w;
    p.outBatchNumel = c * p.outHW;
    const size_t count = n * p.outBatchNumel;
    Buffer input(n * p.inBatchNumel), weights(c * p.KhKw), bias(c), output(count);
    dim3 grid((p.outHW + 255) / 256, c / 4, n);
    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "创建流");
    auto launch = [&](int v)
    {
        void *args[] = {&input.ptr, &weights.ptr, &bias.ptr, &output.ptr, &p};
        check_cuda(cudaLaunchKernel(reinterpret_cast<const void *>(kernels[v]), grid, dim3(256), args, 0, stream),
                   "启动");
    };
    std::vector<float> reference(count), actual(count);
    for(int v = 0; v < variants; ++v)
    {
        launch(v);
        check_cuda(cudaStreamSynchronize(stream), "正确性同步");
        check_cuda(cudaMemcpy(actual.data(), output.ptr, count * sizeof(float), cudaMemcpyDeviceToHost), "读取");
        if(v == 0)
            reference = actual;
        for(size_t i = 0; i < count; ++i)
            if(!std::isfinite(actual[i]) || std::abs(actual[i] - reference[i]) > 1e-5f * (1 + std::abs(reference[i])))
                throw std::runtime_error("版本间结果不一致");
    }
    std::cout << "\n[配置] N=" << n << " C=" << c << " H=W=" << h << " K=" << k << " S=" << stride << " grid=" << grid.x
              << "," << grid.y << "," << grid.z
              << " block=256 工作集MiB=" << (n * size_t(p.inBatchNumel) + count + c * p.KhKw + c) * 4.0 / (1024 * 1024)
              << " 正确性[SUCCESS]\n";
    cudaGraph_t graphs[variants];
    cudaGraphExec_t execs[variants];
    for(int v = 0; v < variants; ++v)
    {
        int active;
        cudaFuncAttributes attr{};
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernels[v], 256, 0), "驻留量");
        check_cuda(cudaFuncGetAttributes(&attr, kernels[v]), "资源");
        std::cout << "[资源] " << names[v] << " registers=" << attr.numRegs << " local=" << attr.localSizeBytes
                  << " shared=" << attr.sharedSizeBytes << " dynamic_shared=" << 0 << " blocks/SM=" << active
                  << " waves=" << double(grid.x * grid.y * grid.z) / (device.multiProcessorCount * active) << "\n";
        if(profile)
            continue;
        check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "捕获");
        for(int i = 0; i < iterations; ++i)
            launch(v);
        check_cuda(cudaStreamEndCapture(stream, &graphs[v]), "结束捕获");
        check_cuda(cudaGraphInstantiate(&execs[v], graphs[v], 0), "实例化");
        for(int i = 0; i < 20; ++i)
            launch(v);
        check_cuda(cudaGraphLaunch(execs[v], stream), "图预热");
    }
    if(profile)
    {
        cudaStreamDestroy(stream);
        return;
    }
    check_cuda(cudaStreamSynchronize(stream), "预热同步");
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "事件");
    check_cuda(cudaEventCreate(&stop), "事件");
    std::array<std::vector<float>, variants> times;
    const int measured_rounds = rounds;
    for(int round = 0; round < measured_rounds; ++round)
        for(int j = 0; j < variants; ++j)
        {
            const int v = (round + j) % variants;
            check_cuda(cudaEventRecord(start, stream), "开始计时");
            check_cuda(cudaGraphLaunch(execs[v], stream), "图执行");
            check_cuda(cudaEventRecord(stop, stream), "结束计时");
            check_cuda(cudaEventSynchronize(stop), "计时同步");
            float ms;
            check_cuda(cudaEventElapsedTime(&ms, start, stop), "耗时");
            const float us = ms * 1000 / iterations;
            times[v].push_back(us);
            std::cout << "sample," << n << ',' << c << ',' << h << ',' << k << ',' << stride << ',' << names[v] << ','
                      << round << ',' << us << '\n';
        }
    for(int v = 0; v < variants; ++v)
    {
        auto sorted = times[v];
        std::sort(sorted.begin(), sorted.end());
        std::cout << "summary," << n << ',' << c << ',' << h << ',' << k << ',' << stride << ',' << names[v] << ','
                  << (sorted[(measured_rounds - 1) / 2] + sorted[measured_rounds / 2]) / 2 << ',' << sorted.front()
                  << ',' << sorted.back() << '\n';
    }
    for(int v = 0; v < variants; ++v)
    {
        cudaGraphExecDestroy(execs[v]);
        cudaGraphDestroy(graphs[v]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}
int main(int argc, char **argv)
{
    try
    {
        const std::string mode = argc == 2 ? argv[1] : "";
        const bool large_batch = mode == "--large-batch";
        const bool profile = mode == "--profile";
        if(argc > 2 || (argc == 2 && !large_batch && !profile))
            throw std::runtime_error("用法：dwconv_bench [--large-batch|--profile]");
        cudaDeviceProp device{};
        check_cuda(cudaGetDeviceProperties(&device, 0), "设备");
        std::cout
            << "[设备] " << device.name << " SM=" << device.major << '.' << device.minor
            << " SM数量=" << device.multiProcessorCount << " CUDA=" << CUDART_VERSION
            << "\n[方法] 热缓存，20次直接预热+1次100节点图预热；9轮轮换顺序，每轮100次；CUDA event计时图内核序列\n";
        int resident;
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernels[0], 256, 0), "驻留量");
        if(profile)
        {
            std::cout << "[诊断] 两个版本，仅执行正确性单次启动\n";
            for(int k : {3, 7, 11})
                run(16, 64, 128, k, 1, device, true);
        }
        else if(large_batch)
        {
            for(int n : {8, 16, 32})
                for(int k : {3, 7, 11})
                    run(n, 64, 128, k, 1, device);
            for(int stride : {1, 2})
                for(int k : {3, 7})
                    run(16, 128, 128, k, stride, device);
        }
        else
            for(int k : {3, 5, 7, 11})
            {
                run(1, 32, 28, k, 1, device);
                for(int wave : {5, 10, 20})
                {
                    const int c = ((device.multiProcessorCount * resident * wave + 15) / 16) * 4;
                    run(1, c, 64, k, 1, device);
                }
                run(4, 128, 128, k, 2, device);
            }
        std::cout << "\n[SUCCESS] dwconv_bench\n";
    }
    catch(const std::exception &e)
    {
        std::cout << "[失败] " << e.what() << '\n';
        return 1;
    }
}
