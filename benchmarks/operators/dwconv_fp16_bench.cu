#include <dwconv_fp16.cuh>
#include <cuda_lab/cuda_check.cuh>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

using cuda_lab::check_cuda;
constexpr int iterations = 100;
constexpr int rounds = 10;
using Launch = void (*)(const __half *, const __half2 *, const __half *, __half *, const Conv2DParam &, int,
                        cudaStream_t);
const Launch launches[] = {launch_baseline_fp16_k128, launch_prefetch_fp16_k128};
const char *names[] = {"baseline", "prefetch"};

template <typename T> struct Buffer
{
    T *ptr = nullptr;
    explicit Buffer(size_t count)
    {
        check_cuda(cudaMalloc(&ptr, count * sizeof(T)), "分配");
    }
    ~Buffer()
    {
        cudaFree(ptr);
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    void upload(const std::vector<T> &data)
    {
        check_cuda(cudaMemcpy(ptr, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice), "上传");
    }
};

void run(int n, int c, int h, int k, int stride, const cudaDeviceProp &device)
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
    std::vector<__half> input(n * p.inBatchNumel), bias(c);
    std::vector<__half2> weights(c / 2 * p.KhKw);
    for(size_t i = 0; i < input.size(); ++i)
        input[i] = __float2half_rn((int(i % 251) - 125) / 509.0f);
    for(int i = 0; i < c; ++i)
        bias[i] = __float2half_rn((i % 17 - 8) / 97.0f);
    for(size_t i = 0; i < weights.size(); ++i)
        weights[i] = __floats2half2_rn((int(i % 127) - 63) / 257.0f, (int((i * 13) % 127) - 63) / 257.0f);
    const size_t count = n * p.outBatchNumel;
    Buffer<__half> di(input.size()), db(bias.size()), output(count);
    Buffer<__half2> dw(weights.size());
    di.upload(input);
    db.upload(bias);
    dw.upload(weights);
    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "创建流");
    auto launch = [&](int v)
    {
        launches[v](di.ptr, dw.ptr, db.ptr, output.ptr, p, n, stream);
        check_cuda(cudaGetLastError(), "启动");
    };
    std::vector<__half> reference(count), actual(count);
    for(int v = 0; v < 2; ++v)
    {
        check_cuda(cudaMemsetAsync(output.ptr, 0xff, count * sizeof(__half), stream), "初始化输出");
        launch(v);
        check_cuda(cudaStreamSynchronize(stream), "正确性同步");
        check_cuda(cudaMemcpy(actual.data(), output.ptr, count * sizeof(__half), cudaMemcpyDeviceToHost), "读取");
        if(v == 0)
            reference = actual;
        for(size_t i = 0; i < count; ++i)
            if(!std::isfinite(__half2float(actual[i])) || __half2float(actual[i]) != __half2float(reference[i]))
                throw std::runtime_error("FP16 版本间结果不一致");
    }
    const int blocks = (p.outHW + 255) / 256 * (c / 8) * n;
    std::cout << "\n[配置] N=" << n << " C=" << c << " H=W=" << h << " K=" << k << " S=" << stride
              << " grid=" << (p.outHW + 255) / 256 << ',' << c / 8 << ',' << n << " block=256 工作集MiB="
              << (input.size() + bias.size() + count + weights.size() * 2) * 2.0 / (1024 * 1024)
              << " 正确性[SUCCESS]\n";
    cudaGraph_t graphs[2];
    cudaGraphExec_t execs[2];
    for(int v = 0; v < 2; ++v)
    {
        check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "捕获");
        for(int i = 0; i < iterations; ++i)
            launch(v);
        check_cuda(cudaStreamEndCapture(stream, &graphs[v]), "结束捕获");
        cudaGraphNode_t node;
        size_t nodes = 1;
        check_cuda(cudaGraphGetRootNodes(graphs[v], &node, &nodes), "读取根节点");
        cudaKernelNodeParams params{};
        check_cuda(cudaGraphKernelNodeGetParams(node, &params), "读取 kernel");
        cudaFuncAttributes attr{};
        int active;
        check_cuda(cudaFuncGetAttributes(&attr, params.func), "资源");
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, params.func, 256, 0), "驻留量");
        std::cout << "[资源] " << names[v] << " registers=" << attr.numRegs << " local=" << attr.localSizeBytes
                  << " shared=" << attr.sharedSizeBytes << " blocks/SM=" << active
                  << " waves=" << double(blocks) / (device.multiProcessorCount * active) << '\n';
        check_cuda(cudaGraphInstantiate(&execs[v], graphs[v], 0), "实例化");
        for(int i = 0; i < 20; ++i)
            launch(v);
        check_cuda(cudaGraphLaunch(execs[v], stream), "图预热");
    }
    check_cuda(cudaStreamSynchronize(stream), "预热同步");
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "事件");
    check_cuda(cudaEventCreate(&stop), "事件");
    std::array<std::vector<float>, 2> times;
    std::cout << "\n[阶段] CUDA Graph 计时\n";
    for(int round = 0; round < rounds; ++round)
        for(int j = 0; j < 2; ++j)
        {
            const int v = (round + j) % 2;
            check_cuda(cudaEventRecord(start, stream), "开始计时");
            check_cuda(cudaGraphLaunch(execs[v], stream), "执行图");
            check_cuda(cudaEventRecord(stop, stream), "结束计时");
            check_cuda(cudaEventSynchronize(stop), "同步");
            float ms;
            check_cuda(cudaEventElapsedTime(&ms, start, stop), "耗时");
            times[v].push_back(ms * 1000 / iterations);
            std::cout << "sample," << n << ',' << c << ',' << h << ',' << k << ',' << stride << ',' << names[v] << ','
                      << round << ',' << times[v].back() << '\n';
        }
    for(int v = 0; v < 2; ++v)
    {
        auto sorted = times[v];
        std::sort(sorted.begin(), sorted.end());
        std::cout << "summary," << n << ',' << c << ',' << h << ',' << k << ',' << stride << ',' << names[v] << ','
                  << (sorted[4] + sorted[5]) / 2 << ',' << sorted.front() << ',' << sorted.back() << '\n';
        cudaGraphExecDestroy(execs[v]);
        cudaGraphDestroy(graphs[v]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}
int main()
{
    try
    {
        cudaDeviceProp device{};
        check_cuda(cudaGetDeviceProperties(&device, 0), "设备");
        std::cout
            << "[设备] " << device.name << " SM=" << device.major << '.' << device.minor
            << " SM数量=" << device.multiProcessorCount << " CUDA=" << CUDART_VERSION
            << "\n[方法] FP16 k128；20次直接预热+100节点图预热；10轮交替顺序，每轮100次；CUDA event计时图内核序列\n";
        for(int k : {3, 7, 11})
            run(1, 32, 28, k, 1, device);
        // 64x64 每组16 blocks：按baseline最多6 blocks/SM覆盖约5/10/20 waves。
        for(int wave : {5, 10, 20})
            run(1, ((device.multiProcessorCount * 6 * wave + 15) / 16) * 8, 64, 7, 1, device);
        for(int n : {8, 16, 32})
            for(int k : {3, 7, 11})
                run(n, 64, 128, k, 1, device);
        for(int k : {3, 7, 11})
            run(16, 128, 128, k, 2, device);
        std::cout << "\n[SUCCESS] dwconv_fp16_bench\n";
    }
    catch(const std::exception &e)
    {
        std::cout << "[失败] " << e.what() << '\n';
        return 1;
    }
}
