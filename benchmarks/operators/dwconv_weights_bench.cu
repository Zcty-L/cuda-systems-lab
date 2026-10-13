#include <dwconv.cuh>
#include <cuda_lab/cuda_check.cuh>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>

using cuda_lab::check_cuda;
constexpr int iterations = 100;
constexpr int rounds = 12;
const char *names[] = {"baseline", "gmem", "gmem_l2"};

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
    std::vector<float> input(n * p.inBatchNumel), bias(c);
    std::vector<float> weights(c * p.KhKw);
    for(size_t i = 0; i < input.size(); ++i)
        input[i] = (int(i % 251) - 125) / 509.0f;
    for(int i = 0; i < c; ++i)
        bias[i] = (i % 17 - 8) / 97.0f;
    for(size_t i = 0; i < weights.size(); ++i)
        weights[i] = (int(i % 127) - 63) / 257.0f;
    const size_t count = n * p.outBatchNumel;
    Buffer<float> di(input.size()), db(bias.size()), output(count);
    Buffer<float> dw(weights.size());
    di.upload(input);
    db.upload(bias);
    dw.upload(weights);
    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "创建流");
    cudaStreamAttrValue policy{};
    policy.accessPolicyWindow.base_ptr = dw.ptr;
    policy.accessPolicyWindow.num_bytes = weights.size() * sizeof(float);
    policy.accessPolicyWindow.hitRatio = 1.0f;
    policy.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    policy.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    if(policy.accessPolicyWindow.num_bytes > device.accessPolicyMaxWindowSize)
        throw std::runtime_error("权重超过 access policy window 上限");
    auto launch = [&](int v)
    {
        dim3 grid((p.outHW + 255) / 256, c / 4, n);
        if(v == 0)
            conv2d_4x128x256_groups_kernel<<<grid, 256, 0, stream>>>(di.ptr, dw.ptr, db.ptr, output.ptr, p);
        if(v == 1)
            conv2d_4x128x256_groups_kernel_gmem<<<grid, 256, 0, stream>>>(di.ptr, dw.ptr, db.ptr, output.ptr, p);
        if(v == 2)
            conv2d_4x128x256_groups_kernel_gmem_l2<<<grid, 256, 0, stream>>>(di.ptr, dw.ptr, db.ptr, output.ptr, p);
        check_cuda(cudaGetLastError(), "启动");
    };
    std::vector<float> reference(count), actual(count);
    for(int v = 0; v < 3; ++v)
    {
        check_cuda(cudaMemsetAsync(output.ptr, 0xff, count * sizeof(float), stream), "初始化输出");
        {
            check_cuda(cudaStreamSynchronize(stream), "策略前同步");
            check_cuda(cudaCtxResetPersistingL2Cache(), "重置 L2");
            policy.accessPolicyWindow.num_bytes = v == 2 ? weights.size() * sizeof(float) : 0;
            check_cuda(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &policy), "流策略");
        }
        launch(v);
        check_cuda(cudaStreamSynchronize(stream), "正确性同步");
        check_cuda(cudaMemcpy(actual.data(), output.ptr, count * sizeof(float), cudaMemcpyDeviceToHost), "读取");
        if(v == 0)
        {
            reference = actual;
            // 小输入全量、大输入均匀抽样，独立核对 CPU 的逐 tap FMA。
            const size_t samples = count < 100000 ? count : 257;
            for(size_t sample = 0; sample < samples; ++sample)
            {
                const size_t index = sample * (count - 1) / (samples - 1);
                const int batch = index / p.outBatchNumel;
                const int channel = index / p.outHW % c;
                const int oh = index % p.outHW / p.out_w;
                const int ow = index % p.out_w;
                float expected = bias[channel];
                for(int kh = 0; kh < k; ++kh)
                    for(int kw = 0; kw < k; ++kw)
                    {
                        const int ih = oh * stride - p.Ph + kh;
                        const int iw = ow * stride - p.Pw + kw;
                        const float value = ih >= 0 && ih < h && iw >= 0 && iw < h
                                                ? input[batch * p.inBatchNumel + channel * p.inHW + ih * h + iw]
                                                : 0.0f;
                        expected = std::fma(weights[channel * p.KhKw + kh * k + kw], value, expected);
                    }
                if(!std::isfinite(actual[index]) || std::abs(actual[index] - expected) > 1e-5f)
                    throw std::runtime_error("CPU 参考校验失败");
            }
        }
        for(size_t i = 0; i < count; ++i)
            if(!std::isfinite(actual[i]) || actual[i] != reference[i])
                throw std::runtime_error("FP32 版本间结果不一致");
    }
    {
        policy.accessPolicyWindow.num_bytes = 0;
        check_cuda(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &policy), "清除流策略");
        check_cuda(cudaCtxResetPersistingL2Cache(), "重置 L2");
    }
    if(profile)
    {
        check_cuda(cudaStreamDestroy(stream), "销毁流");
        std::cout << "[SUCCESS] 三版正确性与 CPU 抽样校验\n";
        return;
    }
    const int blocks = (p.outHW + 255) / 256 * (c / 4) * n;
    std::cout << "\n[配置] N=" << n << " C=" << c << " H=W=" << h << " K=" << k << " S=" << stride
              << " grid=" << (p.outHW + 255) / 256 << ',' << c / 4 << ',' << n
              << " block=256 工作集MiB=" << (input.size() + bias.size() + count + weights.size()) * 4.0 / (1024 * 1024)
              << " 正确性[SUCCESS]\n";
    cudaGraph_t graphs[3];
    cudaGraphExec_t execs[3];
    for(int v = 0; v < 3; ++v)
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
        if(v == 2)
        {
            size_t count_nodes = 0;
            check_cuda(cudaGraphGetNodes(graphs[v], nullptr, &count_nodes), "节点数");
            std::vector<cudaGraphNode_t> nodes(count_nodes);
            check_cuda(cudaGraphGetNodes(graphs[v], nodes.data(), &count_nodes), "节点列表");
            cudaKernelNodeAttrValue attr_policy{};
            policy.accessPolicyWindow.num_bytes = weights.size() * sizeof(float);
            attr_policy.accessPolicyWindow = policy.accessPolicyWindow;
            for(auto node : nodes)
                check_cuda(
                    cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &attr_policy),
                    "图权重策略");
        }
        check_cuda(cudaGraphInstantiate(&execs[v], graphs[v], 0), "实例化");
        for(int i = 0; i < 20; ++i)
            launch(v);
        check_cuda(cudaGraphLaunch(execs[v], stream), "图预热");
    }
    check_cuda(cudaStreamSynchronize(stream), "预热同步");
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "事件");
    check_cuda(cudaEventCreate(&stop), "事件");
    std::array<std::vector<float>, 3> times;
    std::cout << "\n[阶段] CUDA Graph 计时\n";
    for(int round = 0; round < rounds; ++round)
        for(int j = 0; j < 3; ++j)
        {
            const int v = (round + j) % 3;
            {
                check_cuda(cudaCtxResetPersistingL2Cache(), "轮间重置 L2 属性");
                check_cuda(cudaGraphLaunch(execs[v], stream), "各轮图预热");
                check_cuda(cudaStreamSynchronize(stream), "各轮预热同步");
            }
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
    auto baseline_times = times[0];
    std::sort(baseline_times.begin(), baseline_times.end());
    const float baseline_us = (baseline_times[5] + baseline_times[6]) / 2;
    for(int v = 0; v < 3; ++v)
    {
        auto sorted = times[v];
        std::sort(sorted.begin(), sorted.end());
        std::cout << "summary," << n << ',' << c << ',' << h << ',' << k << ',' << stride << ',' << names[v] << ','
                  << (sorted[5] + sorted[6]) / 2 << ',' << sorted.front() << ',' << sorted.back() << '\n';
        if(v != 0)
            std::cout << "[对比 baseline] " << names[v]
                      << " 耗时变化=" << ((sorted[5] + sorted[6]) / 2 / baseline_us - 1) * 100 << "%（负值更快）\n";
        cudaGraphExecDestroy(execs[v]);
        cudaGraphDestroy(graphs[v]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    check_cuda(cudaCtxResetPersistingL2Cache(), "用例结束重置 L2");
}
int main(int argc, char **argv)
{
    try
    {
        cudaDeviceProp device{};
        check_cuda(cudaGetDeviceProperties(&device, 0), "设备");
        std::cout
            << "[设备] " << device.name << " SM=" << device.major << '.' << device.minor
            << " SM数量=" << device.multiProcessorCount << " CUDA=" << CUDART_VERSION
            << "\n[方法] FP32 k128；20次直接预热+100节点图预热；12轮交替顺序，每轮100次；CUDA event计时图内核序列\n";
        const std::string mode = argc > 1 ? argv[1] : "";
        if(!mode.empty() && mode != "--profile" && mode != "--k3" && mode != "--profile-k3")
            throw std::runtime_error("参数：无参数（完整扫描）、--k3、--profile、--profile-k3");
        {
            std::cout << "[L2] maxSetAside=" << device.persistingL2CacheMaxSize
                      << " maxWindow=" << device.accessPolicyMaxWindowSize << '\n';
            if(device.persistingL2CacheMaxSize < 1024 * 1024)
                throw std::runtime_error("设备不支持本实验的 1 MiB L2 保留空间");
            check_cuda(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 1024 * 1024), "设置 L2 保留空间");
            size_t actual_limit;
            check_cuda(cudaDeviceGetLimit(&actual_limit, cudaLimitPersistingL2CacheSize), "读取保留空间");
            std::cout << "[L2] actualSetAside=" << actual_limit << " hitRatio=1；仅 weights；每轮重置属性并预热100次\n";
        }
        if(mode == "--k3" || mode == "--profile-k3")
        {
            run(16, 64, 128, 3, 1, device, mode == "--profile-k3");
            return 0;
        }
        if(mode == "--profile")
        {
            run(16, 64, 128, 7, 1, device, true);
            return 0;
        }
        for(int k : {3, 7, 11})
            run(1, 32, 28, k, 1, device);
        // 64x64 每组16 blocks：按baseline最多6 blocks/SM覆盖约5/10/20 waves。
        for(int wave : {5, 10, 20})
            run(1, ((device.multiProcessorCount * 6 * wave + 15) / 16) * 4, 64, 7, 1, device);
        for(int n : {8, 16, 32})
            for(int k : {3, 7, 11})
                run(n, 64, 128, k, 1, device);
        for(int k : {3, 7, 11})
            run(16, 128, 128, k, 2, device);
        std::cout << "\n[SUCCESS] dwconv_weights_bench\n";
    }
    catch(const std::exception &e)
    {
        std::cout << "[失败] " << e.what() << '\n';
        return 1;
    }
}
