#include <softmax_experimental.cuh>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr int WARMUP = 20;
constexpr int ITERATIONS = 100;
constexpr int REPEATS = 5;
using Variant = cuda_lab::SoftmaxVariant;

void check(cudaError_t status)
{
    if(status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

struct Candidate
{
    Variant variant;
    const char *name;
    cuda_lab::SoftmaxVariantInfo info{};
};

std::vector<Candidate> candidates(int64_t cols, const std::string &selected)
{
    std::vector<Candidate> result;
    for(auto candidate : {Candidate{Variant::Online, "online"}, Candidate{Variant::Block512, "block512"},
                          Candidate{Variant::Cluster2, "cluster2"}, Candidate{Variant::Cluster4, "cluster4"},
                          Candidate{Variant::Block1024, "block1024"}})
    {
        if(selected != "all" && selected != candidate.name)
            continue;
        if(cols > 16384 && (candidate.variant == Variant::Block512 || candidate.variant == Variant::Cluster2))
        {
            if(selected != "all")
                throw std::invalid_argument("所选实现最多支持 16384 列");
            continue;
        }
        const auto status = cuda_lab::query_softmax_variant(candidate.variant, cols, candidate.info);
        if(status == cudaErrorNotSupported)
        {
            std::cout << "[SKIP] " << candidate.name << " 当前设备或构建不支持\n";
            continue;
        }
        check(status);
        result.push_back(candidate);
    }
    if(result.empty())
        throw std::invalid_argument("没有可运行的实现");
    return result;
}

struct Resources
{
    float *input = nullptr;
    float *output = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    std::vector<cudaGraph_t> graphs;
    std::vector<cudaGraphExec_t> executions;
    ~Resources()
    {
        if(stream)
            cudaStreamSynchronize(stream);
        for(auto execution : executions)
            cudaGraphExecDestroy(execution);
        for(auto graph : graphs)
            cudaGraphDestroy(graph);
        if(start)
            cudaEventDestroy(start);
        if(stop)
            cudaEventDestroy(stop);
        if(stream)
            cudaStreamDestroy(stream);
        if(input)
            cudaFree(input);
        if(output)
            cudaFree(output);
    }
};

void measure(int64_t rows, int64_t cols, const std::vector<Candidate> &implementations, std::ostream &csv)
{
    if(rows > std::numeric_limits<int>::max() / 4 ||
       static_cast<uint64_t>(rows) > std::numeric_limits<size_t>::max() / cols / sizeof(float))
        throw std::invalid_argument("输入尺寸过大");
    const size_t elements = static_cast<size_t>(rows) * cols;
    const size_t bytes = elements * sizeof(float);
    Resources r;
    check(cudaMalloc(&r.input, bytes));
    check(cudaMalloc(&r.output, bytes));
    check(cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking));
    check(cudaEventCreate(&r.start));
    check(cudaEventCreate(&r.stop));
    std::vector<float> input(elements);
    std::mt19937 generator(2026);
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    for(float &value : input)
        value = distribution(generator);
    check(cudaMemcpy(r.input, input.data(), bytes, cudaMemcpyHostToDevice));
    input.clear();
    input.shrink_to_fit();

    for(const auto &candidate : implementations)
    {
        // 先启动一次完成惰性加载，再捕获；计时排除图构建与实例化。
        check(cuda_lab::launch_softmax_variant(r.input, r.output, rows, cols, candidate.variant, r.stream));
        check(cudaStreamSynchronize(r.stream));
        check(cudaStreamBeginCapture(r.stream, cudaStreamCaptureModeThreadLocal));
        for(int iteration = 0; iteration < ITERATIONS; ++iteration)
            check(cuda_lab::launch_softmax_variant(r.input, r.output, rows, cols, candidate.variant, r.stream));
        cudaGraph_t graph = nullptr;
        check(cudaStreamEndCapture(r.stream, &graph));
        r.graphs.push_back(graph);
        cudaGraphExec_t execution = nullptr;
        check(cudaGraphInstantiate(&execution, graph, 0));
        r.executions.push_back(execution);
    }

    std::cout << "\n[测量阶段] rows=" << rows << " cols=" << cols << " working_set_bytes=" << 2 * bytes << '\n';
    std::vector<std::vector<double>> samples(implementations.size());
    for(int repeat = 0; repeat < REPEATS; ++repeat)
    {
        // 每轮轮换实现顺序，降低固定顺序造成的温度和时钟偏差。
        for(size_t j = 0; j < implementations.size(); ++j)
        {
            const size_t index = (j + repeat) % implementations.size();
            const auto &candidate = implementations[index];
            for(int warmup = 0; warmup < WARMUP; ++warmup)
                check(cuda_lab::launch_softmax_variant(r.input, r.output, rows, cols, candidate.variant, r.stream));
            check(cudaStreamSynchronize(r.stream));
            check(cudaEventRecord(r.start, r.stream));
            check(cudaGraphLaunch(r.executions[index], r.stream));
            check(cudaEventRecord(r.stop, r.stream));
            check(cudaEventSynchronize(r.stop));
            float elapsed = 0;
            check(cudaEventElapsedTime(&elapsed, r.start, r.stop));
            const double us = elapsed * 1000.0 / ITERATIONS;
            samples[index].push_back(us);
            const auto &info = candidate.info;
            // 与原基准一致：普通浮点运算，不计 exp/max/整数指令。
            // Cluster 增加每个 block 的 K 次 (减、乘、加) 及自身 (减、除)。
            double flops_min = 3.0 * cols, flops_max = flops_min;
            double traffic = 8.0 * cols;
            if(candidate.variant == Variant::Online)
            {
                flops_min = 4.0 * cols + 3.0 * 256;
                flops_max = flops_min + cols;
                traffic = 12.0 * cols;
            }
            else if(info.cluster_blocks > 1)
            {
                flops_min += info.cluster_blocks * (3.0 * info.cluster_blocks + 2);
                flops_max = flops_min;
            }
            csv << std::setprecision(10) << candidate.name << ',' << rows << ',' << cols << ',' << repeat << ','
                << rows * info.cluster_blocks << ',' << info.block_threads << ',' << info.cluster_blocks << ','
                << info.active_units << ',' << static_cast<double>(rows) / info.active_units << ','
                << info.registers_per_thread << ',' << info.local_bytes_per_thread << ',' << 2 * bytes << ',' << us
                << ',' << flops_min / traffic << ',' << flops_max / traffic << ',' << rows * flops_min / (us * 1000.0)
                << ',' << rows * flops_max / (us * 1000.0) << '\n';
        }
    }
    for(size_t i = 0; i < implementations.size(); ++i)
    {
        auto &values = samples[i];
        std::sort(values.begin(), values.end());
        const auto &candidate = implementations[i];
        std::cout << "  [关键结果] " << candidate.name
                  << " waves=" << static_cast<double>(rows) / candidate.info.active_units
                  << " registers=" << candidate.info.registers_per_thread
                  << " local_bytes=" << candidate.info.local_bytes_per_thread << " latency_us[min,median,max]=["
                  << values.front() << ',' << values[REPEATS / 2] << ',' << values.back() << "]\n";
    }
    csv.flush();
    if(!csv)
        throw std::runtime_error("写入 CSV 失败");
    std::cout << "[SUCCESS] 当前形状测量完成\n";
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        int64_t selected_rows = 0, selected_cols = 0;
        std::string selected_variant = "all", output = "softmax-cluster.csv";
        std::set<std::string> seen;
        for(int i = 1; i < argc; i += 2)
        {
            const std::string option = argv[i];
            if(i + 1 == argc || !seen.insert(option).second)
                throw std::invalid_argument("参数缺值或重复");
            const std::string value = argv[i + 1];
            if(option == "--output")
                output = value;
            else if(option == "--variant")
                selected_variant = value;
            else if(option == "--rows" || option == "--cols")
            {
                size_t consumed = 0;
                const auto number = std::stoll(value, &consumed);
                if(consumed != value.size() || number <= 0)
                    throw std::invalid_argument("尺寸必须为正整数");
                (option == "--rows" ? selected_rows : selected_cols) = number;
            }
            else
                throw std::invalid_argument("用法：softmax_cluster_bench [--rows N] [--cols N] [--variant "
                                            "online|block512|block1024|cluster2|cluster4] [--output CSV]");
        }
        if(selected_cols > 32768)
            throw std::invalid_argument("本对照基准最多支持 32768 列");
        if(selected_variant != "all" && selected_variant != "online" && selected_variant != "block512" &&
           selected_variant != "block1024" && selected_variant != "cluster2" && selected_variant != "cluster4")
            throw std::invalid_argument("未知实现");
        int device = 0, driver = 0, runtime = 0;
        cudaDeviceProp properties{};
        check(cudaGetDevice(&device));
        check(cudaGetDeviceProperties(&properties, device));
        check(cudaDriverGetVersion(&driver));
        check(cudaRuntimeGetVersion(&runtime));
        std::cout
            << "[配置] device=" << properties.name << " SM=" << properties.major << '.' << properties.minor
            << " SM_count=" << properties.multiProcessorCount << " runtime=" << runtime << " driver=" << driver
            << " warmup=" << WARMUP << " iterations=" << ITERATIONS << " repeats=" << REPEATS << '\n'
            << "[计时] CUDA event 包围一个含 100 次 kernel 的 CUDA Graph，除以 100；复用缓冲区，热缓存，自适应时钟\n"
            << "[配置] CSV=" << output << '\n';
        std::ofstream csv(output);
        if(!csv)
            throw std::runtime_error("无法打开 CSV 文件");
        csv << "variant,rows,cols,repeat,grid_blocks,block_threads,cluster_blocks,active_units,waves,registers,local_"
               "bytes,working_set_bytes,latency_us,ai_min,ai_max,gflops_min,gflops_max\n";
        const std::vector<int64_t> widths =
            selected_cols ? std::vector<int64_t>{selected_cols} : std::vector<int64_t>{16384, 32768};
        for(const auto cols : widths)
        {
            const auto implementations = candidates(cols, selected_variant);
            std::set<int64_t> row_counts = {1, 8, 32, 512};
            // 取各实现的 5/10/20 waves 行数的并集，所有实现运行同一组形状。
            for(const auto &candidate : implementations)
                for(const int waves : {5, 10, 20})
                    row_counts.insert(static_cast<int64_t>(waves) * candidate.info.active_units);
            if(selected_rows)
                row_counts = {selected_rows};
            for(const auto rows : row_counts)
                measure(rows, cols, implementations, csv);
        }
        std::cout << "\n[SUCCESS] Softmax cluster 对照基准完成\n";
        return 0;
    }
    catch(const std::exception &error)
    {
        std::cout << "[FAILED] " << error.what() << '\n';
        return 1;
    }
}
