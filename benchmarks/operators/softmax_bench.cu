#include <softmax.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr int BENCHMARK_ROWS = 512;
constexpr int BENCHMARK_COLS = 1024;
constexpr int WARMUP_ITERATIONS = 20;
constexpr int BENCHMARK_ITERATIONS = 200;

// 算法级普通浮点运算估算，排除 exp、比较、整数运算与冗余归约。
// 计数推导见 docs/operators/softmax/arithmetic-intensity.md。
void print_fp32_metrics(int64_t rows, int64_t cols, float average_ms)
{
    // FLOPs / (ms * 1e6) = GFLOP/s；与 AI 使用相同的算法运算量口径。
    const double rate_scale = rows / (static_cast<double>(average_ms) * 1.0e6);
    if(cols <= cuda_lab::kSoftmaxBlockMaxColumns)
    {
        std::cout << " AI_estimate=" << 3.0 / (2 * sizeof(float)) << " FLOP/Byte"
                  << " GFLOP/s_estimate=" << 3.0 * cols * rate_scale;
    }
    else
    {
        const double bytes = 3.0 * cols * sizeof(float);
        const double min_flops = 4.0 * cols + 3.0 * cuda_lab::kSoftmaxBlockThreads;
        const double max_flops = min_flops + cols;
        std::cout << " AI_estimate=[" << min_flops / bytes << ',' << max_flops / bytes << "] FLOP/Byte"
                  << " GFLOP/s_estimate=[" << min_flops * rate_scale << ',' << max_flops * rate_scale << ']';
    }
    std::cout << "（普通浮点运算估算，不含 exp）\n";
}

bool check_cuda(cudaError_t status, const char *operation)
{
    if(status == cudaSuccess)
    {
        return true;
    }

    std::cout << "[CUDA ERROR] " << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

void fill_input(std::vector<float> &input, uint32_t seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    for(float &value : input)
    {
        value = distribution(generator);
    }
}

void fill_int8_input(std::vector<int8_t> &input, uint32_t seed)
{
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int32_t> distribution(-128, 127);
    for(int8_t &value : input)
    {
        value = static_cast<int8_t>(distribution(generator));
    }
}

bool run_fp32_benchmark(int64_t rows, int64_t cols)
{
    const size_t element_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t byte_count = element_count * sizeof(float);

    std::vector<float> source(element_count);
    fill_input(source, 2026U);

    float *device_source = nullptr;
    float *device_destination = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const char *kernel = cols <= 1024 ? "warp" : (cols <= cuda_lab::kSoftmaxBlockMaxColumns ? "block" : "online");
    std::cout << "[FP32 性能阶段] kernel=" << kernel << " rows=" << rows << " cols=" << cols
              << " warmup=" << WARMUP_ITERATIONS << " iterations=" << BENCHMARK_ITERATIONS << '\n';

    bool success =
        check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_source), byte_count), "cudaMalloc(benchmark source)") &&
        check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_destination), byte_count),
                   "cudaMalloc(benchmark destination)") &&
        check_cuda(cudaMemcpy(device_source, source.data(), byte_count, cudaMemcpyHostToDevice),
                   "cudaMemcpy(benchmark source)") &&
        check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)") &&
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    if(!success)
    {
        if(start != nullptr)
        {
            cudaEventDestroy(start);
        }
        if(stop != nullptr)
        {
            cudaEventDestroy(stop);
        }
        if(device_source != nullptr)
        {
            cudaFree(device_source);
        }
        if(device_destination != nullptr)
        {
            cudaFree(device_destination);
        }
        return false;
    }

    for(int iteration = 0; iteration < WARMUP_ITERATIONS; iteration++)
    {
        success = success && check_cuda(cuda_lab::launch_softmax(device_source, device_destination, rows, cols),
                                        "cuda_lab::launch_softmax(warmup)");
    }
    success = success && check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)") &&
              check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");

    for(int iteration = 0; iteration < BENCHMARK_ITERATIONS && success; iteration++)
    {
        success = check_cuda(cuda_lab::launch_softmax(device_source, device_destination, rows, cols),
                             "cuda_lab::launch_softmax(benchmark)");
    }

    success = success && check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)") &&
              check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    success = success && check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_source);
    cudaFree(device_destination);

    if(!success)
    {
        return false;
    }

    const float average_ms = elapsed_ms / BENCHMARK_ITERATIONS;
    std::cout << "  [关键结果] latency=" << std::fixed << std::setprecision(6) << average_ms << " ms";
    print_fp32_metrics(rows, cols, average_ms);
    std::cout << std::defaultfloat << std::setprecision(6) << "  [SUCCESS] 性能测试完成\n\n";
    return true;
}

bool run_int8_benchmark(int64_t rows)
{
    const int64_t cols = BENCHMARK_COLS;
    constexpr float INPUT_SCALE = 1.0f / 32.0f;
    constexpr int32_t INPUT_ZERO_POINT = -3;

    const size_t element_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t int8_byte_count = element_count * sizeof(int8_t);
    const size_t float_byte_count = element_count * sizeof(float);

    std::vector<int8_t> source(element_count);
    fill_int8_input(source, 2027U);

    int8_t *device_source = nullptr;
    float *device_float_destination = nullptr;
    int8_t *device_int8_destination = nullptr;

    std::cout << "[INT8 性能阶段] rows=" << rows << " cols=" << cols << " input_scale=" << INPUT_SCALE
              << " input_zero_point=" << INPUT_ZERO_POINT << " warmup=" << WARMUP_ITERATIONS
              << " iterations=" << BENCHMARK_ITERATIONS << '\n';

    bool success = check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_source), int8_byte_count),
                              "cudaMalloc(int8 benchmark source)") &&
                   check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_float_destination), float_byte_count),
                              "cudaMalloc(float benchmark destination)") &&
                   check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_int8_destination), int8_byte_count),
                              "cudaMalloc(int8 benchmark destination)") &&
                   check_cuda(cudaMemcpy(device_source, source.data(), int8_byte_count, cudaMemcpyHostToDevice),
                              "cudaMemcpy(int8 benchmark source)");

    if(!success)
    {
        if(device_source != nullptr)
        {
            cudaFree(device_source);
        }
        if(device_float_destination != nullptr)
        {
            cudaFree(device_float_destination);
        }
        if(device_int8_destination != nullptr)
        {
            cudaFree(device_int8_destination);
        }
        return false;
    }

    auto measure_latency = [](const char *name, auto launch, float &average_ms)
    {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        bool measured = check_cuda(cudaEventCreate(&start), "cudaEventCreate(int8 start)") &&
                        check_cuda(cudaEventCreate(&stop), "cudaEventCreate(int8 stop)");

        if(!measured)
        {
            if(start != nullptr)
            {
                cudaEventDestroy(start);
            }
            if(stop != nullptr)
            {
                cudaEventDestroy(stop);
            }
            return false;
        }

        for(int iteration = 0; iteration < WARMUP_ITERATIONS && measured; iteration++)
        {
            measured = check_cuda(launch(), name);
        }
        measured = measured && check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(int8 warmup)") &&
                   check_cuda(cudaEventRecord(start), "cudaEventRecord(int8 start)");

        for(int iteration = 0; iteration < BENCHMARK_ITERATIONS && measured; iteration++)
        {
            measured = check_cuda(launch(), name);
        }

        measured = measured && check_cuda(cudaEventRecord(stop), "cudaEventRecord(int8 stop)") &&
                   check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(int8 stop)");

        float elapsed_ms = 0.0f;
        measured = measured && check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime(int8)");

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        average_ms = elapsed_ms / BENCHMARK_ITERATIONS;
        return measured;
    };

    float int8_to_float_ms = 0.0f;
    float int8_to_int8_ms = 0.0f;
    success = measure_latency(
                  "cuda_lab::launch_softmax_int8_to_float(benchmark)",
                  [&]()
                  {
                      return cuda_lab::launch_softmax_int8_to_float(device_source, device_float_destination, rows, cols,
                                                                    INPUT_SCALE, INPUT_ZERO_POINT);
                  },
                  int8_to_float_ms) &&
              measure_latency(
                  "cuda_lab::launch_softmax_int8_to_int8(benchmark)",
                  [&]()
                  {
                      return cuda_lab::launch_softmax_int8_to_int8(device_source, device_int8_destination, rows, cols,
                                                                   INPUT_SCALE, INPUT_ZERO_POINT);
                  },
                  int8_to_int8_ms);

    cudaFree(device_source);
    cudaFree(device_float_destination);
    cudaFree(device_int8_destination);

    if(!success)
    {
        return false;
    }

    const double int8_to_float_ai = 3.0 / (sizeof(int8_t) + sizeof(float));
    const double int8_to_int8_ai = 4.0 / (2 * sizeof(int8_t));
    const double int8_to_float_gflops = 3.0 * element_count / (int8_to_float_ms * 1.0e6);
    const double int8_to_int8_gflops = 4.0 * element_count / (int8_to_int8_ms * 1.0e6);

    std::cout << "  [关键结果] INT8->FP32 latency=" << std::fixed << std::setprecision(6) << int8_to_float_ms
              << " ms AI_estimate=" << int8_to_float_ai << " FLOP/Byte GFLOP/s_estimate=" << int8_to_float_gflops
              << "（普通浮点运算估算，不含 exp）\n"
              << "  [关键结果] INT8->INT8 latency=" << std::setprecision(6) << int8_to_int8_ms
              << " ms AI_estimate=" << int8_to_int8_ai << " FLOP/Byte GFLOP/s_estimate=" << int8_to_int8_gflops
              << "（普通浮点运算估算，不含 exp）\n"
              << std::defaultfloat << "  [SUCCESS] INT8 性能测试完成\n\n";
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    int64_t selected_rows = BENCHMARK_ROWS;
    int64_t selected_cols = 0;
    try
    {
        bool rows_set = false;
        bool cols_set = false;
        for(int index = 1; index < argc; index += 2)
        {
            const std::string option = argv[index];
            if(index + 1 >= argc || (option != "--rows" && option != "--cols"))
            {
                throw std::invalid_argument("用法：softmax_bench [--rows 正整数] [--cols 正整数]");
            }
            bool &seen = option == "--rows" ? rows_set : cols_set;
            if(seen)
            {
                throw std::invalid_argument("参数不能重复");
            }
            seen = true;
            const std::string value = argv[index + 1];
            size_t consumed = 0;
            const int64_t parsed = std::stoll(value, &consumed);
            if(consumed != value.size() || parsed <= 0)
            {
                throw std::invalid_argument("行数和列数必须为正整数");
            }
            (option == "--rows" ? selected_rows : selected_cols) = parsed;
        }
        const int64_t max_cols = selected_cols > 0 ? selected_cols : 65536;
        if(selected_rows > std::numeric_limits<int>::max() ||
           static_cast<uint64_t>(max_cols) >
               std::min<uint64_t>(std::numeric_limits<size_t>::max(), std::numeric_limits<int64_t>::max()) /
                   selected_rows / sizeof(float))
        {
            throw std::invalid_argument("矩阵尺寸超出基准支持范围");
        }
    }
    catch(const std::exception &error)
    {
        std::cout << "[FAILED] " << error.what() << '\n';
        return 1;
    }
    int device_id = 0;
    cudaDeviceProp properties{};
    if(!check_cuda(cudaGetDevice(&device_id), "cudaGetDevice") ||
       !check_cuda(cudaGetDeviceProperties(&properties, device_id), "cudaGetDeviceProperties"))
    {
        return 1;
    }
    std::cout << "[配置] device=" << properties.name << " SM " << properties.major << '.' << properties.minor
              << " SM_count=" << properties.multiProcessorCount << " CUDA runtime=" << CUDART_VERSION
              << " rows=" << selected_rows << " int8_cols=" << BENCHMARK_COLS << " warmup=" << WARMUP_ITERATIONS
              << " iterations=" << BENCHMARK_ITERATIONS << '\n'
              << "[阶段] CUDA event 计时 kernel；不含分配、主机设备传输和预热\n\n";
    bool success = true;
    if(selected_cols > 0)
    {
        success = run_fp32_benchmark(selected_rows, selected_cols);
    }
    else
    {
        for(const int64_t cols : {1024, 1025, 2048, 4096, 8192, 8193, 16384, 65536})
        {
            success = run_fp32_benchmark(selected_rows, cols) && success;
        }
        success = run_int8_benchmark(selected_rows) && success;
    }
    std::cout << (success ? "[SUCCESS]" : "[FAILED]") << " softmax_bench\n";
    return success ? 0 : 1;
}
