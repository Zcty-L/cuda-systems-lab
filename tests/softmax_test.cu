#include <softmax.cuh>
#include <softmax_experimental.cuh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

namespace
{
constexpr int MAX_COLS = 1024;
constexpr int WARP_SIZE = 32;
constexpr float INT8_OUTPUT_SCALE = 1.0f / 256.0f;
constexpr int32_t INT8_OUTPUT_ZERO_POINT = -128;

bool check_cuda(cudaError_t status, const char *operation)
{
    if(status == cudaSuccess)
    {
        return true;
    }

    std::cout << "[CUDA ERROR] " << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

void softmax_cpu(const std::vector<float> &source, std::vector<float> &destination, int64_t rows, int64_t cols)
{
    for(int64_t row = 0; row < rows; row++)
    {
        const int64_t row_offset = row * cols;
        float max_value = -std::numeric_limits<float>::infinity();
        for(int64_t col = 0; col < cols; col++)
        {
            max_value = std::max(max_value, source[static_cast<size_t>(row_offset + col)]);
        }

        double sum_value = 0.0;
        for(int64_t col = 0; col < cols; col++)
        {
            sum_value += std::exp(static_cast<double>(source[static_cast<size_t>(row_offset + col)]) - max_value);
        }

        const double reciprocal_sum = 1.0 / sum_value;
        for(int64_t col = 0; col < cols; col++)
        {
            destination[static_cast<size_t>(row_offset + col)] =
                std::exp(static_cast<double>(source[static_cast<size_t>(row_offset + col)]) - max_value) *
                reciprocal_sum;
        }
    }
}

void dequantize_int8_cpu(const std::vector<int8_t> &source, std::vector<float> &destination, float input_scale,
                         int32_t input_zero_point)
{
    for(size_t index = 0; index < source.size(); index++)
    {
        destination[index] = (static_cast<int32_t>(source[index]) - input_zero_point) * input_scale;
    }
}

int8_t quantize_probability_int8(float probability)
{
    int32_t quantized = static_cast<int32_t>(std::nearbyint(probability / INT8_OUTPUT_SCALE));
    quantized += INT8_OUTPUT_ZERO_POINT;
    quantized = std::clamp<int32_t>(quantized, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    return static_cast<int8_t>(quantized);
}

void quantize_softmax_int8_cpu(const std::vector<float> &source, std::vector<int8_t> &destination)
{
    for(size_t index = 0; index < source.size(); index++)
    {
        destination[index] = quantize_probability_int8(source[index]);
    }
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

bool run_accuracy_case(int64_t rows, int64_t cols, uint32_t seed, int source_offset = 0, int destination_offset = 0,
                       int pattern = 0, const cuda_lab::SoftmaxVariant *variant = nullptr)
{
    const size_t element_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t byte_count = element_count * sizeof(float);

    std::vector<float> source(element_count);
    std::vector<float> reference(element_count);
    std::vector<float> actual(element_count);
    fill_input(source, seed);
    if(pattern == 1)
    {
        std::fill(source.begin(), source.end(), 1000.0f);
    }
    else if(pattern == 2)
    {
        for(size_t index = 0; index < element_count; ++index)
        {
            source[index] = index % 17 == 0 ? 1000.0f : -1000.0f;
        }
    }
    else if(pattern == 3)
    {
        // 不同分段的最大值悬殊，最大值位于最后一个分段。
        for(size_t index = 0; index < element_count; ++index)
            source[index] = (index % cols) < static_cast<size_t>(cols / 2) ? -1000.0f : 1000.0f;
    }
    else if(pattern == 4)
    {
        std::fill(source.begin(), source.end(), -1000.0f);
        for(int64_t row = 0; row < rows; ++row)
            source[(row + 1) * cols - 1] = 1000.0f;
    }
    softmax_cpu(source, reference, rows, cols);

    float *source_storage = nullptr;
    float *destination_storage = nullptr;

    std::cout << "[精度阶段] rows=" << rows << " cols=" << cols << " source_offset=" << source_offset
              << " destination_offset=" << destination_offset << " pattern=" << pattern << " stream=nonblocking\n";

    if(!check_cuda(cudaMalloc(&source_storage, byte_count + source_offset * sizeof(float)), "cudaMalloc(source)"))
    {
        return false;
    }
    if(!check_cuda(cudaMalloc(&destination_storage, byte_count + destination_offset * sizeof(float)),
                   "cudaMalloc(destination)"))
    {
        cudaFree(source_storage);
        return false;
    }

    float *device_source = source_storage + source_offset;
    float *device_destination = destination_storage + destination_offset;
    cudaStream_t stream = nullptr;
    if(!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"))
    {
        cudaFree(source_storage);
        cudaFree(destination_storage);
        return false;
    }

    bool success =
        check_cuda(cudaMemcpyAsync(device_source, source.data(), byte_count, cudaMemcpyHostToDevice, stream),
                   "cudaMemcpy(source)") &&
        check_cuda(
            variant ? cuda_lab::launch_softmax_variant(device_source, device_destination, rows, cols, *variant, stream)
                    : cuda_lab::launch_softmax(device_source, device_destination, rows, cols, stream),
            "launch_softmax") &&
        check_cuda(cudaMemcpyAsync(actual.data(), device_destination, byte_count, cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpy(destination)") &&
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(accuracy)");

    cudaStreamDestroy(stream);
    cudaFree(source_storage);
    cudaFree(destination_storage);

    if(!success)
    {
        return false;
    }

    constexpr float ABSOLUTE_TOLERANCE = 1.0e-6f;
    constexpr float RELATIVE_TOLERANCE = 1.0e-5f;
    constexpr float ROW_SUM_TOLERANCE = 2.0e-5f;

    size_t error_count = 0;
    size_t max_error_index = 0;
    float max_absolute_error = 0.0f;
    float max_relative_error = 0.0f;
    float max_row_sum_error = 0.0f;

    for(size_t index = 0; index < element_count; index++)
    {
        const float absolute_error = std::abs(reference[index] - actual[index]);
        const float relative_error = absolute_error / std::max(std::abs(reference[index]), ABSOLUTE_TOLERANCE);
        const float tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * std::abs(reference[index]);

        if(!std::isfinite(actual[index]) || absolute_error > tolerance)
        {
            error_count++;
        }
        if(absolute_error > max_absolute_error)
        {
            max_absolute_error = absolute_error;
            max_error_index = index;
        }
        max_relative_error = std::max(max_relative_error, relative_error);
    }

    for(int64_t row = 0; row < rows; row++)
    {
        double row_sum = 0.0;
        for(int64_t col = 0; col < cols; col++)
        {
            row_sum += actual[static_cast<size_t>(row * cols + col)];
        }
        max_row_sum_error = std::max(max_row_sum_error, static_cast<float>(std::abs(row_sum - 1.0)));
    }

    const bool passed = error_count == 0 && max_row_sum_error <= ROW_SUM_TOLERANCE;
    std::cout << "  [关键结果] errors=" << error_count << '/' << element_count << " max_abs=" << std::scientific
              << max_absolute_error << " max_rel=" << max_relative_error << " row_sum_error=" << max_row_sum_error
              << std::defaultfloat << '\n';

    if(!passed)
    {
        std::cout << "  [FAILED] index=" << max_error_index << " reference=" << reference[max_error_index]
                  << " actual=" << actual[max_error_index] << "\n\n";
        return false;
    }

    std::cout << "  [SUCCESS] 精度验证通过\n\n";
    return true;
}

bool run_int8_accuracy_case(int64_t rows, int64_t cols, float input_scale, int32_t input_zero_point, uint32_t seed)
{
    const size_t element_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t input_byte_count = element_count * sizeof(int8_t);
    const size_t float_byte_count = element_count * sizeof(float);

    std::vector<int8_t> source(element_count);
    std::vector<float> dequantized_source(element_count);
    std::vector<float> float_reference(element_count);
    std::vector<float> float_actual(element_count);
    std::vector<int8_t> int8_reference(element_count);
    std::vector<int8_t> int8_actual(element_count);

    fill_int8_input(source, seed);
    dequantize_int8_cpu(source, dequantized_source, input_scale, input_zero_point);
    softmax_cpu(dequantized_source, float_reference, rows, cols);
    quantize_softmax_int8_cpu(float_reference, int8_reference);

    int8_t *device_source = nullptr;
    float *device_float_destination = nullptr;
    int8_t *device_int8_destination = nullptr;

    std::cout << "[INT8 精度阶段] rows=" << std::setw(4) << rows << " cols=" << std::setw(4) << cols
              << " scale=" << input_scale << " zero_point=" << input_zero_point << '\n';

    bool success = check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_source), input_byte_count),
                              "cudaMalloc(int8 source)") &&
                   check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_float_destination), float_byte_count),
                              "cudaMalloc(float destination)") &&
                   check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_int8_destination), input_byte_count),
                              "cudaMalloc(int8 destination)");

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

    success =
        check_cuda(cudaMemcpy(device_source, source.data(), input_byte_count, cudaMemcpyHostToDevice),
                   "cudaMemcpy(int8 source)") &&
        check_cuda(cuda_lab::launch_softmax_int8_to_float(device_source, device_float_destination, rows, cols,
                                                          input_scale, input_zero_point),
                   "launch_softmax_int8_to_float") &&
        check_cuda(cuda_lab::launch_softmax_int8_to_int8(device_source, device_int8_destination, rows, cols,
                                                         input_scale, input_zero_point),
                   "launch_softmax_int8_to_int8") &&
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(int8 accuracy)") &&
        check_cuda(cudaMemcpy(float_actual.data(), device_float_destination, float_byte_count, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(float destination)") &&
        check_cuda(cudaMemcpy(int8_actual.data(), device_int8_destination, input_byte_count, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(int8 destination)");

    cudaFree(device_source);
    cudaFree(device_float_destination);
    cudaFree(device_int8_destination);

    if(!success)
    {
        return false;
    }

    constexpr float ABSOLUTE_TOLERANCE = 1.0e-6f;
    constexpr float RELATIVE_TOLERANCE = 1.0e-5f;
    size_t float_error_count = 0;
    size_t int8_error_count = 0;
    int32_t max_int8_difference = 0;
    float max_float_absolute_error = 0.0f;
    float max_int8_dequantized_error = 0.0f;
    float max_int8_row_sum_error = 0.0f;

    for(size_t index = 0; index < element_count; index++)
    {
        const float float_absolute_error = std::abs(float_reference[index] - float_actual[index]);
        const float tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * std::abs(float_reference[index]);
        if(!std::isfinite(float_actual[index]) || float_absolute_error > tolerance)
        {
            float_error_count++;
        }
        max_float_absolute_error = std::max(max_float_absolute_error, float_absolute_error);

        const int32_t int8_difference =
            std::abs(static_cast<int32_t>(int8_reference[index]) - static_cast<int32_t>(int8_actual[index]));
        if(int8_difference > 1)
        {
            int8_error_count++;
        }
        max_int8_difference = std::max(max_int8_difference, int8_difference);

        const float dequantized_probability =
            (static_cast<int32_t>(int8_actual[index]) - INT8_OUTPUT_ZERO_POINT) * INT8_OUTPUT_SCALE;
        max_int8_dequantized_error =
            std::max(max_int8_dequantized_error, std::abs(float_reference[index] - dequantized_probability));
    }

    for(int64_t row = 0; row < rows; row++)
    {
        float row_sum = 0.0f;
        for(int64_t col = 0; col < cols; col++)
        {
            const int64_t index = row * cols + col;
            row_sum += (static_cast<int32_t>(int8_actual[static_cast<size_t>(index)]) - INT8_OUTPUT_ZERO_POINT) *
                       INT8_OUTPUT_SCALE;
        }
        max_int8_row_sum_error = std::max(max_int8_row_sum_error, std::abs(row_sum - 1.0f));
    }

    const bool passed = float_error_count == 0 && int8_error_count == 0;
    std::cout << "  [关键结果] float_errors=" << float_error_count << '/' << element_count
              << " float_max_abs=" << std::scientific << max_float_absolute_error
              << " int8_errors_gt_1=" << int8_error_count << " int8_max_diff=" << max_int8_difference
              << " int8_dequant_max_abs=" << max_int8_dequantized_error
              << " int8_row_sum_error=" << max_int8_row_sum_error << std::defaultfloat << '\n';

    std::cout << "  " << (passed ? "[SUCCESS]" : "[FAILED]") << " INT8 输入精度验证" << (passed ? "通过" : "失败")
              << "\n\n";
    return passed;
}

} // namespace

int main()
{
    int device_id = 0;
    cudaDeviceProp device_properties{};
    if(!check_cuda(cudaGetDevice(&device_id), "cudaGetDevice") ||
       !check_cuda(cudaGetDeviceProperties(&device_properties, device_id), "cudaGetDeviceProperties"))
    {
        return 1;
    }

    std::cout << "\n=== CUDA Softmax 算子测试 ===\n\n"
              << "[配置] device=" << device_properties.name << " SM " << device_properties.major << '.'
              << device_properties.minor << " input_types={float32,int8}" << " output_types={float32,int8}"
              << " int8_max_cols=" << MAX_COLS << " fp32_block_max_cols=" << cuda_lab::kSoftmaxBlockMaxColumns
              << " warp_size=" << WARP_SIZE << '\n'
              << "[配置] INT8 output_scale=" << INT8_OUTPUT_SCALE << " output_zero_point=" << INT8_OUTPUT_ZERO_POINT
              << " compute=float32\n\n";

    constexpr std::array<std::pair<int64_t, int64_t>, 20> TEST_CASES = {{
        {1, 1},    {3, 17},   {7, 32},   {9, 33},   {17, 127}, {31, 256}, {33, 511}, {65, 1023}, {65, 1024}, {1, 1025},
        {7, 2047}, {9, 2048}, {3, 2049}, {7, 4096}, {9, 4097}, {3, 8191}, {7, 8192}, {9, 8193},  {3, 16384}, {5, 65537},
    }};

    bool success = true;
    uint32_t seed = 42U;
    for(const auto &[rows, cols] : TEST_CASES)
    {
        success = run_accuracy_case(rows, cols, seed++) && success;
    }

    // 覆盖 online 的标量前缀、尾部，以及输入输出对齐不同的标量写回分支。
    for(int offset = 1; offset <= 3; ++offset)
    {
        success = run_accuracy_case(5, 8192 + offset, seed++, offset, offset) && success;
        success = run_accuracy_case(5, 8192 + offset, seed++, offset, (offset + 1) % 4) && success;
    }
    for(const int64_t cols : {2049, 8193, 65537})
    {
        success = run_accuracy_case(3, cols, seed++, 0, 0, 1) && success;
        success = run_accuracy_case(3, cols, seed++, 0, 0, 2) && success;
    }

    std::cout << "[显式实现阶段] block512、block1024、cluster2、cluster4 与 online\n";
    for(const auto variant :
        {cuda_lab::SoftmaxVariant::Online, cuda_lab::SoftmaxVariant::Block512, cuda_lab::SoftmaxVariant::Block1024,
         cuda_lab::SoftmaxVariant::Cluster2, cuda_lab::SoftmaxVariant::Cluster4})
    {
        const int max_cols =
            (variant == cuda_lab::SoftmaxVariant::Block512 || variant == cuda_lab::SoftmaxVariant::Cluster2) ? 16384
                                                                                                             : 32768;
        cuda_lab::SoftmaxVariantInfo info{};
        const auto status = cuda_lab::query_softmax_variant(variant, max_cols, info);
        if(status == cudaErrorNotSupported)
        {
            std::cout << "[SKIP] variant=" << static_cast<int>(variant) << " 当前设备或构建不支持\n\n";
            continue;
        }
        if(!check_cuda(status, "query_softmax_variant"))
            return 1;
        std::cout << "[配置] variant=" << static_cast<int>(variant) << " threads=" << info.block_threads
                  << " cluster_blocks=" << info.cluster_blocks << " registers=" << info.registers_per_thread
                  << " local_bytes=" << info.local_bytes_per_thread << " active_units=" << info.active_units << "\n\n";
        for(const int cols : {1, 3, 255, 8191, 8192, 8193, 16383, 16384})
            success = run_accuracy_case(3, cols, seed++, 1, 2, 0, &variant) && success;
        if(max_cols == 32768)
            for(const int cols : {16385, 32767, 32768})
                success = run_accuracy_case(3, cols, seed++, 3, 1, 0, &variant) && success;
        for(const int pattern : {1, 2, 3, 4})
            success = run_accuracy_case(3, max_cols - 1, seed++, 0, 0, pattern, &variant) && success;
        success = run_accuracy_case(1, max_cols, seed++, 0, 0, 0, &variant) && success;
        success = run_accuracy_case(257, max_cols, seed++, 0, 0, 0, &variant) && success;
        float dummy = 0;
        cuda_lab::SoftmaxVariantInfo invalid_info{};
        const bool rejected =
            cuda_lab::launch_softmax_variant(nullptr, &dummy, 1, 1, variant) == cudaErrorInvalidValue &&
            cuda_lab::launch_softmax_variant(&dummy, &dummy, 0, 1, variant) == cudaErrorInvalidValue &&
            cuda_lab::launch_softmax_variant(&dummy, &dummy, INT64_MAX, 1, variant) == cudaErrorInvalidValue &&
            cuda_lab::query_softmax_variant(variant, 0, invalid_info) == cudaErrorInvalidValue &&
            (variant == cuda_lab::SoftmaxVariant::Online ||
             (cuda_lab::launch_softmax_variant(&dummy, &dummy, 1, max_cols + 1, variant) == cudaErrorInvalidValue &&
              cuda_lab::query_softmax_variant(variant, max_cols + 1, invalid_info) == cudaErrorInvalidValue));
        success = rejected && success;
        std::cout << (rejected ? "[SUCCESS]" : "[FAILED]") << " 显式实现参数检查\n\n";
    }

    constexpr std::array<std::tuple<int64_t, int64_t, float, int32_t>, 9> INT8_TEST_CASES = {{
        {1, 1, 1.0f / 32.0f, -3},
        {3, 17, 1.0f / 16.0f, -128},
        {7, 32, 1.0f / 32.0f, 127},
        {9, 33, 1.0f / 64.0f, 5},
        {17, 127, 1.0f / 32.0f, -17},
        {31, 256, 1.0f / 64.0f, 0},
        {33, 511, 1.0f / 32.0f, 23},
        {65, 1023, 1.0f / 16.0f, -7},
        {65, 1024, 1.0f / 32.0f, 11},
    }};

    for(const auto &[rows, cols, input_scale, input_zero_point] : INT8_TEST_CASES)
    {
        success = run_int8_accuracy_case(rows, cols, input_scale, input_zero_point, seed++) && success;
    }

    std::cout << "[参数验证阶段] 空指针、尺寸及 INT8 量化参数\n";
    float float_value = 0.0f;
    int8_t int8_value = 0;
    const bool invalid_rejected =
        cuda_lab::launch_softmax(nullptr, &float_value, 1, 1) == cudaErrorInvalidValue &&
        cuda_lab::launch_softmax(&float_value, &float_value, 0, 1) == cudaErrorInvalidValue &&
        cuda_lab::launch_softmax(&float_value, &float_value, 1, 0) == cudaErrorInvalidValue &&
        cuda_lab::launch_softmax(&float_value, &float_value, static_cast<int64_t>(std::numeric_limits<int>::max()) + 1,
                                 1025) == cudaErrorInvalidValue &&
        cuda_lab::launch_softmax_int8_to_float(&int8_value, &float_value, 1, MAX_COLS + 1, 1.0f, 0) ==
            cudaErrorInvalidValue &&
        cuda_lab::launch_softmax_int8_to_float(&int8_value, &float_value, 1, 1, 0.0f, 0) == cudaErrorInvalidValue &&
        cuda_lab::launch_softmax_int8_to_float(&int8_value, &float_value, 1, 1, std::numeric_limits<float>::quiet_NaN(),
                                               0) == cudaErrorInvalidValue &&
        cuda_lab::launch_softmax_int8_to_int8(&int8_value, &int8_value, 1, 1, 1.0f, 128) == cudaErrorInvalidValue;
    success = invalid_rejected && success;
    std::cout << "  [关键结果] invalid_rejected=" << invalid_rejected << '\n'
              << "  " << (invalid_rejected ? "[SUCCESS]" : "[FAILED]") << " 参数验证\n\n";

    std::cout << (success ? "[SUCCESS]" : "[FAILED]") << " Softmax 算子测试" << (success ? "通过" : "失败") << '\n';
    return success ? 0 : 1;
}
