#include "common.h"
#include "data.h"
#include <cuda_lab/cuda_check.cuh>
#include <iomanip>
#include <sstream>

namespace dwconv_bench
{

double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void check(cudaError_t error)
{
    cuda_lab::check_cuda(error, "dwconv");
}

Stream::Stream()
{
    check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
}

Stream::~Stream()
{
    cudaStreamDestroy(value);
}

const auto kShapeCases = make_shape_cases();

namespace
{

struct Events
{
    cudaEvent_t start{}, stop{};

    Events()
    {
        check(cudaEventCreate(&start));
        const auto error = cudaEventCreate(&stop);
        if(error != cudaSuccess)
        {
            cudaEventDestroy(start);
            check(error);
        }
    }

    ~Events()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

struct Result
{
    std::string status = "pass", reason;
    bool checked = false;
    std::vector<double> samples_us;
    double us = 0, min_us = 0, max_us = 0, prepare_ms = 0, post_ms = 0, host_layout_ms = 0;
    int algorithm = -1, engine = -1;
    std::optional<size_t> workspace_bytes = 0;
};

std::string json_string(const std::string &input)
{
    static constexpr char hex[] = "0123456789abcdef";
    std::string output;
    output += '"';
    for(unsigned char c : input)
    {
        if(c == '"' || c == '\\')
        {
            output += '\\';
            output += char(c);
        }
        else if(c < 0x20)
        {
            output += "\\u00";
            output += hex[c >> 4];
            output += hex[c & 15];
        }
        else
            output += char(c);
    }
    output += '"';
    return output;
}

void emit_result(size_t id, const std::string &kernel, const std::string &dtype, const Options &options,
                 const Result &result)
{
    std::ostringstream output;
    output << std::setprecision(12);
    output << "DWCONV_RESULT {\"schema_version\":1,\"status\":" << json_string(result.status) << ",\"case\":" << id
           << ",\"kernel\":" << json_string(kernel) << ",\"dtype\":" << json_string(dtype)
           << ",\"layout\":" << json_string(options.layout)
           << ",\"timing_mode\":\"event\",\"warmup\":" << options.warmup << ",\"iterations\":" << options.iterations
           << ",\"samples\":" << options.samples << ",\"checked\":" << (result.checked ? "true" : "false");

    if(result.status == "pass")
    {
        output << ",\"us\":" << result.us << ",\"min_us\":" << result.min_us << ",\"max_us\":" << result.max_us
               << ",\"prepare_ms\":" << result.prepare_ms << ",\"post_ms\":" << result.post_ms
               << ",\"host_layout_ms\":" << result.host_layout_ms << ",\"algo\":" << result.algorithm
               << ",\"engine\":" << result.engine
               << ",\"workspace_bytes\":" << (result.workspace_bytes ? std::to_string(*result.workspace_bytes) : "null")
               << ",\"samples_us\":[";

        for(size_t i = 0; i < result.samples_us.size(); ++i)
        {
            if(i)
                output << ',';
            output << result.samples_us[i];
        }
        output << ']';
    }
    else
        output << ",\"reason\":" << json_string(result.reason);
    output << '}';
    std::cout << output.str() << '\n';
}

// NOTE:测速代码
template <class Launch> void measure(Launch launch, const Options &options, cudaStream_t stream, Result &result)
{
    Events events;
    for(int i = 0; i < options.warmup; ++i)
        launch();
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));

    for(int sample = 0; sample < options.samples; ++sample)
    {
        check(cudaEventRecord(events.start, stream));
        for(int i = 0; i < options.iterations; ++i)
            launch();
        check(cudaEventRecord(events.stop, stream));
        check(cudaGetLastError());
        check(cudaEventSynchronize(events.stop));

        float ms = 0;
        check(cudaEventElapsedTime(&ms, events.start, events.stop));
        const double us = double(ms) * 1000 / options.iterations;

        if(!std::isfinite(us) || us <= 0)
            throw std::runtime_error("无效的 CUDA Event 计时结果");

        result.samples_us.push_back(us);
        std::cout << "[采样] event sample=" << sample << " us=" << us << '\n';
    }

    auto sorted = result.samples_us;
    std::sort(sorted.begin(), sorted.end());
    result.us = (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) / 2;
    result.min_us = sorted.front();
    result.max_us = sorted.back();

    std::cout << "[计时] event 中位数=" << result.us << " us, min=" << result.min_us << ", max=" << result.max_us
              << "，预热=" << options.warmup << "，每样本迭代=" << options.iterations << "，样本=" << options.samples
              << '\n';
}

} // namespace

template <class T>
Result run_forward(const ShapeCase &s, const std::string &kernel, const Options &options, const Backend &backend)
{
    Result result;
    const auto prepare = Clock::now();
    ForwardData<T> data(s, backend, options, options.check_cpu);
    auto forward = backend.prepare(data.make_case(s, options, kernel));
    check(cudaDeviceSynchronize());
    result.prepare_ms = elapsed_ms(prepare);
    result.host_layout_ms = data.host.layout_ms;
    result.algorithm = forward->algorithm;
    result.engine = forward->engine;
    result.workspace_bytes = forward->workspace_bytes;
    std::cout << "[准备] " << result.prepare_ms << " ms, 主机布局重排=" << result.host_layout_ms
              << " ms; 库初始化、算法选择、权重打包和 H2D 均在计时之外\n";
    if(options.check_cpu)
    {
        // 可选校验先于预热与采样，校验失败不会生成成功耗时。
        forward->launch();
        check(cudaGetLastError());
        check(cudaStreamSynchronize(forward->stream));
        const auto actual = data.download();
        validate_forward(s, data.host.param, data.host.input, data.host.weights, data.host.bias, actual,
                         forward->fp16_tolerance, forward->output_nhwc);
        result.checked = true;
    }
    measure([&] { forward->launch(); }, options, forward->stream, result);
    const auto post = Clock::now();
    forward.reset();
    data.release();
    result.post_ms = elapsed_ms(post);
    return result;
}

namespace
{

int parse_number(const std::string &text)
{
    size_t used = 0;
    const int result = std::stoi(text, &used);
    if(used != text.size() || result < 0)
        throw std::invalid_argument("参数必须为非负整数：" + text);
    return result;
}

void print_shape(size_t id, const ShapeCase &s)
{
    std::cout << "[配置] case=" << id << " NCHW=" << s.n << ',' << s.c << ',' << s.h << ',' << s.w << " K=" << s.kh
              << 'x' << s.kw << " stride=" << s.sh << 'x' << s.sw << " padding=" << s.ph << 'x' << s.pw
              << " bias=" << s.bias << '\n';
}

void usage(const std::vector<Backend> &backends)
{
    std::cout << "用法：dwconv_compare_bench [--kernel ";
    for(const auto &backend : backends)
        for(const auto &kernel : backend.kernels)
            std::cout << kernel << '|';
    std::cout << "all]\n"
                 "  [--dtype fp32|fp16|all] [--layout nchw|nhwc] [--case ID|all] [--quick] [--list-cases]\n"
                 "  [--warmup N] [--iterations N] [--samples N] [--check-cpu] [--help]\n"
                 "默认 baseline / fp32 / nchw / 全部兼容 shape; 预热20次, 每样本100次, 采样5次。\n"
                 "Event 包围普通连续 launch, 报告每次调用的样本中位数；不使用 CUDA Graph。\n"
                 "每条结果输出 DWCONV_RESULT JSON, 含 pass/skip/error; --check-cpu 在计时前校验。\n"
                 "cuDNN 别名 cudnn-graph/cudnn-autotune 均归一化为 cudnn, 只测前向。\n";
}

} // namespace

int run_benchmarks(int argc, char **argv, const std::vector<Backend> &backends)
{
    try
    {
        Options options;
        for(int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if(arg == "--help")
            {
                usage(backends);
                return 0;
            }
            if(arg == "--quick")
            {
                options.quick = true;
                continue;
            }
            if(arg == "--list-cases")
            {
                options.list = true;
                continue;
            }
            if(arg == "--check-cpu")
            {
                options.check_cpu = true;
                continue;
            }
            if(arg != "--kernel" && arg != "--dtype" && arg != "--layout" && arg != "--case" && arg != "--warmup" &&
               arg != "--iterations" && arg != "--samples")
                throw std::invalid_argument("未知参数：" + arg);
            if(++i == argc)
                throw std::invalid_argument("缺少参数值：" + arg);
            const std::string val = argv[i];
            if(arg == "--kernel")
                options.kernel = val;
            else if(arg == "--dtype")
                options.dtype = val;
            else if(arg == "--layout")
                options.layout = val;
            else if(arg == "--case")
                options.case_index = val == "all" ? -1 : parse_number(val);
            else if(arg == "--warmup")
                options.warmup = parse_number(val);
            else if(arg == "--iterations")
                options.iterations = parse_number(val);
            else
                options.samples = parse_number(val);
        }
        if(options.kernel == "cudnn-autotune" || options.kernel == "cudnn-graph")
            options.kernel = "cudnn";
        if(options.dtype != "fp32" && options.dtype != "fp16" && options.dtype != "all")
            throw std::invalid_argument("未知 dtype: " + options.dtype);
        if(options.layout != "nchw" && options.layout != "nhwc")
            throw std::invalid_argument("未知 layout: " + options.layout);
        if(options.layout == "nhwc" && options.kernel != "all" && options.kernel != "pytorch" &&
           options.kernel.rfind("cudnn", 0) != 0)
            throw std::invalid_argument("NHWC 支持 pytorch/cuDNN");
        bool found = options.kernel == "all";
        for(const auto &backend : backends)
            found = found ||
                    std::find(backend.kernels.begin(), backend.kernels.end(), options.kernel) != backend.kernels.end();
        if(!found)
            throw std::invalid_argument("当前目标未包含 kernel: " + options.kernel);
        const size_t case_count = options.quick ? kQuickCases.size() : kShapeCases.size();
        if(options.iterations == 0 || options.samples == 0 || options.case_index >= int(case_count))
            throw std::invalid_argument("iterations/samples 必须大于0, case 必须在所选 shape 集合范围内");
        if(options.list)
        {
            for(size_t id = 0; id < case_count; ++id)
                print_shape(id, kShapeCases[id]);
            return 0;
        }
        int device_id = 0, runtime = 0;
        check(cudaGetDevice(&device_id));
        cudaDeviceProp device{};
        check(cudaGetDeviceProperties(&device, device_id));
        check(cudaFree(nullptr));
        check(cudaRuntimeGetVersion(&runtime));
        std::cout << "[设备] " << device.name << " SM=" << device.major << '.' << device.minor << " CUDA=" << runtime
                  << "\n[计时] mode=event; TRT 使用非默认流，其他前向使用默认流；包含普通 launch 的提交间隙\n";
        int completed = 0, skipped = 0, failed = 0;
        for(size_t id = 0; id < case_count; ++id)
        {
            if(options.case_index >= 0 && options.case_index != int(id))
                continue;
            const auto &shape = kShapeCases[id];
            for(const std::string dtype : {std::string("fp32"), std::string("fp16")})
            {
                if(options.dtype != "all" && options.dtype != dtype)
                    continue;
                for(const auto &backend : backends)
                    for(const auto &kernel : backend.kernels)
                    {
                        if(options.kernel != "all" && options.kernel != kernel)
                            continue;
                        std::cout << '\n';
                        print_shape(id, shape);
                        std::cout << "[配置] kernel=" << kernel << " dtype=" << dtype << " layout=" << options.layout
                                  << '\n';
                        Result result;
                        try
                        {
                            if(!backend.compatible(shape, kernel, dtype, options))
                                throw Unsupported("kernel/dtype/shape/layout 组合不支持");
                            result = dtype == "fp16" ? run_forward<__half>(shape, kernel, options, backend)
                                                     : run_forward<float>(shape, kernel, options, backend);
                            ++completed;
                        }
                        catch(const Unsupported &error)
                        {
                            result.status = "skip";
                            result.reason = error.what();
                            ++skipped;
                        }
                        catch(const std::exception &error)
                        {
                            result.status = "error";
                            result.reason = error.what();
                            ++failed;
                        }
                        emit_result(id, kernel, dtype, options, result);
                    }
            }
        }
        if(failed)
        {
            std::cout << "[失败] dwconv_compare_bench 完成=" << completed << " 跳过=" << skipped << " 失败=" << failed
                      << '\n';
            return 1;
        }
        if(!completed)
            throw std::runtime_error("没有兼容的 kernel / dtype / shape 组合");
        std::cout << "[SUCCESS] dwconv_compare_bench 完成=" << completed << " 跳过=" << skipped << '\n';
        return 0;
    }
    catch(const std::exception &error)
    {
        std::cout << "[失败] " << error.what() << '\n';
        return 1;
    }
}

} // namespace dwconv_bench
