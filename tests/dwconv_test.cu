#include <dwconv.cuh>
#include <dwconv_fp16.cuh>
#include <dwconv_opt.cuh>
#include <dwconv_pytorch.cuh>
#include <cuda_lab/cuda_check.cuh>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>
#include <memory>
#ifdef DWCONV_HAS_CUDNN
#include <dwconv_cudnn.cuh>
#endif

namespace
{

using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void check(cudaError_t error)
{
    cuda_lab::check_cuda(error, "dwconv_test");
}

bool is_cudnn(const std::string &kernel)
{
    return kernel == "cudnn" || kernel == "cudnn-fused" || kernel == "cudnn-separate";
}

struct Options
{
    std::string kernel = "baseline", dtype = "fp32", layout = "nchw";
    int case_index = -1, warmup = 20, iterations = 100;
    bool check_cpu = false, list = false, quick = false;
};
Options options;

struct ShapeCase
{
    int n, c, h, w, kh, kw, sh, sw, ph, pw;
    bool bias;
};

// {B, C, H, W, Kh, Kw, Sh, Sw, Ph, Pw, 是否启用 bias}
const std::array<ShapeCase, 23> kQuickCases = {{
    {32, 128, 80, 80, 7, 7, 1, 1, 3, 3, true},   // throughput
    {4, 128, 80, 80, 7, 7, 1, 1, 3, 3, true},    // main
    {4, 128, 64, 64, 7, 7, 1, 1, 3, 3, true},    // k7_out32
    {4, 128, 96, 96, 7, 7, 1, 1, 3, 3, true},    // k7_out48
    {1, 32, 43, 43, 7, 7, 1, 1, 3, 3, true},     // k7_out22
    {1, 32, 80, 80, 5, 5, 1, 1, 2, 2, true},     // common
    {2, 64, 64, 64, 3, 3, 1, 1, 1, 1, true},     // aligned
    {1, 32, 43, 43, 3, 3, 1, 1, 1, 1, true},     // boundary
    {1, 32, 43, 43, 11, 11, 1, 1, 5, 5, true},   // k11_tail
    {32, 128, 80, 80, 11, 11, 1, 1, 5, 5, true}, // k11_throughput
    {4, 128, 80, 80, 11, 11, 1, 1, 5, 5, true},  // k11_main
    {4, 128, 64, 64, 11, 11, 1, 1, 5, 5, true},  // k11_out32
    {1, 32, 65, 81, 11, 11, 1, 1, 5, 5, true},   // k11_rectangular
    {1, 32, 43, 43, 11, 11, 1, 1, 5, 5, true},   // k11_stride1
    {1, 32, 43, 43, 3, 3, 1, 1, 1, 1, true},     // stride1
    {1, 4, 43, 43, 7, 7, 1, 1, 3, 3, true},      // min_channels
    {1, 36, 43, 43, 7, 7, 1, 1, 3, 3, true},     // non_power_channels
    {1, 32, 61, 61, 7, 7, 1, 1, 3, 3, true},     // dispatch_out31
    {1, 32, 63, 63, 7, 7, 1, 1, 3, 3, true},     // dispatch_out32
    {1, 32, 65, 65, 7, 7, 1, 1, 3, 3, true},     // dispatch_out33
    {1, 32, 107, 107, 7, 7, 1, 1, 3, 3, true},   // dispatch_blocks96
    {1, 32, 111, 111, 7, 7, 1, 1, 3, 3, true},   // dispatch_blocks104
    {1, 32, 65, 81, 7, 7, 1, 1, 3, 3, true},     // rectangular
}};

std::vector<ShapeCase> make_shape_cases()
{
    constexpr int batches[] = {1, 2, 4, 8, 16, 32, 64};
    constexpr int channels[] = {32, 64, 128, 256, 512, 1024};
    constexpr int spatial_sizes[] = {40, 80, 160};
    constexpr int kernel_sizes[] = {3, 5, 7, 9, 11};
    std::vector<ShapeCase> cases(kQuickCases.begin(), kQuickCases.end());

    for(int k : kernel_sizes)
        for(int n : batches)
            for(int c : channels)
                for(int h : spatial_sizes)
                    cases.push_back({n, c, h, h, k, k, 1, 1, k / 2, k / 2, true});
    return cases;
}

const std::vector<ShapeCase> kShapeCases = make_shape_cases();

Conv2DParam make_param(const ShapeCase &s)
{
    Conv2DParam p{};
    p.in_h = s.h;
    p.in_w = s.w;
    p.in_ch = p.out_ch = s.c;
    p.Kh = s.kh;
    p.Kw = s.kw;
    p.KhKw = s.kh * s.kw;
    p.Sh = s.sh;
    p.Sw = s.sw;
    p.Ph = s.ph;
    p.Pw = s.pw;
    p.inHW = s.h * s.w;
    p.inBatchNumel = s.c * p.inHW;
    p.out_h = (s.h + 2 * s.ph - s.kh) / s.sh + 1;
    p.out_w = (s.w + 2 * s.pw - s.kw) / s.sw + 1;
    p.outHW = p.out_h * p.out_w;
    p.outBatchNumel = s.c * p.outHW;
    p.inChKhKw = s.c * p.KhKw;
    p.kernelWeightsCount = p.inChKhKw;
    p.biasWeightsCount = s.c;
    return p;
}

struct Events
{
    cudaEvent_t start{}, stop{};
    Events()
    {
        check(cudaEventCreate(&start));
        check(cudaEventCreate(&stop));
    }
    ~Events()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

template <class Launch> void measure(Launch launch, cudaStream_t stream = nullptr)
{
    Events events;
    for(int i = 0; i < options.warmup; ++i)
        launch();
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    check(cudaEventRecord(events.start, stream));
    for(int i = 0; i < options.iterations; ++i)
        launch();
    check(cudaEventRecord(events.stop, stream));
    check(cudaGetLastError());
    check(cudaEventSynchronize(events.stop));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, events.start, events.stop));
    std::cout << "\n[阶段] kernel launch：平均=" << ms * 1000 / options.iterations << " us，预热=" << options.warmup
              << "，迭代=" << options.iterations << "\n";
}

template <class T> struct DeviceBuffer
{
    T *ptr = nullptr;
    explicit DeviceBuffer(size_t n)
    {
        check(cudaMalloc(&ptr, n * sizeof(T)));
    }
    ~DeviceBuffer()
    {
        cudaFree(ptr);
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    void upload(const std::vector<T> &v)
    {
        check(cudaMemcpy(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    void release()
    {
        check(cudaFree(ptr));
        ptr = nullptr;
    }
};

template <class T> T convert(float v)
{
    if constexpr(std::is_same_v<T, __half>)
        return __float2half_rn(v);
    else
        return v;
}
template <class T> float value(T v)
{
    if constexpr(std::is_same_v<T, __half>)
        return __half2float(v);
    else
        return v;
}

template <class T>
void validate_forward(const ShapeCase &s, const Conv2DParam &p, const std::vector<T> &input,
                      const std::vector<T> &weights, const std::vector<T> &bias, const std::vector<T> &actual,
                      bool cudnn = false, bool nhwc = false)
{
    const auto start = Clock::now();
    double max_error = 0;
    for(int b = 0; b < s.n; ++b)
        for(int c = 0; c < s.c; ++c)
            for(int y = 0; y < p.out_h; ++y)
                for(int x = 0; x < p.out_w; ++x)
                {
                    double sum = s.bias ? value(bias[c]) : 0;
                    for(int ky = 0; ky < s.kh; ++ky)
                        for(int kx = 0; kx < s.kw; ++kx)
                        {
                            int iy = y * s.sh + ky - s.ph, ix = x * s.sw + kx - s.pw;
                            if(iy < 0 || iy >= s.h || ix < 0 || ix >= s.w)
                                continue;
                            double a = value(input[(b * s.c + c) * p.inHW + iy * s.w + ix]);
                            double w = value(weights[c * p.KhKw + ky * s.kw + kx]);
                            if constexpr(std::is_same_v<T, __half>)
                                sum = std::fma(float(a), float(w), float(sum));
                            else
                                sum = std::fma(a, w, sum);
                        }
                    if constexpr(std::is_same_v<T, __half>)
                        sum = __half2float(__double2half(sum));
                    size_t index = nhwc ? ((size_t(b) * p.out_h + y) * p.out_w + x) * s.c + c
                                        : (size_t(b) * s.c + c) * p.outHW + y * p.out_w + x;
                    double got = value(actual[index]), error = std::abs(got - sum);
                    double tolerance = std::is_same_v<T, __half> ? (cudnn ? 2e-3 * (1 + std::abs(sum)) : 0)
                                                                 : 3e-4 * (1 + std::abs(sum));
                    if(!std::isfinite(got) || error > tolerance)
                        throw std::runtime_error("CPU 校验失败 index=" + std::to_string(index));
                    max_error = std::max(max_error, error);
                }
    std::cout << "\n[校验] CPU 最大绝对误差=" << max_error << "，耗时=" << elapsed_ms(start) << " ms [SUCCESS]\n";
}

void launch_fp32(const std::string &kernel, float *x, float *w, float *b, float *y, Conv2DParam p, int n)
{
    dim3 grid((p.outHW + 255) / 256, p.in_ch / 4, n);
    if(kernel == "opt")
        launch_opt_fp32(x, w, b, y, p, n);
    else if(kernel == "pytorch")
        launch_pytorch_forward_fp32(x, w, b, y, p, n);
    else if(kernel == "prefetch")
        conv2d_4x128x256_groups_kernel_prefetch<<<grid, 256>>>(x, w, b, y, p);
    else if(kernel == "baseline-k32")
        conv2d_4x32x256_groups_kernel<<<grid, 256>>>(x, w, b, y, p);
    else if(kernel == "baseline-k64")
        conv2d_4x64x256_groups_kernel<<<grid, 256>>>(x, w, b, y, p);
    else if(kernel == "baseline-k128")
        conv2d_4x128x256_groups_kernel<<<grid, 256>>>(x, w, b, y, p);
    else if(kernel == "gmem")
        conv2d_4x128x256_groups_kernel_gmem<<<grid, 256>>>(x, w, b, y, p);
    else if(kernel == "gmem-l2-entry")
        conv2d_4x128x256_groups_kernel_gmem_l2<<<grid, 256>>>(x, w, b, y, p);
    else
        launch_baseline_fp32(x, w, b, y, p, n);
}

// weights：PyTorch 使用普通 half 权重，其余实现使用配对 half2 权重。
void launch_fp16(const std::string &kernel, const __half *x, const void *weights, const __half *b, __half *y,
                 const Conv2DParam &p, int n)
{
    const auto *w = static_cast<const __half2 *>(weights);
    if(kernel == "opt")
        launch_opt_fp16(x, w, b, y, p, n);
    else if(kernel == "pytorch")
        launch_pytorch_forward_fp16(x, static_cast<const __half *>(weights), b, y, p, n);
    else if(kernel == "prefetch")
        launch_prefetch_fp16_k128(x, w, b, y, p, n);
    else if(kernel == "baseline-k32")
        launch_baseline_fp16_k32(x, w, b, y, p, n);
    else if(kernel == "baseline-k64")
        launch_baseline_fp16_k64(x, w, b, y, p, n);
    else if(kernel == "baseline-k128")
        launch_baseline_fp16_k128(x, w, b, y, p, n);
    else
        launch_baseline_fp16(x, w, b, y, p, n);
}

template <class T> void run_forward(const ShapeCase &s, const std::string &kernel)
{
    const auto p = make_param(s);
    const auto prepare = Clock::now();
    std::vector<T> input(size_t(s.n) * p.inBatchNumel), weights(s.c * p.KhKw), bias(s.c);

    for(size_t i = 0; i < input.size(); ++i)
        input[i] = convert<T>((int(i % 251) - 125) / 509.0f);
    for(size_t i = 0; i < weights.size(); ++i)
        weights[i] = convert<T>((int(i % 127) - 63) / 257.0f);
    for(size_t i = 0; i < bias.size(); ++i)
        bias[i] = convert<T>((int(i % 17) - 8) / 97.0f);
    using Weight = std::conditional_t<std::is_same_v<T, __half>, __half2, float>;

    std::vector<Weight> packed;
    if constexpr(std::is_same_v<T, __half>)
    {
        if(kernel != "pytorch" && !is_cudnn(kernel))
        {
            packed.resize(weights.size() / 2);
            for(int pair = 0; pair < s.c / 2; ++pair)
                for(int k = 0; k < p.KhKw; ++k)
                    packed[pair * p.KhKw + k] =
                        __halves2half2(weights[2 * pair * p.KhKw + k], weights[(2 * pair + 1) * p.KhKw + k]);
        }
    }
    else
    {
        packed = weights;
    }

    size_t count = size_t(s.n) * p.outBatchNumel;
    DeviceBuffer<T> x(input.size()), b(bias.size()), y(count);
    DeviceBuffer<Weight> w((kernel == "pytorch" || is_cudnn(kernel))
                               ? (weights.size() * sizeof(T) + sizeof(Weight) - 1) / sizeof(Weight)
                               : packed.size());

#ifdef DWCONV_HAS_CUDNN
    std::unique_ptr<CudnnForward> cudnn;
    if(is_cudnn(kernel))
    {
        cudnn = std::make_unique<CudnnForward>(p, s.n, std::is_same_v<T, __half>, nullptr,
                                               kernel == "cudnn-fused"      ? CudnnForwardMode::Fused
                                               : kernel == "cudnn-separate" ? CudnnForwardMode::Separate
                                                                            : CudnnForwardMode::Graph,
                                               options.layout == "nhwc" ? CudnnLayout::NHWC : CudnnLayout::NCHW);
        const auto info = cudnn->info();
        std::cout << "[cuDNN] version=" << info.version << " algo=" << info.algorithm << " engine=" << info.engine
                  << " workspace=" << info.workspace_bytes << " bytes，layout=" << options.layout
                  << "，FP32 累加，FP32 禁用 TF32，计时包含 convolution"
                  << (kernel == "cudnn-separate" ? (s.bias ? " + add bias" : "") : " fused bias") << "\n";
    }
#endif

    std::vector<T> layout_input;
    double layout_ms = 0;
    if(options.layout == "nhwc")
    {
        const auto start = Clock::now();
        layout_input.resize(input.size());
        for(int n = 0; n < s.n; ++n)
            for(int hw = 0; hw < p.inHW; ++hw)
                for(int c = 0; c < s.c; ++c)
                    layout_input[(size_t(n) * p.inHW + hw) * s.c + c] = input[(size_t(n) * s.c + c) * p.inHW + hw];
        layout_ms = elapsed_ms(start);
    }
    x.upload(options.layout == "nhwc" ? layout_input : input);
    b.upload(bias);
    if(kernel == "pytorch" || is_cudnn(kernel))
        check(cudaMemcpy(w.ptr, weights.data(), weights.size() * sizeof(T), cudaMemcpyHostToDevice));
    else
        w.upload(packed);
    if(options.check_cpu)
        check(cudaMemset(y.ptr, 0xff, count * sizeof(T)));

#ifdef DWCONV_HAS_CUDNN
    if(cudnn)
    {
        cudnn->bind(x.ptr, w.ptr, s.bias ? b.ptr : nullptr, y.ptr);
        // 首次执行验证支持情况并完成库的懒初始化，不计入 kernel 时间。
        cudnn->launch(x.ptr, w.ptr, s.bias ? b.ptr : nullptr, y.ptr);
    }
#endif

    check(cudaDeviceSynchronize());
    std::cout << "\n[阶段] GPU-Mem / preprocess：" << elapsed_ms(prepare)
              << " ms（主机数据生成、按需权重配对、分配、H2D；cuDNN 含算法选择和 workspace）\n";
    std::cout << "[布局] layout=" << options.layout << " 主机输入重排=" << layout_ms
              << " ms（已计入 preprocess，kernel 阶段输入输出原生布局；不含外部 GPU 转换）\n";
    measure(
        [&]
        {

#ifdef DWCONV_HAS_CUDNN
            if(cudnn)
            {
                cudnn->launch(x.ptr, w.ptr, s.bias ? b.ptr : nullptr, y.ptr);
                return;
            }
#endif

            if constexpr(std::is_same_v<T, __half>)
                launch_fp16(kernel, x.ptr, w.ptr, s.bias ? b.ptr : nullptr, y.ptr, p, s.n);
            else
                launch_fp32(kernel, x.ptr, w.ptr, s.bias ? b.ptr : nullptr, y.ptr, p, s.n);
        });
    const auto post = Clock::now();
    std::vector<T> actual;
    if(options.check_cpu)
    {
        actual.resize(count);
        check(cudaMemcpy(actual.data(), y.ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

#ifdef DWCONV_HAS_CUDNN
    cudnn.reset();
#endif

    x.release();
    w.release();
    b.release();
    y.release();
    std::cout << "\n[阶段] free-Mem / postprocess：" << elapsed_ms(post) << " ms（释放 GPU 内存，启用校验时含 D2H）\n";
    if(options.check_cpu)
        validate_forward(s, p, input, weights, bias, actual, is_cudnn(kernel), options.layout == "nhwc");
    else
        std::cout << "[校验] CPU 校验关闭\n";
}

template <class T> struct Buffer
{
    T *ptr = nullptr;
    size_t size;
    explicit Buffer(size_t n) : size(n)
    {
        check(cudaMalloc(&ptr, n * sizeof(T)));
        check(cudaMemset(ptr, 0xff, n * sizeof(T)));
    }
    ~Buffer()
    {
        cudaFree(ptr);
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    void upload(const std::vector<T> &x)
    {
        check(cudaMemcpy(ptr, x.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
    void compare(const std::vector<double> &ref, const char *name)
    {
        std::vector<T> x(size);
        check(cudaMemcpy(x.data(), ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
        double max_error = 0;
        for(size_t i = 0; i < size; ++i)
        {
            double e = std::abs(value(x[i]) - ref[i]);
            const double tolerance = std::is_same_v<T, __half> ? 1e-3 : 3e-4;
            if(!std::isfinite(value(x[i])) || e > tolerance * (1 + std::abs(ref[i])))
                throw std::runtime_error(std::string(name) + " 不匹配 index=" + std::to_string(i) + " actual=" +
                                         std::to_string(value(x[i])) + " expected=" + std::to_string(ref[i]));
            max_error = std::max(max_error, e);
        }
        std::cout << "[结果] " << name << " 最大绝对误差=" << max_error << " [SUCCESS]\n";
    }
};

struct Stream
{
    cudaStream_t value{};
    Stream()
    {
        check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
    }
    ~Stream()
    {
        cudaStreamDestroy(value);
    }
};

template <class T> void run_backward(const ShapeCase &shape)
{
    const auto prepare = Clock::now();
    const auto p = make_param(shape);
    const int n = shape.n, c = shape.c, h = shape.h, w = shape.w, kh = shape.kh, kw = shape.kw;
    const int sh = shape.sh, sw = shape.sw, ph = shape.ph, pw = shape.pw;
    const bool bias = shape.bias;
    std::vector<T> x(n * p.inBatchNumel), k(c * kh * kw), b(c), g(n * p.outBatchNumel);
    auto fill = [](std::vector<T> &v, int mod, float divisor)
    {
        for(size_t i = 0; i < v.size(); ++i)
            v[i] = convert<T>((int((i * 13 + 7) % mod) - mod / 2) / divisor);
    };
    fill(x, 29, 31);
    fill(k, 17, 19);
    fill(b, 11, 13);
    fill(g, 23, 29);
    std::vector<double> y, dx, dk, db;
    if(options.check_cpu)
    {
        y.resize(g.size());
        dx.resize(x.size());
        dk.resize(k.size());
        db.resize(c);
    }
    // 独立参考：按前向输出遍历并 scatter 梯度，避免复用 GPU gather 的索引逻辑。
    auto reference = [&](bool gradients)
    {
        double loss = 0;
        for(int batch = 0; batch < n; ++batch)
            for(int ch = 0; ch < c; ++ch)
                for(int oy = 0; oy < p.out_h; ++oy)
                    for(int ox = 0; ox < p.out_w; ++ox)
                    {
                        int o = (batch * c + ch) * p.outHW + oy * p.out_w + ox;
                        double sum = bias ? value(b[ch]) : 0;
                        if(gradients)
                            db[ch] += value(g[o]);
                        for(int r = 0; r < kh; ++r)
                            for(int s = 0; s < kw; ++s)
                            {
                                int iy = oy * sh - ph + r, ix = ox * sw - pw + s;
                                if(iy < 0 || iy >= h || ix < 0 || ix >= w)
                                    continue;
                                int i = (batch * c + ch) * h * w + iy * w + ix, t = ch * kh * kw + r * kw + s;
                                sum += double(value(x[i])) * value(k[t]);
                                if(gradients)
                                {
                                    dx[i] += double(value(g[o])) * value(k[t]);
                                    dk[t] += double(value(g[o])) * value(x[i]);
                                }
                            }
                        if(gradients)
                            y[o] = sum;
                        loss += sum * value(g[o]);
                    }
        return loss;
    };
    Buffer<T> bx(x.size()), bk(k.size()), bb(b.size()), bg(g.size()), by(g.size()), bdx(x.size()), bdk(k.size()),
        bdb(c);
    bx.upload(x);
    bk.upload(k);
    bb.upload(b);
    bg.upload(g);
    Stream stream;
    // 上传在默认流完成后，非阻塞计算流才能读取输入。
    check(cudaDeviceSynchronize());
    std::cout << "\n[阶段] GPU-Mem / preprocess：" << elapsed_ms(prepare) << " ms（数据生成、分配、H2D）\n";
    auto backward =
        [](const T *x, const T *w, const T *g, T *dx, T *dk, T *db, const Conv2DParam &p, int n, cudaStream_t stream)
    {
        if constexpr(std::is_same_v<T, __half>)
            launch_pytorch_backward_fp16(x, w, g, dx, dk, db, p, n, stream);
        else
            launch_pytorch_backward_fp32(x, w, g, dx, dk, db, p, n, stream);
    };
    // backward 的一次迭代包含 dX、dK、db 三个 kernel。
    measure([&] { backward(bx.ptr, bk.ptr, bg.ptr, bdx.ptr, bdk.ptr, bdb.ptr, p, n, stream.value); }, stream.value);
    if(options.check_cpu)
    {
        const auto cpu_start = Clock::now();
        std::cout << "\n[阶段] CPU double 前向与反向参考\n";
        reference(true);
        if(x.size() <= 100)
        {
            std::cout << "\n[阶段] 小张量有限差分检查\n";
            auto finite_difference = [&](std::vector<T> &v, const std::vector<double> &grad)
            {
                for(size_t i = 0; i < v.size(); ++i)
                {
                    T saved = v[i];
                    v[i] = convert<T>(value(saved) + 0.001f);
                    float hi = v[i];
                    double a = reference(false);
                    v[i] = convert<T>(value(saved) - 0.001f);
                    float lo = v[i];
                    double z = reference(false);
                    v[i] = saved;
                    if(std::abs((a - z) / double(hi - lo) - grad[i]) > 1e-7 * (1 + std::abs(grad[i])))
                        throw std::runtime_error("有限差分不匹配");
                }
            };
            finite_difference(x, dx);
            finite_difference(k, dk);
            if(bias)
                finite_difference(b, db);
            std::cout << "[SUCCESS] 有限差分\n";
        }
        std::cout << "[校验] CPU 参考耗时=" << elapsed_ms(cpu_start) << " ms\n";
        if constexpr(std::is_same_v<T, __half>)
            launch_pytorch_forward_fp16(bx.ptr, bk.ptr, bias ? bb.ptr : nullptr, by.ptr, p, n, stream.value);
        else
            launch_pytorch_forward_fp32(bx.ptr, bk.ptr, bias ? bb.ptr : nullptr, by.ptr, p, n, stream.value);
        check(cudaGetLastError());
        check(cudaStreamSynchronize(stream.value));
        by.compare(y, "前向");
        bdx.compare(dx, "dX");
        bdk.compare(dk, "dK");
        bdb.compare(db, "db");
        // 重复调用必须覆盖旧结果；分别验证选择性梯度及其最小输入依赖。
        backward(nullptr, bk.ptr, bg.ptr, bdx.ptr, nullptr, nullptr, p, n, stream.value);
        check(cudaGetLastError());
        backward(bx.ptr, nullptr, bg.ptr, nullptr, bdk.ptr, nullptr, p, n, stream.value);
        check(cudaGetLastError());
        backward(nullptr, nullptr, bg.ptr, nullptr, nullptr, bdb.ptr, p, n, stream.value);
        check(cudaGetLastError());
        backward(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, p, n, stream.value);
        check(cudaStreamSynchronize(stream.value));
        std::cout << "\n[阶段] 选择性梯度与覆盖写入\n";
        bdx.compare(dx, "dX");
        bdk.compare(dk, "dK");
        bdb.compare(db, "db");
    }
    else
        std::cout << "[校验] CPU 校验关闭\n";
    const auto post = Clock::now();
    for(Buffer<T> *buffer : {&bx, &bk, &bb, &bg, &by, &bdx, &bdk, &bdb})
    {
        check(cudaFree(buffer->ptr));
        buffer->ptr = nullptr;
    }
    std::cout << "\n[阶段] free-Mem / postprocess：" << elapsed_ms(post)
              << " ms（GPU 释放；梯度 D2H 比较计入校验阶段）\n";
}

const std::array<std::string, 13> kKernels = {
    "baseline",         "baseline-k128", "baseline-k64",  "baseline-k32", "prefetch",    "opt",           "pytorch",
    "pytorch-backward", "gmem",          "gmem-l2-entry", "cudnn",        "cudnn-fused", "cudnn-separate"};
void usage()
{
    std::cout << "用法：dwconv_test [--kernel baseline|baseline-k128|baseline-k64|baseline-k32|prefetch|opt|pytorch|"
                 "pytorch-backward|gmem|gmem-l2-entry|cudnn|cudnn-fused|cudnn-separate|all]\n"
                 "  [--dtype fp32|fp16|all] [--layout nchw|nhwc] [--case ID|all] [--check-cpu] [--warmup N] "
                 "[--iterations N]\n"
                 "  [--quick] [--list-cases] [--help]\n"
                 "--quick：仅运行前 23 组常规用例；可与 --case、--list-cases 组合。\n"
                 "默认：baseline(FP32/FP16 均按面积分派)、fp32、全部兼容 shape、CPU 校验关闭、预热 20 "
                 "次、计时 100 次。\n"
                 "pytorch / backward / cudnn 支持 FP32/FP16；cudnn 默认 Graph 融合（cudnn-graph 为别名）。\n"
                 "cudnn-fused 为 legacy 融合，cudnn-separate 为分离 bias；layout 默认 nchw，nhwc 仅用于 cuDNN。\n"
                 "gmem 仅 FP32；gmem-l2-entry 只测试计算入口，不配置 L2 驻留。\n";
}

int parse_number(const std::string &text)
{
    size_t used = 0;
    int result = std::stoi(text, &used);
    if(used != text.size() || result < 0)
        throw std::runtime_error("参数必须为非负整数：" + text);
    return result;
}

bool compatible(const ShapeCase &s, const std::string &kernel, const std::string &dtype)
{
    if(is_cudnn(kernel))
    {
#ifdef DWCONV_HAS_CUDNN
        return kernel == "cudnn-separate" || s.bias;
#else
        return false;
#endif
    }
    if(options.layout != "nchw")
        return false;
    if(kernel == "pytorch" || kernel == "pytorch-backward")
        return true;
    if(!s.bias || s.c % (dtype == "fp16" ? 8 : 4) != 0)
        return false;
    if(dtype == "fp16" && (kernel == "gmem" || kernel == "gmem-l2-entry"))
        return false;
    int capacity = kernel == "baseline-k32" ? 32 : kernel == "baseline-k64" ? 64 : 128;
    return s.kh * s.kw <= capacity;
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if(arg == "--help")
            {
                usage();
                return 0;
            }
            if(arg == "--quick")
            {
                options.quick = true;
                continue;
            }
            if(arg == "--check-cpu")
            {
                options.check_cpu = true;
                continue;
            }
            if(arg == "--list-cases")
            {
                options.list = true;
                continue;
            }
            if(arg != "--kernel" && arg != "--dtype" && arg != "--case" && arg != "--layout" && arg != "--warmup" &&
               arg != "--iterations")
                throw std::runtime_error("未知参数：" + arg);
            if(++i == argc)
                throw std::runtime_error("缺少参数值：" + arg);
            std::string val = argv[i];
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
            else
                options.iterations = parse_number(val);
        }
        if(options.kernel == "cudnn-graph")
            options.kernel = "cudnn";
        if(options.layout != "nchw" && options.layout != "nhwc")
            throw std::runtime_error("未知 layout：" + options.layout);
        if(options.layout == "nhwc" && options.kernel != "all" && !is_cudnn(options.kernel))
            throw std::runtime_error("NHWC 目前仅支持 cuDNN 路径，请指定 --kernel cudnn（或 all）");

#ifndef DWCONV_HAS_CUDNN
        if(is_cudnn(options.kernel))
            throw std::runtime_error("未启用 cuDNN；安装开发库后使用 -DDWCONV_TEST_CUDNN=ON 重新配置构建");
#endif

        if(options.kernel != "all" && std::find(kKernels.begin(), kKernels.end(), options.kernel) == kKernels.end())
            throw std::runtime_error("未知 kernel：" + options.kernel);
        if(options.dtype != "fp32" && options.dtype != "fp16" && options.dtype != "all")
            throw std::runtime_error("未知 dtype：" + options.dtype);
        const size_t case_count = options.quick ? kQuickCases.size() : kShapeCases.size();
        if(options.iterations == 0 || options.case_index >= int(case_count))
            throw std::runtime_error("iterations 必须大于 0，case 必须在所选 shape 集合范围内");
        auto print_shape = [](size_t id, const ShapeCase &s)
        {
            std::cout << "[配置] case=" << id << " NCHW=" << s.n << "," << s.c << "," << s.h << "," << s.w
                      << " K=" << s.kh << "x" << s.kw << " stride=" << s.sh << "x" << s.sw << " padding=" << s.ph << "x"
                      << s.pw << " bias=" << s.bias << "\n";
        };
        if(options.list)
        {
            for(size_t i = 0; i < case_count; ++i)
                print_shape(i, kShapeCases[i]);
            return 0;
        }
        cudaDeviceProp device{};
        check(cudaGetDeviceProperties(&device, 0));
        check(cudaFree(nullptr)); // 初始化上下文，不计入各 shape 的准备耗时。
        int runtime = 0;
        check(cudaRuntimeGetVersion(&runtime));
        std::cout << "[设备] " << device.name << " SM " << device.major << "." << device.minor << " CUDA=" << runtime
                  << "\n[计时] 默认流前向 / 非默认流反向；CUDA Event 连续 launch 均值；"
                     "不使用 CUDA Graph，短 kernel 可能包含主机提交间隙。\n";

        int completed = 0, skipped = 0, selected_shapes = 0;

        for(size_t i = 0; i < case_count; ++i)
        {
            if(options.case_index >= 0 && options.case_index != int(i))
                continue;
            ++selected_shapes;
            const auto &s = kShapeCases[i];
            for(const auto &dtype : {std::string("fp32"), std::string("fp16")})
            {
                if(options.dtype != "all" && options.dtype != dtype)
                    continue;
                for(const auto &kernel : kKernels)
                {
                    if(options.kernel != "all" && options.kernel != kernel)
                        continue;
                    if(!compatible(s, kernel, dtype))
                    {
                        ++skipped;
                        continue;
                    }
                    std::cout << "\n";
                    print_shape(i, s);
                    std::cout << "[配置] kernel=" << kernel << " dtype=" << dtype << " layout=" << options.layout
                              << " CPU=" << options.check_cpu << "\n";
#ifdef DWCONV_HAS_CUDNN
                    try
#endif
                    {
                        if(kernel == "pytorch-backward")
                        {
                            if(dtype == "fp16")
                                run_backward<__half>(s);
                            else
                                run_backward<float>(s);
                        }
                        else if(dtype == "fp16")
                            run_forward<__half>(s, kernel);
                        else
                            run_forward<float>(s, kernel);
                        ++completed;
                    }
#ifdef DWCONV_HAS_CUDNN
                    catch(const CudnnUnsupported &e)
                    {
                        ++skipped;
                        std::cout << "[SKIP] " << kernel << ": " << e.what() << "\n";
                    }
#endif
                }
            }
        }
        if(completed == 0)
            throw std::runtime_error("没有兼容的 kernel / dtype / shape 组合");
        std::cout << "\n[SUCCESS] dwconv_test shape 数=" << selected_shapes
                  << " 完成组合数(shape/kernel/dtype)=" << completed << " 不兼容组合跳过=" << skipped
                  << " CPU 校验=" << (options.check_cpu ? "开启" : "关闭，仅确认 CUDA 执行成功") << "\n";
        return 0;
    }
    catch(const std::exception &e)
    {
        std::cout << "\n[失败] " << e.what() << "\n";
        return 1;
    }
}
