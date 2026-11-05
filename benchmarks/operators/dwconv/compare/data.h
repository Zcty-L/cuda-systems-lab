#pragma once

#include "common.h"

// 基准内部的数据准备与 CPU 参考，均在正式计时区间之外执行。
namespace dwconv_bench
{

template <class T> T convert(float input)
{
    if constexpr(std::is_same_v<T, __half>)
        return __float2half_rn(input);
    else
        return input;
}

template <class T> float value(T input)
{
    if constexpr(std::is_same_v<T, __half>)
        return __half2float(input);
    else
        return input;
}

template <class T> struct DeviceBuffer
{
    T *ptr = nullptr;
    size_t count;

    explicit DeviceBuffer(size_t size) : count(size)
    {
        if(count)
            check(cudaMalloc(&ptr, count * sizeof(T)));
    }

    ~DeviceBuffer()
    {
        cudaFree(ptr);
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    void upload(const std::vector<T> &input)
    {
        check(cudaMemcpy(ptr, input.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void release()
    {
        check(cudaFree(ptr));
        ptr = nullptr;
    }
};

template <class T> struct ForwardHostData
{
    using Weight = std::conditional_t<std::is_same_v<T, __half>, __half2, float>;

    Conv2DParam param;
    std::vector<T> input, weights, bias, layout_input;
    std::vector<Weight> packed;
    double layout_ms = 0;

    ForwardHostData(const ShapeCase &shape, const Backend &backend, const Options &options)
        : param(make_param(shape)), input(size_t(shape.n) * param.inBatchNumel), weights(shape.c * param.KhKw),
          bias(shape.c)
    {
        for(size_t i = 0; i < input.size(); ++i)
            input[i] = convert<T>((int(i % 251) - 125) / 509.0f);
        for(size_t i = 0; i < weights.size(); ++i)
            weights[i] = convert<T>((int(i % 127) - 63) / 257.0f);
        for(size_t i = 0; i < bias.size(); ++i)
            bias[i] = convert<T>((int(i % 17) - 8) / 97.0f);

        if constexpr(std::is_same_v<T, __half>)
        {
            if(backend.fp16_packed_weights)
            {
                packed.resize(weights.size() / 2);
                for(int pair = 0; pair < shape.c / 2; ++pair)
                    for(int k = 0; k < param.KhKw; ++k)
                        packed[pair * param.KhKw + k] = __halves2half2(weights[2 * pair * param.KhKw + k],
                                                                       weights[(2 * pair + 1) * param.KhKw + k]);
            }
        }
        else if(!backend.host_weights)
            packed = weights;

        if(options.layout == "nhwc")
        {
            const auto start = Clock::now();
            layout_input.resize(input.size());
            for(int n = 0; n < shape.n; ++n)
                for(int hw = 0; hw < param.inHW; ++hw)
                    for(int c = 0; c < shape.c; ++c)
                        layout_input[(size_t(n) * param.inHW + hw) * shape.c + c] =
                            input[(size_t(n) * shape.c + c) * param.inHW + hw];
            layout_ms = elapsed_ms(start);
        }
    }

    size_t device_weight_count(const Backend &backend) const
    {
        return backend.host_weights          ? 0
               : backend.fp16_packed_weights ? packed.size()
                                             : (weights.size() * sizeof(T) + sizeof(Weight) - 1) / sizeof(Weight);
    }
};

template <class T> struct ForwardData
{
    using Weight = typename ForwardHostData<T>::Weight;

    ForwardHostData<T> host;
    const size_t count;
    DeviceBuffer<T> x, b, y;
    DeviceBuffer<Weight> w;

    ForwardData(const ShapeCase &shape, const Backend &backend, const Options &options, bool poison_output)
        : host(shape, backend, options), count(size_t(shape.n) * host.param.outBatchNumel), x(host.input.size()),
          b(backend.host_weights ? 0 : host.bias.size()), y(count), w(host.device_weight_count(backend))
    {
        x.upload(options.layout == "nhwc" ? host.layout_input : host.input);
        if(!backend.host_weights)
        {
            b.upload(host.bias);
            if(backend.fp16_packed_weights)
                w.upload(host.packed);
            else
                check(cudaMemcpy(w.ptr, host.weights.data(), host.weights.size() * sizeof(T), cudaMemcpyHostToDevice));
        }
        if(poison_output)
            check(cudaMemset(y.ptr, 0xff, count * sizeof(T)));

        // TRT 使用非阻塞流，准备后端前先完成默认流上的上传和输出初始化。
        check(cudaStreamSynchronize(nullptr));
    }

    // 返回视图引用本对象和调用方配置；后端释放前必须保持这些对象有效。
    ForwardCase make_case(const ShapeCase &shape, const Options &options, const std::string &kernel) const
    {
        return {shape,
                host.param,
                options,
                kernel,
                std::is_same_v<T, __half>,
                x.ptr,
                w.ptr,
                shape.bias ? b.ptr : nullptr,
                y.ptr,
                host.weights.data(),
                shape.bias ? host.bias.data() : nullptr};
    }

    std::vector<T> download() const
    {
        std::vector<T> actual(count);
        check(cudaMemcpy(actual.data(), y.ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        return actual;
    }

    void release()
    {
        x.release();
        w.release();
        b.release();
        y.release();
    }
};

template <class T>
void validate_forward(const ShapeCase &s, const Conv2DParam &p, const std::vector<T> &input,
                      const std::vector<T> &weights, const std::vector<T> &bias, const std::vector<T> &actual,
                      double fp16_tolerance = 0, bool nhwc = false)
{
    const auto start = Clock::now();
    double max_error = 0;
    std::cout << "[校验] abs(error) <= " << (std::is_same_v<T, __half> ? fp16_tolerance : 3e-4)
              << " * (1 + abs(ref))\n";

    for(int b = 0; b < s.n; ++b)
        for(int c = 0; c < s.c; ++c)
            for(int y = 0; y < p.out_h; ++y)
                for(int x = 0; x < p.out_w; ++x)
                {
                    double sum = s.bias ? value(bias[c]) : 0;
                    for(int ky = 0; ky < s.kh; ++ky)
                    {
                        for(int kx = 0; kx < s.kw; ++kx)
                        {
                            int iy = y * s.sh + ky - s.ph;
                            int ix = x * s.sw + kx - s.pw;
                            if(iy < 0 || iy >= s.h || ix < 0 || ix >= s.w)
                                continue;
                            double a = value(input[(b * s.c + c) * p.inHW + iy * s.w + ix]);
                            double w = value(weights[c * p.KhKw + ky * s.kw + kx]);
                            if constexpr(std::is_same_v<T, __half>)
                                sum = std::fma(float(a), float(w), float(sum));
                            else
                                sum = std::fma(a, w, sum);
                        }
                    }

                    if constexpr(std::is_same_v<T, __half>)
                        sum = __half2float(__double2half(sum));

                    size_t index = nhwc ? ((size_t(b) * p.out_h + y) * p.out_w + x) * s.c + c
                                        : (size_t(b) * s.c + c) * p.outHW + y * p.out_w + x;
                    double got = value(actual[index]), error = std::abs(got - sum);

                    double tolerance =
                        std::is_same_v<T, __half> ? fp16_tolerance * (1 + std::abs(sum)) : 3e-4 * (1 + std::abs(sum));
                    if(!std::isfinite(got) || error > tolerance)
                        throw std::runtime_error("CPU 校验失败 index=" + std::to_string(index) +
                                                 " actual=" + std::to_string(got) + " expected=" + std::to_string(sum) +
                                                 " tolerance=" + std::to_string(tolerance));
                    max_error = std::max(max_error, error);
                }
    std::cout << "\n[校验] CPU 最大绝对误差=" << max_error << "，耗时=" << elapsed_ms(start) << " ms [SUCCESS]\n";
}

} // namespace dwconv_bench
