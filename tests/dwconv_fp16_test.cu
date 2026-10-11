#include <dwconv_fp16.cuh>
#include <cuda_lab/cuda_check.cuh>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace
{
template <typename T> struct DeviceBuffer
{
    T *ptr = nullptr;
    explicit DeviceBuffer(size_t count)
    {
        cuda_lab::check_cuda(cudaMalloc(&ptr, count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer()
    {
        cudaFree(ptr);
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

void run_case(int n, int c, int h, int w, int kh, int kw, int sh, int sw, int ph, int pw)
{
    Conv2DParam p{};
    p.in_h = h;
    p.in_w = w;
    p.in_ch = c;
    p.out_ch = c;
    p.Kh = kh;
    p.Kw = kw;
    p.KhKw = kh * kw;
    p.Sh = sh;
    p.Sw = sw;
    p.Ph = ph;
    p.Pw = pw;
    p.inHW = h * w;
    p.inBatchNumel = c * p.inHW;
    p.out_h = (h + 2 * ph - kh) / sh + 1;
    p.out_w = (w + 2 * pw - kw) / sw + 1;
    p.outHW = p.out_h * p.out_w;
    p.outBatchNumel = c * p.outHW;
    std::cout << "\n[配置] NCHW=" << n << "," << c << "," << h << "," << w << " kernel=" << kh << "x" << kw
              << " stride=" << sh << "x" << sw << " padding=" << ph << "x" << pw << "\n";
    std::vector<__half> input(n * p.inBatchNumel), weights(c * p.KhKw), bias(c);
    std::vector<__half> expected(n * p.outBatchNumel), actual(expected.size());
    for(size_t i = 0; i < input.size(); ++i)
        input[i] = __float2half_rn((int(i % 251) - 125) / 509.0f);
    for(size_t i = 0; i < weights.size(); ++i)
        weights[i] = __float2half_rn((int(i % 127) - 63) / 257.0f);
    for(size_t i = 0; i < bias.size(); ++i)
        bias[i] = __float2half_rn((int(i) - 3) / 97.0f);
    std::cout << "\n[阶段] CPU depthwise 参考计算\n";
    for(int b = 0; b < n; ++b)
        for(int ch = 0; ch < c; ++ch)
            for(int y = 0; y < p.out_h; ++y)
                for(int x = 0; x < p.out_w; ++x)
                {
                    __half sum = bias[ch];
                    for(int ky = 0; ky < kh; ++ky)
                        for(int kx = 0; kx < kw; ++kx)
                        {
                            int iy = y * sh + ky - ph;
                            int ix = x * sw + kx - pw;
                            if(iy >= 0 && iy < h && ix >= 0 && ix < w)
                                sum = __double2half(std::fma(
                                    static_cast<double>(__half2float(input[(b * c + ch) * h * w + iy * w + ix])),
                                    static_cast<double>(__half2float(weights[ch * kh * kw + ky * kw + kx])),
                                    static_cast<double>(__half2float(sum))));
                        }
                    expected[(b * c + ch) * p.outHW + y * p.out_w + x] = sum;
                }
    std::vector<__half2> packed(weights.size() / 2);
    for(int pair = 0; pair < c / 2; ++pair)
        for(int k = 0; k < p.KhKw; ++k)
            packed[pair * p.KhKw + k] =
                __halves2half2(weights[(2 * pair) * p.KhKw + k], weights[(2 * pair + 1) * p.KhKw + k]);
    DeviceBuffer<__half> di(input.size()), db(bias.size()), dout(actual.size());
    DeviceBuffer<__half2> dw(packed.size());
    cuda_lab::check_cuda(cudaMemcpy(di.ptr, input.data(), input.size() * sizeof(__half), cudaMemcpyHostToDevice),
                         "上传输入");
    cuda_lab::check_cuda(cudaMemcpy(dw.ptr, packed.data(), packed.size() * sizeof(__half2), cudaMemcpyHostToDevice),
                         "上传权重");
    cuda_lab::check_cuda(cudaMemcpy(db.ptr, bias.data(), bias.size() * sizeof(__half), cudaMemcpyHostToDevice),
                         "上传偏置");
    const char *names[] = {"k128 base-fp16", "k64 base-fp16", "k32 base-fp16", "k128 prefetch-fp16"};
    const int capacities[] = {128, 64, 32, 128};
    using Launch =
        void (*)(const __half *, const __half2 *, const __half *, __half *, const Conv2DParam &, int, cudaStream_t);
    const Launch launches[] = {launch_baseline_fp16_k128, launch_baseline_fp16_k64, launch_baseline_fp16_k32,
                               launch_prefetch_fp16_k128};
    for(int variant = 0; variant < 4; ++variant)
    {
        if(p.KhKw > capacities[variant])
            continue;
        std::cout << "\n[阶段] " << names[variant] << " 正确性验证\n";
        cuda_lab::check_cuda(cudaMemset(dout.ptr, 0xff, actual.size() * sizeof(__half)), "初始化输出");
        launches[variant](di.ptr, dw.ptr, db.ptr, dout.ptr, p, n, nullptr);
        cuda_lab::check_cuda(cudaGetLastError(), "启动 kernel");
        cuda_lab::check_cuda(cudaDeviceSynchronize(), "执行 kernel");
        cuda_lab::check_cuda(
            cudaMemcpy(actual.data(), dout.ptr, actual.size() * sizeof(__half), cudaMemcpyDeviceToHost), "读取输出");
        float max_error = 0;
        for(size_t i = 0; i < actual.size(); ++i)
        {
            float error = std::abs(__half2float(actual[i]) - __half2float(expected[i]));
            if(!std::isfinite(__half2float(actual[i])) || error != 0.0f)
                throw std::runtime_error("dwconv FP16 结果不匹配，index=" + std::to_string(i));
            max_error = std::max(max_error, error);
        }
        std::cout << "[结果] 元素数=" << actual.size() << " 最大绝对误差=" << max_error << " [SUCCESS]\n";
    }
}
} // namespace
int main()
{
    try
    {
        cudaDeviceProp device{};
        cuda_lab::check_cuda(cudaGetDeviceProperties(&device, 0), "查询设备");
        std::cout << "[设备] " << device.name << " SM " << device.major << "." << device.minor << "\n";
        run_case(1, 8, 1, 1, 1, 1, 1, 1, 0, 0);
        run_case(2, 16, 17, 19, 3, 3, 1, 1, 1, 1);
        run_case(1, 24, 23, 29, 5, 3, 2, 1, 2, 1);
        run_case(1, 8, 16, 16, 2, 2, 1, 1, 0, 0);
        run_case(1, 8, 16, 16, 1, 1, 1, 1, 0, 0);
        run_case(2, 16, 19, 37, 1, 31, 1, 2, 0, 15);
        run_case(2, 16, 19, 23, 4, 8, 2, 1, 1, 3);
        run_case(2, 16, 19, 23, 7, 9, 1, 2, 3, 4);
        run_case(2, 16, 19, 23, 8, 8, 2, 1, 3, 3);
        run_case(1, 8, 19, 23, 11, 11, 1, 1, 5, 5);
        run_case(1, 8, 19, 23, 8, 16, 1, 2, 0, 0);
        run_case(2, 16, 23, 29, 5, 5, 2, 1, 2, 2);
        run_case(2, 16, 23, 29, 7, 7, 1, 2, 3, 3);
        // 9x9 分派：pair_blocks 分别为 136 和 140。
        run_case(1, 8, 64, 136, 9, 9, 1, 1, 4, 4);
        run_case(1, 8, 64, 140, 9, 9, 1, 1, 4, 4);
        run_case(2, 16, 43, 47, 9, 9, 2, 2, 4, 4);
        run_case(2, 16, 43, 47, 11, 11, 2, 2, 5, 5);
        // 超出共享输入容量时回退到 pair / specialized 路径。
        run_case(1, 8, 192, 420, 9, 9, 3, 3, 4, 4);
        run_case(1, 8, 43, 47, 11, 11, 3, 3, 5, 5);
        std::cout << "\n[SUCCESS] dwconv_fp16_test\n";
        return 0;
    }
    catch(const std::exception &e)
    {
        std::cout << "\n[失败] " << e.what() << "\n";
        return 1;
    }
}
