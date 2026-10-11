#include <dwconv.cuh>
#include <cuda_lab/cuda_check.cuh>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace
{
bool prefetch_only = false;
struct DeviceBuffer
{
    float *ptr = nullptr;
    explicit DeviceBuffer(size_t count)
    {
        cuda_lab::check_cuda(cudaMalloc(&ptr, count * sizeof(float)), "cudaMalloc");
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
    std::vector<float> input(n * p.inBatchNumel), weights(c * p.KhKw), bias(c);
    std::vector<float> expected(n * p.outBatchNumel), actual(expected.size());
    for(size_t i = 0; i < input.size(); ++i)
        input[i] = (int(i % 29) - 14) / 16.0f;
    for(size_t i = 0; i < weights.size(); ++i)
        weights[i] = (int(i % 17) - 8) / 32.0f;
    for(size_t i = 0; i < bias.size(); ++i)
        bias[i] = (int(i) - 3) / 8.0f;
    std::cout << "\n[阶段] CPU depthwise 参考计算\n";
    for(int b = 0; b < n; ++b)
        for(int ch = 0; ch < c; ++ch)
            for(int y = 0; y < p.out_h; ++y)
                for(int x = 0; x < p.out_w; ++x)
                {
                    float sum = bias[ch];
                    for(int ky = 0; ky < kh; ++ky)
                        for(int kx = 0; kx < kw; ++kx)
                        {
                            int iy = y * sh + ky - ph;
                            int ix = x * sw + kx - pw;
                            if(iy >= 0 && iy < h && ix >= 0 && ix < w)
                                sum += input[(b * c + ch) * h * w + iy * w + ix] * weights[ch * kh * kw + ky * kw + kx];
                        }
                    expected[(b * c + ch) * p.outHW + y * p.out_w + x] = sum;
                }
    DeviceBuffer di(input.size()), dw(weights.size()), db(bias.size()), dout(actual.size());
    cuda_lab::check_cuda(cudaMemcpy(di.ptr, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice),
                         "上传输入");
    cuda_lab::check_cuda(cudaMemcpy(dw.ptr, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice),
                         "上传权重");
    cuda_lab::check_cuda(cudaMemcpy(db.ptr, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice),
                         "上传偏置");
    const char *names[] = {"k128 基线",           "k64 基线",      "k32 基线",
                           "k128 输入寄存器预取", "k128 全局权重", "k128 全局权重 L2 入口"};
    const int capacities[] = {128, 64, 32, 128, 128, 128};
    for(int variant = 0; variant < 6; ++variant)
    {
        if(prefetch_only && variant != 3)
            continue;
        if(p.KhKw > capacities[variant])
            continue;
        std::cout << "\n[阶段] " << names[variant] << " 正确性验证\n";
        cuda_lab::check_cuda(cudaMemset(dout.ptr, 0xff, actual.size() * sizeof(float)), "初始化输出");
        dim3 grid((p.outHW + 255) / 256, c / 4, n);
        if(variant == 5)
            conv2d_4x128x256_groups_kernel_gmem_l2<<<grid, 256>>>(di.ptr, dw.ptr, db.ptr, dout.ptr, p);
        else if(variant == 4)
            conv2d_4x128x256_groups_kernel_gmem<<<grid, 256>>>(di.ptr, dw.ptr, db.ptr, dout.ptr, p);
        else if(variant == 3)
            conv2d_4x128x256_groups_kernel_prefetch<<<grid, 256>>>(di.ptr, dw.ptr, db.ptr, dout.ptr, p);
        else if(variant == 1)
            conv2d_4x64x256_groups_kernel<<<grid, 256>>>(di.ptr, dw.ptr, db.ptr, dout.ptr, p);
        else if(variant == 2)
            conv2d_4x32x256_groups_kernel<<<grid, 256>>>(di.ptr, dw.ptr, db.ptr, dout.ptr, p);
        else
            conv2d_4x128x256_groups_kernel<<<grid, 256>>>(di.ptr, dw.ptr, db.ptr, dout.ptr, p);
        cuda_lab::check_cuda(cudaGetLastError(), "启动 kernel");
        cuda_lab::check_cuda(cudaDeviceSynchronize(), "执行 kernel");
        cuda_lab::check_cuda(cudaMemcpy(actual.data(), dout.ptr, actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
                             "读取输出");
        float max_error = 0;
        for(size_t i = 0; i < actual.size(); ++i)
        {
            float error = std::abs(actual[i] - expected[i]);
            if(!std::isfinite(actual[i]) || error > 1e-5f + 1e-5f * std::abs(expected[i]))
                throw std::runtime_error("dwconv 结果不匹配，index=" + std::to_string(i));
            max_error = std::max(max_error, error);
        }
        std::cout << "[结果] 元素数=" << actual.size() << " 最大绝对误差=" << max_error << " [SUCCESS]\n";
    }
}
} // namespace
int main(int argc, char **argv)
{
    try
    {
        if(argc == 2 && std::string(argv[1]) == "--prefetch-only")
            prefetch_only = true;
        else if(argc != 1)
            throw std::runtime_error("用法：dwconv_test [--prefetch-only]");
        cudaDeviceProp device{};
        cuda_lab::check_cuda(cudaGetDeviceProperties(&device, 0), "查询设备");
        std::cout << "[设备] " << device.name << " SM " << device.major << "." << device.minor << "\n";
        run_case(1, 4, 1, 1, 1, 1, 1, 1, 0, 0);
        run_case(2, 8, 17, 19, 3, 3, 1, 1, 1, 1);
        run_case(1, 12, 23, 29, 5, 3, 2, 1, 2, 1);
        run_case(1, 4, 16, 16, 2, 2, 1, 1, 0, 0);
        run_case(1, 8, 16, 16, 1, 1, 1, 1, 0, 0);
        run_case(2, 8, 19, 37, 1, 31, 1, 2, 0, 15);
        run_case(2, 8, 19, 23, 4, 8, 2, 1, 1, 3);
        run_case(2, 8, 19, 23, 7, 9, 1, 2, 3, 4);
        run_case(2, 8, 19, 23, 8, 8, 2, 1, 3, 3);
        run_case(1, 4, 19, 23, 11, 11, 1, 1, 5, 5);
        run_case(1, 4, 19, 23, 8, 16, 1, 2, 0, 0);
        run_case(1, 8, 23, 29, 5, 5, 2, 1, 2, 2);
        // 7x7 分派：grouped_blocks 分别为 100 和 104。
        run_case(1, 16, 64, 100, 7, 7, 1, 1, 3, 3);
        run_case(1, 16, 64, 104, 7, 7, 1, 1, 3, 3);
        run_case(1, 8, 23, 29, 9, 9, 2, 1, 4, 4);
        // 11x11 共享输入路径及 stride=3 超出 48 KiB 后的回退路径。
        run_case(2, 8, 43, 47, 11, 11, 2, 2, 5, 5);
        run_case(1, 8, 43, 47, 11, 11, 3, 3, 5, 5);
        std::cout << "\n[SUCCESS] dwconv_test\n";
        return 0;
    }
    catch(const std::exception &e)
    {
        std::cout << "\n[失败] " << e.what() << "\n";
        return 1;
    }
}
