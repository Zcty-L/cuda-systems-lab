#pragma once
#include <dwconv/common.h>
#include <array>
#include <vector>

// 基准用例及参数构建；case 编号供 C++ 入口和批量脚本共同使用。
namespace dwconv_bench
{

struct ShapeCase
{
    int n, c, h, w, kh, kw, sh, sw, ph, pw;
    bool bias;
};

inline constexpr std::array<ShapeCase, 23> kQuickCases = {{
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

inline std::vector<ShapeCase> make_shape_cases()
{
    constexpr int batches[] = {1, 2, 4, 8, 16, 32, 64};
    constexpr int channels[] = {32, 64, 128, 256, 512, 1024};
    constexpr int spatial_sizes[] = {40, 80, 160};
    constexpr int kernel_sizes[] = {3, 5, 7, 9, 11};
    std::vector<ShapeCase> cases(kQuickCases.begin(), kQuickCases.end());

    // 7*6*3*5=630
    for(int k : kernel_sizes)
        for(int n : batches)
            for(int c : channels)
                for(int h : spatial_sizes)
                    cases.push_back({n, c, h, h, k, k, 1, 1, k / 2, k / 2, true});
    return cases;
}

inline Conv2DParam make_param(const ShapeCase &s)
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
    p.use_bias = s.bias;
    return p;
}

} // namespace dwconv_bench
