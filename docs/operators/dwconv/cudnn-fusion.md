# cuDNN 融合与布局对比

## 实现与口径

三条路径共用 `ops/dwconv/dwconv_cudnn.cu/.cuh`，支持 NCHW/NHWC、groups=C、dilation=1、普通 `[C,1,Kh,Kw]` 权重和相同输入数据：

| 测试选项 | 实现 | 算法选择 |
| --- | --- | --- |
| `cudnn-separate` | `cudnnConvolutionForward` + `cudnnAddTensor` | v7 heuristic 首个成功且符合数学模式的算法 |
| `cudnn-fused` | `cudnnConvolutionBiasActivationForward` | 固定 `IMPLICIT_PRECOMP_GEMM`，activation=IDENTITY，alpha1=1、alpha2=0，z 与 y 共用指针 |
| `cudnn`（默认 cuDNN 路径） | Backend Operation Graph：Conv → ADD | virtual FP32 中间张量，heuristic A / FALLBACK 首个可用 plan |

`cudnn-graph` 是 `cudnn` 的兼容别名，`all` 不重复执行它。`CudnnForward` 构造接口也默认 Graph 模式；整个 `dwconv_test` 仍默认 baseline。前一轮日志中的旧名称 `cudnn` 指分离路径、`cudnn-graph` 指 Graph，下面历史融合数据按实际计算路径标注。

legacy 融合接口的 IDENTITY 模式要求固定算法，且接口自 cuDNN 9.0 起已弃用，见 [官方说明](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html#cudnnconvolutionbiasactivationforward)。Graph 实现直接使用 cuDNN 提供的 C API，无额外 frontend 依赖；图语义参考 [官方 ConvBiasAct 模式](https://docs.nvidia.com/cudnn/developer/graph-api/)。cuDNN operation graph 与 CUDA Graph replay 是不同机制。

FP32 禁用 TF32；Graph 过滤降精度输入、低精度归约及 FP32 Tensor Core engine。FP16 使用 FP32 累加；分离路径在卷积输出时先舍入 half，再加 bias，融合路径允许最后统一舍入。融合模式要求非空 bias；不支持则报告 `[SKIP]`，不会拆图回退。

资源与算法选择、workspace、variant pack、一次试运行及库懒初始化都在 preprocess 阶段；postprocess 释放资源。kernel 阶段保持既有测试口径：CUDA Event 测量连续 API 提交，包括全部内部 kernel 和可能的主机提交空隙，不包括 CPU 校验，不使用 CUDA Graph replay。不进行穷举 autotune。

## 环境与验证

- RTX 5070 Ti Laptop，SM 120，46 SM，L2 36 MiB；CUDA 13.2，驱动 596.21，cuDNN 9.23.1。
- 前一轮 NCHW：23 组 quick shape × FP32/FP16 × 3 路径，共 138 个组合全部通过 CPU 校验，无跳过。
- FP32 对 double 参考容差 `3e-4*(1+abs(ref))`；FP16 对 FP32 累加参考容差 `2e-3*(1+abs(ref))`。
- 下列测量使用 stride=1、padding=K/2、bias 开启；复用同一组输入，缓存不主动清空，未锁频。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDWCONV_TEST_CUDNN=ON
cmake --build build --target dwconv_test device_query --parallel
./build/tools/device_query
for kernel in cudnn-separate cudnn-fused cudnn; do
    ./build/tests/dwconv_test --quick --kernel "$kernel" --dtype all \
        --check-cpu --warmup 0 --iterations 1
done

source /home/if/miniforge3/etc/profile.d/conda.sh
conda activate py311
python benchmarks/operators/dwconv_cudnn_bench.py --kernels cudnn-separate cudnn-fused cudnn \
    --layouts nchw --output build/dwconv-cudnn-fusion/recheck
```

上一轮融合基准复用统一测试，case={7,5,1,0,8,9,22}、每次预热 100、计时 200 次、5 轮；轮换三条路径的执行顺序。原始日志、逐轮 CSV 与汇总保存于 `build/dwconv-cudnn-fusion/perf/`，准备/释放耗时单独存储。每轮独立进程，短用例可能受动态时钟和主机提交影响。此实验比较固定应用 shape，不扫描吞吐平台；cuDNN 内部 grid/block 和寄存器资源由库决定，不推断 waves 或理论驻留。

## NCHW 融合对比（前一轮）

单位 μs，取 5 轮中位数；下降比例相对分离路径，负值表示更慢。这里是 **API 提交的事件区间**，不是各 GPU kernel 时长之和。

| case / NCHW / K | 精度 | 分离 | legacy 融合 | Graph 融合 | legacy / Graph 耗时下降 |
| --- | --- | ---: | ---: | ---: | ---: |
| 7: 1×32×43×43 / 3 | fp32 | 28.43 | 36.58 | 12.03 | -28.7% / 57.7% |
| 7: 1×32×43×43 / 3 | fp16 | 85.79 | 31.81 | 12.49 | 62.9% / 85.4% |
| 5: 1×32×80×80 / 5 | fp32 | 85.50 | 35.73 | 16.72 | 58.2% / 80.4% |
| 5: 1×32×80×80 / 5 | fp16 | 116.71 | 27.77 | 16.89 | 76.2% / 85.5% |
| 1: 4×128×80×80 / 7 | fp32 | 427.57 | 406.48 | 392.68 | 4.9% / 8.2% |
| 1: 4×128×80×80 / 7 | fp16 | 449.43 | 411.03 | 411.62 | 8.5% / 8.4% |
| 0: 32×128×80×80 / 7 | fp32 | 4775.19 | 4281.60 | 4307.42 | 10.3% / 9.8% |
| 0: 32×128×80×80 / 7 | fp16 | 4540.47 | 4251.97 | 4206.52 | 6.4% / 7.4% |
| 8: 1×32×43×43 / 11 | fp32 | 36.29 | 37.21 | 20.84 | -2.5% / 42.6% |
| 8: 1×32×43×43 / 11 | fp16 | 33.16 | 25.83 | 21.91 | 22.1% / 33.9% |
| 9: 32×128×80×80 / 11 | fp32 | 8551.68 | 7968.41 | 7947.57 | 6.8% / 7.1% |
| 9: 32×128×80×80 / 11 | fp16 | 8218.53 | 7897.90 | 7943.61 | 3.9% / 3.3% |
| 22: 1×32×65×81 / 7 | fp32 | 37.24 | 27.17 | 23.47 | 27.0% / 37.0% |
| 22: 1×32×65×81 / 7 | fp16 | 36.31 | 24.74 | 24.27 | 31.9% / 33.2% |

大输入 case 0/9 的融合耗时下降约 **3%–10%**；两种融合方式的中位数相差约 0.3%–1.1%，不足以认定其中一种稳定更快。Graph 在本轮小输入的事件区间通常更短，但不应把这些倍率等同于 GPU 计算加速比。

波动示例（5 轮最小–最大 μs）：

| case / 精度 | 分离 | legacy 融合 | Graph 融合 |
| --- | ---: | ---: | ---: |
| 7 / fp32 | 25.76–100.59 | 19.80–70.00 | 11.37–40.16 |
| 7 / fp16 | 73.04–104.93 | 20.40–84.69 | 10.03–15.26 |
| 0 / fp32 | 4761.36–4856.56 | 4059.86–4381.57 | 4211.61–4432.59 |
| 9 / fp16 | 8188.75–8454.30 | 7657.37–8092.61 | 7826.25–8374.32 |

## 实际 kernel 轨迹

另用 Nsight Systems 2025.6.3 对 case 7/0、两种精度、三条路径独立采样。每个进程执行一次 preprocess 试运行和 3 次计时迭代；以下计数为总实例数除以 4，性能表没有启用 profiler。

| case / 精度 | 分离 kernel/次 | legacy 融合 kernel/次 | Graph 融合 kernel/次 |
| --- | ---: | ---: | ---: |
| 7 / FP32 | 2 | 1 | 1 |
| 7 / FP16 | 6 | 1 | 1 |
| 0 / FP32 | 2 | 1 | 1 |
| 0 / FP16 | 2 | 1 | 1 |

- 分离 FP32 和大输入 FP16：`conv2d_grouped_direct_kernel` + `op_generic_tensor_kernel`（bias）。融合后独立 bias kernel 消失，减少一次输出读写和一次提交。
- case 7 FP16 分离：2 个 `nchwToNhwcKernel` + 1 个 XMMA helper + 1 个 XMMA depthwise 卷积 + 1 个 `nhwcToNchwKernel` + 1 个 bias kernel。两种融合均改用单个 `conv2d_grouped_direct_kernel`，因此收益还包含算法和内部布局转换的变化。
- 在这 4 个 profile 配置中，legacy 与 Graph 融合使用相同的 kernel 符号；分离与融合的 direct kernel 模板实例不同。融合不是严格的“只删除 bias kernel”实验。
- case 7 FP16 的 profile 中，平均每次调用的 GPU kernel 时长之和约为分离 12.78 μs、legacy 4.93 μs、Graph 4.12 μs；它们远小于上面的 API 事件区间。profile 含首次试运行且未锁频，仅用于说明执行结构和提交空隙，不能混入重复测速表或据此认定两种融合算法有稳定差距。

性能样本中，legacy 两条路径均为 algo=1；Graph 均为 engine=0、workspace=0。分离路径的 workspace：case 7 FP16 为 238,046 bytes，case 5 FP16 为 822,622 bytes，其余所测配置为 0；legacy 融合的 workspace 查询返回同样大小，即使实际 kernel 轨迹已不同。因此 algo 编号或 workspace 大小不能代替 profiler 证据。

复现 profiler（替换 case、dtype 和 kernel 可覆盖全部 12 个组合）：

```bash
./build/tools/device_query
nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
    --output build/dwconv-cudnn-fusion/profile-7-fp16-cudnn-graph \
    ./build/tests/dwconv_test --quick --case 7 --kernel cudnn-graph \
    --dtype fp16 --warmup 0 --iterations 3
nsys stats --report cuda_gpu_kern_sum --format csv \
    build/dwconv-cudnn-fusion/profile-7-fp16-cudnn-graph.nsys-rep
```

当前 cuDNN 入口默认 Graph 融合，legacy 融合与分离路径保留为显式对照。


## NHWC 与 NCHW 对比

布局参数只改变物理存储和描述符 stride，逻辑 N/C/H/W、卷积和 bias 不变。Graph 的 X/Y/virtual 张量在 NHWC 下使用 `[H*W*C,1,W*C,C]` stride；depthwise 权重的每组 C=1，所以 KCRS 与 KRSC 实际内存顺序一致。FP32 仍禁用 TF32，避免精度变化影响对比。

`--layout nhwc` 在 preprocess 中进行主机输入重排并单独打印耗时；校验直接读取 NHWC 输出。下表比较原生布局的计算阶段，**不是 NCHW 输入 → GPU 转 NHWC → 卷积 → GPU 转回 NCHW 的端到端时间**。若调用方已经使用 NHWC，则无需这些外部转换。

```bash
./build/tests/dwconv_test --quick --kernel cudnn --layout nhwc --dtype all \
    --check-cpu --warmup 0 --iterations 1
# 默认仅测 Graph + NCHW；布局对比显式启用两种布局
python benchmarks/operators/dwconv_cudnn_bench.py --layouts nchw nhwc
# 可选：同时比较 legacy 融合和分离路径
python benchmarks/operators/dwconv_cudnn_bench.py --kernels cudnn cudnn-fused cudnn-separate \
    --layouts nchw nhwc --output build/dwconv-cudnn-layout/all-modes
```

本轮 Graph NHWC 的 23 组 quick shape × 两种精度，共 **46 个组合通过 CPU 校验**，无跳过。三条 NHWC 路径另外通过 case 7/22 的两种精度抽样校验；NCHW 统一入口和 `cudnn-graph` 别名也通过抽样回归。CMake 新增 `dwconv_cudnn_nhwc_test`。

以下仍为预热 100、计时 200 次、5 轮中位数；7 个 shape × 两种精度 × 两种布局，共 140 次独立进程测量，轮换布局顺序。原始结果在 `build/dwconv-cudnn-layout/perf/`，完整表包含准备、释放、主机重排时间及最小/最大值。单位 μs，倍率为 NHWC/NCHW，**大于 1 表示 NHWC 更慢**。

| case / 逻辑 NCHW / K | 精度 | NCHW | NHWC | NHWC/NCHW |
| --- | --- | ---: | ---: | ---: |
| 7: 1×32×43×43 / 3 | fp32 | 11.19 | 17.91 | 1.60× |
| 7: 1×32×43×43 / 3 | fp16 | 13.27 | 14.85 | 1.12× |
| 5: 1×32×80×80 / 5 | fp32 | 16.93 | 60.47 | 3.57× |
| 5: 1×32×80×80 / 5 | fp16 | 18.54 | 37.32 | 2.01× |
| 1: 4×128×80×80 / 7 | fp32 | 407.81 | 1694.29 | 4.15× |
| 1: 4×128×80×80 / 7 | fp16 | 419.67 | 1712.07 | 4.08× |
| 0: 32×128×80×80 / 7 | fp32 | 4285.40 | 14990.10 | 3.50× |
| 0: 32×128×80×80 / 7 | fp16 | 4203.51 | 14978.70 | 3.56× |
| 8: 1×32×43×43 / 11 | fp32 | 21.55 | 93.10 | 4.32× |
| 8: 1×32×43×43 / 11 | fp16 | 21.50 | 52.48 | 2.44× |
| 9: 32×128×80×80 / 11 | fp32 | 8023.74 | 29972.40 | 3.74× |
| 9: 32×128×80×80 / 11 | fp16 | 7420.13 | 29575.70 | 3.99× |
| 22: 1×32×65×81 / 7 | fp32 | 22.29 | 90.64 | 4.07× |
| 22: 1×32×65×81 / 7 | fp16 | 23.06 | 49.86 | 2.16× |

**当前 heuristic 选择下，NHWC 没有更快。** case 0/9 的大输入耗时约为 NCHW 的 3.50–3.99 倍；case 1 约 4.1 倍。两种布局在全部性能样本中均选择 engine=0、workspace=0，未进行穷举 autotune，不能把该结果推广为所有 cuDNN NHWC 算法都慢。

大输入的 5 轮范围（μs）如下，差异明显大于测量波动：

| case / 精度 | NCHW min–max | NHWC min–max |
| --- | ---: | ---: |
| 0 / fp32 | 4256.18–4498.70 | 14642.00–16391.10 |
| 0 / fp16 | 4152.89–4419.38 | 14713.80–15575.20 |
| 9 / fp32 | 7046.58–8102.19 | 29908.40–30847.80 |
| 9 / fp16 | 7091.62–7635.76 | 29442.60–30301.40 |

case 0 的主机输入重排中位数为 FP32 110.29 ms、FP16 60.47 ms，已包含在 preprocess 中，未进入上表。它是本测试的 CPU 准备成本，不代表实际应用中的 GPU 转置成本。

为检查是否只与 Graph 接口有关，另对 case 7/0 做了 3 轮 legacy 融合与分离路径的布局对照（预热 100、计时 200，结果在 `build/dwconv-cudnn-layout/controls/`）。case 0 的中位数如下，同样是 NHWC 更慢：

| 精度 / 路径 | NCHW μs | NHWC μs | NHWC/NCHW |
| --- | ---: | ---: | ---: |
| fp32 / cudnn-separate | 4809.32 | 15357.80 | 3.19× |
| fp32 / cudnn-fused | 4219.71 | 14551.20 | 3.45× |
| fp16 / cudnn-separate | 4720.36 | 15279.50 | 3.24× |
| fp16 / cudnn-fused | 4182.69 | 14817.70 | 3.54× |

Nsight Systems 对 case 7/0 × 两种精度 × 两种布局独立采样：**Graph 每次调用都是一个 `conv2d_grouped_direct_kernel`**，布局不同对应不同模板实例；没有额外 GPU 转置或独立 bias kernel，也未选到 XMMA depthwise kernel。下表为 profiler 中 4 次调用（一次试运行 + 三次计时）的 GPU kernel 时长中位数，仅辅助判断执行结构，不与上面的 5 轮无 profiler 测量混用。

| case / 精度 | NCHW kernel μs | NHWC kernel μs |
| --- | ---: | ---: |
| 7 / fp32 | 4.10 | 11.66 |
| 7 / fp16 | 4.15 | 7.93 |
| 0 / fp32 | 2667.29 | 13873.72 |
| 0 / fp16 | 2942.57 | 13958.93 |

因此，可以排除外部转换被算进 kernel 时间这一解释；当前差异发生在所选 direct kernel 的布局实现上。这里没有测量访存事务、occupancy 等指标，不进一步归因为某个具体硬件瓶颈，也没有验证所有候选 engine。保留 **Graph + NCHW** 为默认 cuDNN 配置，NHWC 通过参数显式选择。

复现补充对照与 profiler：

```bash
python benchmarks/operators/dwconv_cudnn_bench.py --cases 7 0 \
    --kernels cudnn-separate cudnn-fused --layouts nchw nhwc --rounds 3 \
    --output build/dwconv-cudnn-layout/controls
./build/tools/device_query
nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
    --output build/dwconv-cudnn-layout/profile-0-fp16-nhwc \
    ./build/tests/dwconv_test --quick --case 0 --kernel cudnn --layout nhwc \
    --dtype fp16 --warmup 0 --iterations 3
nsys stats --report cuda_gpu_kern_sum --format csv \
    build/dwconv-cudnn-layout/profile-0-fp16-nhwc.nsys-rep
```
