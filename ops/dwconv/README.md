# Depthwise Conv2D


直接导入 `/home/if/Codes/CUDAOP/op/dwconv/` 下的三个 FP32 基线 kernel，仅调整名称、依赖与书写格式，保留计算逻辑。

| 角色 | 原始位置 | 当前名称 |
| --- | --- | --- |
| 基线 | `dwconv_k128.cu` 第 178 行 | `conv2d_4x128x256_groups_kernel` |
| 基线 | `dwconv_k64.cu` 第 174 行 | `conv2d_4x64x256_groups_kernel` |
| 基线 | `dwconv_k32.cu` 第 176 行 | `conv2d_4x32x256_groups_kernel` |

三个基线一次加载每组四个通道的全部权重，使用 warp shuffle 广播 bias。当前 FP32 k128/k64/k32 仅在 shared 权重写入后同步一次，后续只读循环不再同步；FP16 baseline 与 prefetch 原本也采用这一同步方式。k32、k64 在卷积累加后加 bias，k128 以 bias 初始化累加器。另提供 k128 输入寄存器预取版本 `prefetch`，详见下文。

## 调用约束

- 输入、输出为连续 NCHW FP32；权重为连续 `[C, Kh, Kw]`，bias 为 `[C]`。
- `groups = in_ch = out_ch`，通道倍增系数为 1，通道数是 4 的倍数。
- k32 基线要求 `1 <= Kh * Kw <= 32`，k64 基线要求 `1 <= Kh * Kw <= 64`，k128 基线和 prefetch共同支持 `1 <= Kh * Kw <= 128`；dilation 为 1，尺寸及 stride 为正，输出尺寸为正。
- 调用方填写 `Conv2DParam` 的尺寸、步长、padding 与派生元素数，确保索引运算不溢出且 grid 不超设备限制。
- 固定 `block=(256,1,1)`，`grid=(ceil(outHW/256), C/4, N)`。接口直接暴露 kernel，不提供参数校验层。
- `Conv2DParam` 所有字段统一为 `int`；尺寸、坐标、元素数及索引中间值须在有符号 32 位范围内，kernel 未使用的字段可置零。无需 cuDNN。

## 构建与正确性验证

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target dwconv_test device_query --parallel
./build/tools/device_query
ctest --test-dir build -R '^dwconv_test$' --output-on-failure
```

`tests/dwconv_test.cu` 是统一入口，全局 `std::vector<ShapeCase>` 由函数构建：编号 0–22 为 `opt_conv2d_groups_fp32.cu` 的 23 组常规用例，stride 统一改为 1，保留原有输入尺寸、核尺寸和通道数（原名称中的输出尺寸及分派阈值不再适用）；编号 23–652 为 N={1,2,4,8,16,32,64}、C={32,64,128,256,512,1024}、H=W={40,80,160}、Kh=Kw={3,5,7,9,11} 的全部 630 种组合，stride=1、padding=K/2，启用 bias；N、C、H、K 取值覆盖 CUDAOP 的 `dwconv_k128.cu` 完整组合。所有实现扫描同一张表，按通道对齐、核容量和 bias 支持筛选；`--case ID` 可固定同一 shape 比较不同实现。通过 `--list-cases` 查询编号。`--quick` 仅选择前 23 组常规用例，CTest 使用此模式；默认仍遍历全部 653 组。全量 CPU 校验耗时较长，可用 `--quick` 或 `--case ID` 缩小范围。

```bash
# 默认：FP32 baseline 按核面积分派，CPU 校验关闭，预热 20 次、计时 100 次
./build/tests/dwconv_test
./build/tests/dwconv_test --list-cases
./build/tests/dwconv_test --kernel opt --dtype fp16 --case 1
./build/tests/dwconv_test --quick --kernel cudnn --layout nhwc --dtype all --check-cpu --warmup 1 --iterations 2
# legacy 融合对照
./build/tests/dwconv_test --kernel cudnn-fused --layout nhwc --dtype fp16 --case 7
./build/tests/dwconv_test --kernel prefetch --dtype fp32 --warmup 20 --iterations 100
# 快速测试全部兼容实现；开启独立 CPU 校验（与 CTest 一致）
./build/tests/dwconv_test --quick --kernel all --dtype all --check-cpu --warmup 0 --iterations 1
./build/tests/dwconv_test --kernel pytorch-backward --dtype fp32 --check-cpu --iterations 1
```

`--kernel` 支持 `baseline`（FP32/FP16 均按核面积选择 k32/k64/k128）、`baseline-k128`、`baseline-k64`、`baseline-k32`、`prefetch`、`opt`、`pytorch`（前向）、`pytorch-backward`、`gmem`、`gmem-l2-entry`、`cudnn`、`cudnn-fused`、`cudnn-separate` 或 `all`；`--dtype` 支持 `fp32`、`fp16` 或 `all`。PyTorch 前向、反向均支持 FP32/FP16，gmem 仅支持 FP32。不兼容组合跳过并汇总，完全没有可执行组合时报错。`gmem-l2-entry` 仅覆盖计算入口；实际 L2 策略对比仍使用 `dwconv_weights_bench`。

cuDNN 的 FP32/FP16 计算封装在 [dwconv_cudnn.cu](dwconv_cudnn.cu) / [dwconv_cudnn.cuh](dwconv_cudnn.cuh)，由可选目标 `cuda_ops_dwconv_cudnn` 构建：

| `--kernel` | 计算路径 |
| --- | --- |
| `cudnn-separate` | 卷积 + 独立 bias；v7 heuristic 选算法 |
| `cudnn-fused` | `cudnnConvolutionBiasActivationForward`，IDENTITY + 固定 `IMPLICIT_PRECOMP_GEMM` |
| `cudnn`（默认 cuDNN 路径） | Backend Graph API 的 Conv → ADD，virtual FP32 中间张量；heuristic A / FALLBACK 选择首个可用融合 plan |

`--kernel cudnn` 与 `CudnnForward` 构造接口默认使用 Graph 融合，`cudnn-graph` 保留为别名；`--kernel all` 不重复执行别名。整个测试程序的默认 kernel 仍为 baseline。

三者均支持 `--layout nchw|nhwc`（默认 nchw），保持 groups=C 和相同逻辑输入。测试在主机准备阶段将 NCHW 数据重排为 NHWC，并单独打印该耗时；CPU 校验直接按 NHWC 索引读取输出。计算阶段输入输出均采用指定的原生布局，不包含外部 GPU 布局转换。depthwise 的每组输入通道为 1，KCRS / KRSC 权重内存顺序相同，无需重排。NHWC 目前仅适用于 cuDNN；其他实现不会被当作 NHWC 执行。FP32 禁用 TF32；FP16 使用 FP32 累加。融合模式要求 bias，不支持的配置输出 `[SKIP]`，不回退为分离计算。Graph 路径直接使用 cuDNN C API，无额外 frontend 依赖；它不是 CUDA Graph replay。

`CudnnForward` 构造时选择算法/plan 并分配 workspace，`bind` 准备 Graph 的 variant pack，`launch` 异步执行，析构释放资源。测试的 preprocess 还包含一次试运行以验证支持情况、完成懒初始化；所有这些均不计入 kernel 时间。输出包括版本、算法/engine 编号和 workspace。cuDNN FP16 使用 `2e-3 * (1 + abs(ref))` 校验容差，允许归约及 bias 舍入差异，其余实现的规则不变。

CMake 默认 `DWCONV_TEST_CUDNN=ON`；缺少开发库时其余实现仍可构建，显式选择 cuDNN 路径会报错，`all` 跳过它们。可用 `-DDWCONV_TEST_CUDNN=OFF` 关闭。

融合与布局的正确性、重复测量和 profiler 结果见 [cuDNN 融合与布局对比](../../docs/operators/dwconv/cudnn-fusion.md)，复测入口为 [dwconv_cudnn_bench.py](../../benchmarks/operators/dwconv_cudnn_bench.py)。

分阶段计时：

- **GPU-Mem / preprocess**：主机计时，包含数据生成、FP16 权重配对、GPU 分配与 H2D；CUDA 上下文初始化在阶段之前完成。
- **kernel launch**：预热后使用 CUDA Event 记录连续启动的总 GPU 时间并除以迭代次数。前向使用默认流，反向使用非默认流，每次反向包含 dX、dK、db 三个 kernel。不使用 CUDA Graph，短 kernel 可能包含主机提交间隙。
- **free-Mem / postprocess**：主机计时，包含 GPU 释放，前向启用校验时还包含 D2H。反向的 D2H 比较归入独立校验阶段。主机容器析构不计入该阶段。

默认不计算 CPU 参考，也不下载输出。`--check-cpu` 开启独立参考计算并单独报告耗时，不计入 kernel 时间：FP32 使用 double 参考、误差限 `3e-4 * (1 + abs(ref))`；baseline/opt/prefetch/PyTorch FP16 前向以 FP32 FMA 累加、最后舍入到 half，要求一致。PyTorch FP16 反向对 double 参考使用 `1e-3 * (1 + abs(ref))` 容差，包含最终 half 舍入误差。PyTorch 反向保留有限差分、选择性梯度、非默认流及覆盖写入检查。未开启校验时，`[SUCCESS]` 仅表示 CUDA 执行成功。

该入口用于快速分阶段测量；正式性能结论仍使用 `benchmarks/` 下的重复采样基准并遵循[基准规范](../../benchmarks/README.md)。

## FP16 基线

另提供 `launch_prefetch_fp16_k128`，约束与 k128 baseline 相同。权重一次加载至 2 KiB shared，输入使用两套 `half2[4][4]` 寄存器片段，按“加载 A → 加载 B → 计算 A → 加载下一 A → 计算 B”预取。保留 baseline 的地址计算、边界判断、bias 初始化与逐 tap FP32 FMA 顺序；只在初始化权重后同步。该版本接入全部 19 组 FP16 正确性用例，测量结果见 [FP16 prefetch 报告](../../docs/operators/dwconv/prefetch-fp16.md)。

`dwconv_fp16.cu` 参考 `/home/if/Codes/CUDAOP/op/dwconv/dwconv_fp16_common.cuh` 中 `UseBiasShuffle=true` 路径，提供自动分派和三个固定容量入口：

| 入口 | 卷积核面积范围 |
| --- | --- |
| `launch_baseline_fp16` | 自动选择：1–32 用 k32，33–64 用 k64，65–128 用 k128 |
| `launch_baseline_fp16_k128` | `1 <= KhKw <= 128` |
| `launch_baseline_fp16_k64` | `1 <= KhKw <= 64` |
| `launch_baseline_fp16_k32` | `1 <= KhKw <= 32` |

三个入口复用容量模板，分别实例化 kernel；直接加载 bias 并转为 FP32。声明位于 `dwconv_fp16.cuh`，最后一个参数可指定 CUDA stream，默认使用默认流；调用方负责检查启动错误与同步。

输入、输出仍为连续 NCHW，输入、bias、输出均为 `__half`。每个 block 覆盖四个通道对，即八个通道，因此通道数必须为 8 的倍数，`groups = in_ch = out_ch`，dilation 为 1。其余尺寸和索引约束与 FP32 相同；launch 使用 `block=256`、`grid=(ceil(outHW/256), C/8, N)`。

权重须预先按相邻通道配对为连续 `[C/2, KhKw]` 的 `__half2` 数组：

```cpp
packed_weights[pair * KhKw + k] =
    __halves2half2(weights[(2 * pair) * KhKw + k],
                  weights[(2 * pair + 1) * KhKw + k]);
```

kernel 一次加载全部配对权重到共享内存，直接读取 bias 并转为 FP32 初始化 `float2 output_frag[4]`。half2 输入和权重转为 float2，两个通道独立按原始 tap 顺序执行 FP32 FMA，写回时才舍入到 FP16，与 PyTorch 前向的累加精度一致；此处无需跨线程归约。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target dwconv_test --parallel
./build/tests/dwconv_test --dtype fp16 --kernel all --check-cpu --warmup 0 --iterations 1
```

历史 FP16 累加版本的原始基线测试包含 11 组配置，分别运行容量范围内的基线，共 27 次 kernel 验证，覆盖多 batch、多通道组、padding、stride、输出尾块、非整除 4 的 tap 数和 32/64/128 容量边界。CPU 从未打包的权重独立计算，用 double FMA 后逐步舍入到 half 模拟 FP16 累加，并要求结果数值完全一致；使用非整齐二进制输入值覆盖舍入行为。已通过正确性验证，未进行性能测量。

## 导入的 opt 实现

从 `/home/if/Codes/CUDAOP/op/dwconv/opt_conv2d_groups_fp32.cu` 和 `opt_conv2d_groups_fp16.cu` 提取 kernel 及完整 launch 分派逻辑，分别保存在 `dwconv_opt_fp32.cu` 和 `dwconv_opt_fp16.cu` 中，由 `cuda_ops_dwconv` 构建。原文件中的独立 main、计时、CSV、cuDNN 和 CPU 工具依赖未引入。

公共接口位于 `dwconv_opt.cuh`：

- `launch_opt_fp32(inputs, weights, bias, outputs, param, n, stream = nullptr)`：连续 NCHW FP32，权重为 `[C, KhKw]`，`C % 4 == 0`。
- `launch_opt_fp16(inputs, packed_weights, bias, outputs, param, n)`：连续 NCHW FP16，权重配对方式与 FP16 baseline 相同，`C % 8 == 0`。

两者要求 `groups = in_ch = out_ch`、dilation 为 1、`1 <= KhKw <= 128`，参数及派生尺寸有效，FP32 使用指定流（默认流为默认值），FP16 使用默认流异步启动，由调用方检查错误和同步。FP32、FP16 均直接从全局内存读取 bias 初始化累加器，不再通过 shared memory 广播 bias；FP16 使用 FP32 FMA 累加，写回时舍入到 FP16。shared-input 的动态共享内存大小同步移除 bias 空间，48 KiB 分派上限保留；FP32 的 `ldg32_nc_0` 调用映射到仓库中等价的 32 位 `ldg_nc_0` wrapper。

| 版本 | 原始分派策略 |
| --- | --- |
| FP32 | 3×3、5×5 使用尺寸特化；7×7 在 grouped blocks 小于 104 时使用单通道 kernel，否则使用四通道特化；11×11 在动态共享内存不超过 48 KiB 时使用 16×16 共享输入 tile；其余走通用 kernel |
| FP16 | 3×3、5×5、7×7 使用单通道对 kernel；9×9 在 pair blocks 至少为 140 且共享内存不超过 48 KiB 时使用共享输入 tile，否则使用单通道对；11×11 在共享内存允许时使用共享输入 tile，否则使用四通道对特化；其余走通用 kernel |

上述阈值来自原文件。FP32 测量见 [opt 与 baseline 对比](../../docs/operators/dwconv/opt-fp32.md)；FP16 全量测量见 [FP16 全量测试](../../docs/operators/dwconv/fp16-full.md)。

此前统一 `dwconv_test` 使用原 17 组 FP32、19 组 FP16 配置，合并去重后运行全部兼容的 baseline 和 opt。新增用例覆盖 3/5/7/9/11 尺寸分派、FP32 grouped blocks=100/104、FP16 pair blocks=136/140、共享输入 tile 的边界与 stride=1/2，以及 stride=3 时超出 48 KiB 的回退。当时的 FP16 累加版本与逐步 half 舍入 CPU 参考结果数值完全一致；当前 opt 已统一为 FP32 FMA 累加、写回时舍入到 FP16。

## PyTorch 原生 FP32/FP16 前向与反向

`dwconv_pytorch.cu` 抽离并简化自 PyTorch 提交 `9a3eac3b8ffba6e54e61df7ebecfb0afcc5e5a74` 的 [DepthwiseConv2d.cu](https://github.com/pytorch/pytorch/blob/9a3eac3b8ffba6e54e61df7ebecfb0afcc5e5a74/aten/src/ATen/native/cuda/DepthwiseConv2d.cu)。版权声明及许可保存在 [PYTORCH-LICENSE](PYTORCH-LICENSE)。仅依赖 CUDA Runtime，不依赖 PyTorch、ATen 或 cuDNN。

接口见 `dwconv_pytorch.cuh`：

- `launch_pytorch_forward_fp32(input, weight, bias, output, param, n, stream)`，bias 可为空。
- `launch_pytorch_backward_fp32(input, weight, grad_output, grad_input, grad_weight, grad_bias, param, n, stream)`，三个梯度输出可分别为空，全部为空时不启动 kernel。

FP16 对应接口为 `launch_pytorch_forward_fp16` 和 `launch_pytorch_backward_fp16`，所有数据指针使用 `__half`。权重仍为普通 `[C,Kh,Kw]`，无需 baseline 的 half2 配对；沿用上游 Half 的 FP32 乘加、共享内存及 shuffle 归约，写回时舍入到 FP16。db 也使用这一精度策略。

沿用 `Conv2DParam`，连续 NCHW FP32 或 FP16，权重 `[C,Kh,Kw]`，通道倍增为 1、dilation 为 1。支持任意正通道数，没有 4/8 通道对齐或核面积 128 的限制。尺寸与 stride 必须为正，padding 非负，所有尺寸、坐标、索引及循环中间值须在有符号 32 位范围内，派生字段与输出公式一致；接口不进行参数校验。指针不得重叠。调用方负责启动错误检查与同步。

保留上游 generic 前向的有效核边界裁剪、通用 dX gather，以及 dK 的 warp 按 batch 分工和 block 内归约。保留 FP32/FP16 类型模板，删除 dilation、通道倍增与核尺寸特化，使用固定 256 线程；ATen accessor 替换为裸指针，block 归约替换为简单 shuffle 和共享内存实现。这是简化移植版，不能据此推断完整 PyTorch 的性能。db 是本地补充的归约路径，并非从上游该文件抽离。

输出均覆盖写入，不需要清零或工作区，不使用全局原子加。只计算 dX 时 input 可为空；只计算 dK 时 weight 可为空；只计算 db 时 input、weight 均可为空。db 表示假设存在逐通道加性偏置时的梯度；无偏置层通常直接传空 grad_bias。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target dwconv_test --parallel
./build/tests/dwconv_test --kernel pytorch --dtype fp16 --check-cpu --warmup 0 --iterations 1
./build/tests/dwconv_test --kernel pytorch-backward --dtype all --check-cpu --warmup 0 --iterations 1
```

测试以 CPU double 前向及 scatter 反向为独立参考，小张量额外做有限差分。此前使用统一测试的 43 组配置，分别验证两种精度，包括非对齐通道、多 batch（含超过 8 个 warp 的 batch）、非方形核、stride 1/2/3、全 padding 输出、未采样输入、输出尾部、可选 bias、非默认流、选择性梯度和重复调用覆盖写入。本次仅运行正确性验证，单次 launch 时间不作为性能结论。

## FP32 输入寄存器预取

`dwconv.cu` 中的 `conv2d_4x128x256_groups_kernel_prefetch`，声明位于 `dwconv.cuh`，调用约束及 grid/block 与 k128 baseline 相同，原有 kernel 保留。

四个通道的完整权重一次加载到 2 KiB shared memory，越界 tap 补零；初始化后权重只读，仅需要一次 block 同步。输入使用两套各含 16 个 float 的寄存器片段，先加载 A，再按“预取 B → 计算 A → 预取 A → 计算 B”交替执行。新版输入加载完全沿用 baseline 的整数类型、空间边界判断和普通 C++ 全局加载；末轮无效 tap 由零权重处理，输出尾线程仍参与计算，最终屏蔽输出写入。与 baseline 相同，验证使用有限数值输入。

该结构为输入加载与独立乘加提供重叠机会，但实际 overlap、寄存器占用及性能取决于编译调度和设备；没有使用异步 shared 拷贝。统一 `dwconv_test` 支持该版本的分阶段计时和可选 CPU 校验。

验证命令：`./build/tests/dwconv_test --kernel prefetch --check-cpu --warmup 0 --iterations 1`。新增版本已通过全部 17 组配置。此前 k32 累加器初始化问题已由用户修复，2026-10-10 完整 `dwconv_test` 复测通过。

`dwconv_bench` 比较 baseline 和 prefetch。实现差别、性能数据和原因分析见[FP32 prefetch 与 baseline 对比](../../docs/operators/dwconv/prefetch-fp32.md)。`--large-batch` 覆盖 N=8/16/32、H=W=128；`--profile` 仅执行两个版本单次启动，供 NCU 采集。

## 权重缓存实验

`dwconv.cu` 中的权重缓存实验提供 `conv2d_4x128x256_groups_kernel_gmem` 和 `conv2d_4x128x256_groups_kernel_gmem_l2` 两个入口。两者基于当前 k128 baseline，仅将 shared 权重读取改为直接 `__ldg`，共用计算实现；后者由启动方额外设置仅覆盖 weights 的 L2 优先保留窗口，input 使用默认策略。

`dwconv_weights_bench` 默认比较 baseline、gmem、gmem_l2，两个版本均报告相对 baseline 的耗时变化。`--k3` 单测 N=16/C=64/H=W=128/K=3；`--profile`、`--profile-k3` 分别执行 K=7、K=3 的三版单次启动，供 NCU 采集。`dwconv_test` 覆盖两个计算入口，基准程序另外校验启用缓存策略后的结果。详见[权重缓存报告](../../docs/operators/dwconv/weights-cache.md)。

FP32 opt 的 bias 直接读取与 warp shuffle 广播对照入口为 `dwconv_bias_bench`。

完整 shape 集合的 cuDNN、baseline、PyTorch、opt 对比见 [FP32 全量测试](../../docs/operators/dwconv/fp32-full.md) 和 [FP16 全量测试](../../docs/operators/dwconv/fp16-full.md)，运行入口为 [dwconv_full_bench.py](../../benchmarks/operators/dwconv_full_bench.py)。
