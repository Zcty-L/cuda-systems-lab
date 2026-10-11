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

原始基线测试使用 CPU depthwise 参考计算分别验证四个 kernel，仅运行各版本容量范围内的配置。11 组配置覆盖 bias、多个 batch 和通道组、padding、不同 stride、输出尾块、完整 256 元素输出块及面积 1、4、9、15、31、32、63、64、121、128 的卷积核，包含三个基线的容量边界和非整除 4 的尾部。

本次只导入实现并验证正确性，没有性能计时或性能结论；后续测量应遵循[基准规范](../../benchmarks/README.md)。

## FP16 基线

另提供 `launch_prefetch_fp16_k128`，约束与 k128 baseline 相同。权重一次加载至 2 KiB shared，输入使用两套 `half2[4][4]` 寄存器片段，按“加载 A → 加载 B → 计算 A → 加载下一 A → 计算 B”预取。保留 baseline 的地址计算、边界判断、bias 初始化与逐 tap FP16 FMA 顺序；只在初始化权重后同步。该版本接入全部 19 组 FP16 正确性用例，测量结果见 [FP16 prefetch 报告](../../docs/operators/dwconv/fp16-prefetch.md)。

`dwconv_fp16.cu` 参考 `/home/if/Codes/CUDAOP/op/dwconv/dwconv_fp16_common.cuh` 中 `UseBiasShuffle=true` 路径，提供三个入口：

| 入口 | 卷积核面积范围 |
| --- | --- |
| `launch_baseline_fp16_k128` | `1 <= KhKw <= 128` |
| `launch_baseline_fp16_k64` | `1 <= KhKw <= 64` |
| `launch_baseline_fp16_k32` | `1 <= KhKw <= 32` |

三个入口复用容量模板，分别实例化 kernel；使用寄存器广播 bias。声明位于 `dwconv_fp16.cuh`，最后一个参数可指定 CUDA stream，默认使用默认流；调用方负责检查启动错误与同步。

输入、输出仍为连续 NCHW，输入、bias、输出均为 `__half`。每个 block 覆盖四个通道对，即八个通道，因此通道数必须为 8 的倍数，`groups = in_ch = out_ch`，dilation 为 1。其余尺寸和索引约束与 FP32 相同；launch 使用 `block=256`、`grid=(ceil(outHW/256), C/8, N)`。

权重须预先按相邻通道配对为连续 `[C/2, KhKw]` 的 `__half2` 数组：

```cpp
packed_weights[pair * KhKw + k] =
    __halves2half2(weights[(2 * pair) * KhKw + k],
                  weights[(2 * pair + 1) * KhKw + k]);
```

kernel 一次加载全部配对权重到共享内存，以 shuffle 广播的 bias 初始化累加器，按原始顺序执行 `__hfma2`。累加精度为 FP16，每步 FMA 舍入到 FP16。该数值语义与 FP32 累加不同。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target dwconv_fp16_test --parallel
ctest --test-dir build -R '^dwconv(_fp16)?_test$' --output-on-failure
```

FP16 原始基线测试包含 11 组配置，分别运行容量范围内的基线，共 27 次 kernel 验证，覆盖多 batch、多通道组、padding、stride、输出尾块、非整除 4 的 tap 数和 32/64/128 容量边界。CPU 从未打包的权重独立计算，用 double FMA 后逐步舍入到 half 模拟 FP16 累加，并要求结果数值完全一致；使用非整齐二进制输入值覆盖舍入行为。已通过正确性验证，未进行性能测量。

## FP32 输入寄存器预取

`dwconv.cu` 中的 `conv2d_4x128x256_groups_kernel_prefetch`，声明位于 `dwconv.cuh`，调用约束及 grid/block 与 k128 baseline 相同，原有 kernel 保留。

四个通道的完整权重一次加载到 2 KiB shared memory，越界 tap 补零；初始化后权重只读，仅需要一次 block 同步。输入使用两套各含 16 个 float 的寄存器片段，先加载 A，再按“预取 B → 计算 A → 预取 A → 计算 B”交替执行。新版输入加载完全沿用 baseline 的整数类型、空间边界判断和普通 C++ 全局加载；末轮无效 tap 由零权重处理，输出尾线程仍参与计算，最终屏蔽输出写入。与 baseline 相同，验证使用有限数值输入。

该结构为输入加载与独立乘加提供重叠机会，但实际 overlap、寄存器占用及性能取决于编译调度和设备；没有使用异步 shared 拷贝。`dwconv_test` 将其加入现有 17 组 CPU 参考正确性验证，不包含性能计时。

验证命令：`ctest --test-dir build -R '^dwconv_prefetch_test$' --output-on-failure`（等价于 `./build/tests/dwconv_test --prefetch-only`）。新增版本已通过全部 17 组配置。此前 k32 累加器初始化问题已由用户修复，2026-10-10 完整 `dwconv_test` 复测通过。

`dwconv_bench` 比较 baseline 和 prefetch。实现差别、性能数据和原因分析见[prefetch 与 baseline 对比](../../docs/operators/dwconv/prefetch-matched.md)。`--large-batch` 覆盖 N=8/16/32、H=W=128；`--profile` 仅执行两个版本单次启动，供 NCU 采集。

## 权重缓存实验

`dwconv.cu` 中的权重缓存实验提供 `conv2d_4x128x256_groups_kernel_gmem` 和 `conv2d_4x128x256_groups_kernel_gmem_l2` 两个入口。两者基于当前 k128 baseline，仅将 shared 权重读取改为直接 `__ldg`，共用计算实现；后者由启动方额外设置仅覆盖 weights 的 L2 优先保留窗口，input 使用默认策略。

`dwconv_weights_bench` 默认比较 baseline、gmem、gmem_l2，两个版本均报告相对 baseline 的耗时变化。`--k3` 单测 N=16/C=64/H=W=128/K=3；`--profile`、`--profile-k3` 分别执行 K=7、K=3 的三版单次启动，供 NCU 采集。`dwconv_test` 覆盖两个计算入口，基准程序另外校验启用缓存策略后的结果。详见[权重缓存报告](../../docs/operators/dwconv/weights-cache.md)。
