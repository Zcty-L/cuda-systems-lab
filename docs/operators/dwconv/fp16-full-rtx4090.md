# RTX 4090 上的 DWConv FP16 全量前向测试

## 环境与范围

2026-10-13 在与[FP32 测试](fp32-full-rtx4090.md)相同的物理 GPU 0 上完成。设备查询返回 `NVIDIA GeForce RTX 4090`，SM 8.9、128 SM、23.52 GiB 显存、72 MiB L2；GPU UUID 为 `GPU-f6ffe6fd-aecd-d62b-3522-19e8975458b2`。驱动 590.48.01、CUDA 13.1、cuDNN 9.24.0。设备自身未报告“4090D”，因此本次结果按实际查询名称归档。

沿用[FP16 全量对比](fp16-full.md)的 653 个 shape：23 个快速用例，以及 N={1,2,4,8,16,32,64}、C={32,64,128,256,512,1024}、H=W={40,80,160}、K={3,5,7,9,11} 的 630 个规则组合。FP16、NCHW、stride=1、padding=K/2，包含 bias。比较 cuDNN Backend Graph Conv→ADD、自动分派 baseline、opt 和仓库中抽离的 PyTorch CUDA 前向 kernel；PyTorch 项不包含框架调度。各路径使用 FP32 累加，结果写回 FP16。

四种实现单 GPU 串行扫描，同一 shape 相邻执行，顺序随 case 轮换。每项独立进程，预热 20 次，CUDA Event 测量 100 次连续 launch 的平均值。计时包含卷积和 bias，不包含主机数据生成、权重打包、分配、H2D、cuDNN plan/workspace 准备及释放。不锁频、不清缓存；仅进行一轮全量扫描，短 kernel 可能受主机提交间隙影响。全量扫描关闭 CPU 数值校验，其 `[SUCCESS]` 仅表示 CUDA 执行成功。

## 复现与数据

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target dwconv_test device_query -j 8
CUDA_VISIBLE_DEVICES=0 ./build/tools/device_query
CUDA_VISIBLE_DEVICES=0 conda run -n py311 python benchmarks/operators/dwconv_full_bench.py \
  --dtype fp16 --include-opt --output build/dwconv-full-fp16-rtx4090-gpu0
```

性能扫描从 18:04 至 18:39 完成。`build/dwconv-full-fp16-rtx4090-gpu0/` 保留逐项命令和原始输出；[实验归档](results/2026-10-13-rtx4090/README.md)保存逐项 CSV、汇总、五张吞吐量图、配置、设备信息和 case 列表。配置记录测试二进制 SHA256。`relative_to_cudnn = kernel_us / cudnn_us`，小于 1 表示耗时低于当前 cuDNN plan。

## 正确性与完整性

完成 653 × 4 = 2,612 项 FP16 前向记录：2,608 项 `ok`、4 项 `unsupported`、0 项执行失败。不支持项为 case 15（C=4）和 case 16（C=36）的 baseline、opt，原因是这两个 FP16 接口要求 C 为 8 的倍数；cuDNN 与 PyTorch 对这两个 shape 均执行成功。所有配置的输入和输出合计大小均低于所选 GPU 的物理显存容量。

额外选取 case 7、5、22、401、8，分别覆盖 K=3、5、7、9、11；每个 case 对四种实现开启 `--check-cpu`，预热 2 次、计时 5 次。20 项全部通过独立 CPU 参考校验，最大绝对误差 `0.000244141`；逐项结果见[CPU 校验记录](results/2026-10-13-rtx4090/fp16/cpu-check.csv)。其余 648 个 shape 未做 CPU 数值校验。

## 性能结果

下表只使用 630 个规则组合，每种 K 有 126 个 shape；数值为逐 shape 相对 cuDNN 耗时的中位数，越小越快。最后一列直接计算每个 shape 的 `opt_us / baseline_us` 后取中位数。

| K | cuDNN | baseline | PyTorch | opt | opt / baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3×3 | 1.000 | 0.457 | 0.682 | 0.331 | 0.646 |
| 5×5 | 1.000 | 0.672 | 0.826 | 0.583 | 0.900 |
| 7×7 | 1.000 | 0.757 | 0.701 | 0.657 | 0.900 |
| 9×9 | 1.000 | 0.537 | 0.606 | 0.167 | 0.324 |
| 11×11 | 1.000 | 0.551 | 0.552 | 0.168 | 0.318 |
| 全部 | 1.000 | 0.695 | 0.708 | 0.343 | 0.639 |

在 630 个规则组合中，baseline、PyTorch、opt 分别有 604、568、630 个 shape 的耗时低于所选 cuDNN plan。opt 相比 baseline 的逐 shape 耗时比值在 9×9 和 11×11 组明显较低；这是当前设备、实现分派和 cuDNN heuristic 下的一轮结果，不代表穷举 cuDNN engine 后的最优表现。

cuDNN 在 653 个 shape 中选择 engine 0 共 521 项、engine 11 共 132 项，workspace 均为零。engine 11 仅出现于 9×9、11×11；这两组各有 63 个规则组合使用 engine 11，另有 6 个 11×11 快速用例使用该 engine。部分小输入上的 cuDNN 耗时明显偏高。例如 case 8 的原测为 160.748 μs，以相同预热和迭代次数单独重测为 166.963 μs，仍选择 engine 11。该现象在当前条件下可复现，但仅凭计时不能确定底层瓶颈。

| case | N×C×H×W | K | cuDNN μs | baseline μs | PyTorch μs | opt μs |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 32×128×80×80 | 7 | 827.09 | 586.81 | 561.61 | 531.23 |
| 7 | 1×32×43×43 | 3 | 4.31 | 4.59 | 2.66 | 2.36 |
| 8 | 1×32×43×43 | 11 | 160.75 | 24.80 | 7.52 | 6.72 |
| 9 | 32×128×80×80 | 11 | 8492.54 | 1376.45 | 1505.29 | 398.42 |
| 401 | 1×32×40×40 | 9 | 164.43 | 17.48 | 6.60 | 4.60 |
| 652 | 64×1024×160×160 | 11 | 104356.00 | 92397.80 | 94726.00 | 32247.10 |

完整逐项数据和 cuDNN engine 记录见归档。几个百分点的差异需要重复测量后再判断稳定性。
