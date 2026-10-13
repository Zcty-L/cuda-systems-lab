# RTX 4090 上的 DWConv FP32 全量前向测试

## 环境与范围

2026-10-13 在物理 GPU 0 上完成。设备查询返回 `NVIDIA GeForce RTX 4090`，SM 8.9、128 SM、23.52 GiB 显存、72 MiB L2；GPU UUID 为 `GPU-f6ffe6fd-aecd-d62b-3522-19e8975458b2`。驱动 590.48.01、CUDA 13.1、cuDNN 9.24.0。设备自身未报告“4090D”，故本文按查询结果命名；本次数据不能直接标作已确认的 4090D 测量。

沿用[FP32 全量对比](fp32-full.md)的 653 个 shape：23 个快速用例和 630 个规则组合（N={1,2,4,8,16,32,64}、C={32,64,128,256,512,1024}、H=W={40,80,160}、K={3,5,7,9,11}）。FP32、NCHW、stride=1、padding=K/2，包含 bias。比较 cuDNN Backend Graph Conv→ADD、自动分派 baseline、opt 和仓库中抽离的 PyTorch CUDA 前向 kernel；最后一项不包含 PyTorch 框架调度。

单 GPU 串行扫描，每个 shape 的四种实现轮换执行顺序；各进程预热 20 次，CUDA Event 计时 100 次连续 launch 的平均值。计时包含卷积和 bias，不包含数据生成、分配、传输、cuDNN plan/workspace 准备及释放。不锁频、不清缓存；本次只做一次全量扫描，短 kernel 可能受主机提交间隙影响。全量扫描关闭 CPU 数值校验，其 `[SUCCESS]` 仅证明 CUDA 执行成功。

## 复现与数据

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target dwconv_test device_query -j 8
CUDA_VISIBLE_DEVICES=0 ./build/tools/device_query
CUDA_VISIBLE_DEVICES=0 conda run -n py311 python benchmarks/operators/dwconv_full_bench.py \
  --dtype fp32 --include-opt --output build/dwconv-full-fp32-rtx4090-gpu0
```

性能扫描从 17:08 至 17:42 完成。`build/dwconv-full-fp32-rtx4090-gpu0/` 保留逐项原始日志；[实验归档](results/2026-10-13-rtx4090/README.md)保存 `summary.csv`、`comparison.csv`、`aggregates.csv`、五张吞吐量图、配置、设备信息和 case 列表。配置记录测试二进制 SHA256。`relative_to_cudnn = kernel_us / cudnn_us`，小于 1 表示耗时低于 cuDNN。

## 正确性与完整性

完成 653 × 4 = 2,612 项 FP32 前向测量，全部状态为 `ok`，失败及不支持项均为 0。所有输入和输出合计大小均小于所选 GPU 的物理显存容量。

额外选取 case 7、5、22、401、8，分别覆盖 K=3、5、7、9、11；每个 case 对四种实现开启 `--check-cpu`，预热 2 次、计时 5 次。20 项全部通过独立 CPU double 参考校验，最大绝对误差为 `9.23838e-07`；逐项结果见[CPU 校验记录](results/2026-10-13-rtx4090/fp32/cpu-check.csv)。其余 648 个 shape 未做 CPU 数值校验。

## 性能结果

下表只使用 630 个规则组合，每个 K 有 126 个 shape；数值是逐 shape 相对 cuDNN 耗时的中位数，越小越快。最后一列直接计算逐 shape 的 `opt_us / baseline_us` 后取中位数。

| K | cuDNN | baseline | PyTorch | opt | opt / baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3×3 | 1.000 | 0.605 | 0.656 | 0.592 | 0.924 |
| 5×5 | 1.000 | 0.663 | 0.786 | 0.440 | 0.656 |
| 7×7 | 1.000 | 0.749 | 0.686 | 0.485 | 0.627 |
| 9×9 | 1.000 | 0.856 | 1.076 | 0.851 | 0.996 |
| 11×11 | 1.000 | 0.860 | 0.961 | 0.419 | 0.490 |
| 全部 | 1.000 | 0.727 | 0.763 | 0.480 | 0.648 |

在 630 个规则组合中，baseline、PyTorch、opt 各有 621、503、627 个 shape 的耗时低于 cuDNN。9×9 的 opt / baseline 中位数为 0.996，单次扫描不支持稳定提升的判断；11×11 的差异较大。cuDNN heuristic 在 653 项中选择 engine 0 共 611 项、engine 11 共 42 项，均为零 workspace；engine 11 全部出现在 11×11。此处比较的是所选 plan，而非穷举 cuDNN engine 后的最优值。

| case | N×C×H×W | K | cuDNN μs | baseline μs | PyTorch μs | opt μs |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 32×128×80×80 | 7 | 809.50 | 576.84 | 539.69 | 366.66 |
| 7 | 1×32×43×43 | 3 | 4.58 | 3.01 | 2.67 | 2.56 |
| 9 | 32×128×80×80 | 11 | 1554.34 | 1330.27 | 1501.68 | 647.62 |
| 401 | 1×32×40×40 | 9 | 7.14 | 8.66 | 6.78 | 8.52 |
| 634 | 32×1024×160×160 | 11 | 275923.00 | 44790.50 | 47467.40 | 20980.30 |
| 652 | 64×1024×160×160 | 11 | 554152.00 | 89470.50 | 95700.60 | 41996.90 |

部分 11×11 大配置的 cuDNN engine 11 耗时明显偏高。例如 case 529（1×32×160×160）原测 252.508 μs，以相同预热和迭代次数单独重测为 250.409 μs，仍选择 engine 11；这支持该现象在当前条件下可复现，但不能仅据计时确定底层瓶颈。完整逐项数据可用于检查各 shape 的差异；少量百分比级别差异需要重复测量后再判断。
