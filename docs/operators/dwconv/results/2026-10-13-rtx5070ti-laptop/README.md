# DWConv 全量实验归档

归档日期：2026-10-13；设备：RTX 5070 Ti Laptop GPU。FP32 测量跨 10 月 12–13 日，FP16 测量于 10 月 13 日完成。

报告：[FP32](../../fp32-full.md) · [FP16](../../fp16-full.md)。本目录保存已完成实验的结果快照，后续测量使用新的归档目录。

每种精度包含 `summary.csv`（逐 shape/kernel 原始指标）、`comparison.csv`（逐 shape 横向比较）、`aggregates.csv`（按 K 汇总）、`config.json`（测量配置及二进制 SHA256）、`device.txt`（设备信息）、`cases.txt`（用例列表）及 `plots/`。配置中的绝对路径保留原始测量环境。逐用例日志与续跑文件仍在 `build/dwconv-full-{fp32,fp16}/`。

## 数据与图片

| 精度 | 原始指标 | 横向比较 | 汇总 |
| --- | --- | --- | --- |
| FP32 | [summary.csv](fp32/summary.csv) | [comparison.csv](fp32/comparison.csv) | [aggregates.csv](fp32/aggregates.csv) |
| FP16 | [summary.csv](fp16/summary.csv) | [comparison.csv](fp16/comparison.csv) | [aggregates.csv](fp16/aggregates.csv) |

图片为 3×6 子图，行对应 H=W=40/80/160，列对应 C=32/64/128/256/512/1024；横轴为 batch size，纵轴为 TFLOPs/s，使用 Times New Roman。只绘制 630 个规则配置，FP32 超物理显存容量的点用黑色叉号标记。TFLOPs/s = 2×N×C×Hout×Wout×Kh×Kw / (kernel_us×10⁶)，FLOPs 不计 bias，但计时包含 bias。

| K | FP32 | FP16 |
| --- | --- | --- |
| 3 | [图片](fp32/plots/benchmark_fp32_k3.png) | [图片](fp16/plots/benchmark_fp16_k3.png) |
| 5 | [图片](fp32/plots/benchmark_fp32_k5.png) | [图片](fp16/plots/benchmark_fp16_k5.png) |
| 7 | [图片](fp32/plots/benchmark_fp32_k7.png) | [图片](fp16/plots/benchmark_fp16_k7.png) |
| 9 | [图片](fp32/plots/benchmark_fp32_k9.png) | [图片](fp16/plots/benchmark_fp16_k9.png) |
| 11 | [图片](fp32/plots/benchmark_fp32_k11.png) | [图片](fp16/plots/benchmark_fp16_k11.png) |

## 重新绘图

在项目根目录执行（依赖 matplotlib 和 Times New Roman 字体；WSL 自动读取 Windows 字体，也可传入 `--font-file`）：

```bash
conda activate py311
archive=docs/operators/dwconv/results/2026-10-13-rtx5070ti-laptop
for dtype in fp32 fp16; do
    for k in 3 5 7 9 11; do
        python benchmarks/operators/dwconv_plot.py \
            --csv "$archive/$dtype/comparison.csv" --dtype "$dtype" --k "$k" \
            --output "$archive/$dtype/plots/benchmark_${dtype}_k${k}.png"
    done
done
```
