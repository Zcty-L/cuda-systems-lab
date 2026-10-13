# RTX 4090 DWConv 全量实验归档

测量日期：2026-10-13。正式报告见[FP32 全量前向测试](../../fp32-full-rtx4090.md)和[FP16 全量前向测试](../../fp16-full-rtx4090.md)。设备查询名称为 RTX 4090，物理 GPU 0；本归档不把设备自行标注为 4090D。

[FP32 数据](fp32/)和[FP16 数据](fp16/)各包含 `summary.csv`（逐 shape/实现）、`comparison.csv`（横向比较）、`aggregates.csv`（按 K 汇总）、`cpu-check.csv`（抽样 CPU 数值校验）、`config.json`、`device.txt`、`gpu-identification.csv` 和 `cases.txt`。配置中的绝对路径是测量时的原始路径。逐项命令与原始输出保留在 `build/dwconv-full-fp32-rtx4090-gpu0/`、`build/dwconv-full-fp16-rtx4090-gpu0/`；CPU 校验原始输出分别保留在 `build/dwconv-fp32-rtx4090-cpu-check.log` 和 `build/dwconv-fp16-rtx4090-cpu-check.log`。

## 吞吐量图

每张图按 H=W=40/80/160（行）和 C=32/64/128/256/512/1024（列）排列 18 个子图，横轴为 batch size，纵轴为 TFLOPs/s；每个 K 使用 126 个规则配置。吞吐量按 `2×N×C×Hout×Wout×Kh×Kw / (kernel_us×10⁶)` 计算，FLOPs 不计 bias，计时包含 bias。两种精度的输入和输出均未超过物理显存。图标题使用设备查询名称；本机缺少 Times New Roman，绘图时使用 Liberation Serif。

| K | FP32 | FP16 |
| --- | --- | --- |
| 3×3 | [图片](fp32/plots/benchmark_fp32_k3.png) | [图片](fp16/plots/benchmark_fp16_k3.png) |
| 5×5 | [图片](fp32/plots/benchmark_fp32_k5.png) | [图片](fp16/plots/benchmark_fp16_k5.png) |
| 7×7 | [图片](fp32/plots/benchmark_fp32_k7.png) | [图片](fp16/plots/benchmark_fp16_k7.png) |
| 9×9 | [图片](fp32/plots/benchmark_fp32_k9.png) | [图片](fp16/plots/benchmark_fp16_k9.png) |
| 11×11 | [图片](fp32/plots/benchmark_fp32_k11.png) | [图片](fp16/plots/benchmark_fp16_k11.png) |

在仓库根目录重新生成：

```bash
for dtype in fp32 fp16; do
    for k in 3 5 7 9 11; do
        conda run -n py311 python benchmarks/operators/dwconv_plot.py \
            --csv "docs/operators/dwconv/results/2026-10-13-rtx4090/$dtype/comparison.csv" \
            --dtype "$dtype" --k "$k" \
            --output "docs/operators/dwconv/results/2026-10-13-rtx4090/$dtype/plots/benchmark_${dtype}_k${k}.png"
    done
done
```
