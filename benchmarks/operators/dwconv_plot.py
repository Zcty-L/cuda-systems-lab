"""将全量 DWConv comparison.csv 绘制为按 H、C 分面的吞吐量图。"""

import argparse
import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager


def configure_font(font_file):
    """优先使用指定字体文件；否则尝试新罗马字体和现有衬线字体。"""
    if font_file:
        font_manager.fontManager.addfont(str(font_file))
    else:
        for path in Path('/mnt/c/Windows/Fonts').glob('times*.ttf'):
            font_manager.fontManager.addfont(str(path))
    for family in ('Times New Roman', 'Liberation Serif'):
        try:
            path = font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams['font.family'] = family
        logging.info('字体：%s (%s)', family, path)
        return
    raise RuntimeError('未找到 Times New Roman 或 Liberation Serif，请通过 --font-file 指定字体文件')


def device_label(csv_path):
    """从 comparison.csv 同目录的设备查询记录读取图标题。"""
    device_file = csv_path.with_name('device.txt')
    if not device_file.exists():
        raise RuntimeError(f'缺少设备记录：{device_file}')
    for line in device_file.read_text().splitlines():
        if line.strip().startswith('Name '):
            return line.split('Name', 1)[1].strip().removeprefix('NVIDIA GeForce ').removesuffix(' GPU')
    raise RuntimeError(f'设备记录缺少 Name：{device_file}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--dtype', choices=['fp32', 'fp16'], required=True)
    parser.add_argument('--k', type=int, default=11)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font-file', type=Path, help='Times New Roman 字体文件路径')
    args = parser.parse_args()
    configure_font(args.font_file)
    with args.csv.open() as file:
        # 全量脚本的前 23 项为快速用例，排除它们以免重复 shape。
        rows = [r for r in csv.DictReader(file)
                if int(r['case']) >= 23 and int(r['kh']) == args.k and int(r['kw']) == args.k]
    if not rows:
        parser.error('没有匹配的规则网格数据')
    channels = sorted({int(r['c']) for r in rows})
    heights = sorted({int(r['h']) for r in rows})
    batches = sorted({int(r['n']) for r in rows})
    kernels = [('baseline', 'baseline', '#2196F3'), ('opt', 'opt', '#9C27B0'),
               ('pytorch', 'PyTorch CUDA', '#4CAF50'), ('cudnn', 'cuDNN Graph', '#FF9800')]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(len(heights), len(channels), figsize=(4 * len(channels), 4 * len(heights)),
                             squeeze=False, sharex=True, sharey=True)
    count = 0
    max_rate = 0.0
    for hi, h in enumerate(heights):
        for ci, c in enumerate(channels):
            ax = axes[hi, ci]
            group = sorted([r for r in rows if int(r['h']) == h and int(r['c']) == c], key=lambda r: int(r['n']))
            assert len(group) == len(batches), '网格不完整或存在重复 shape'
            for kernel, label, color in kernels:
                xs, ys = [], []
                for r in group:
                    if r[kernel + '_status'] != 'ok':
                        raise ValueError(f"case={r['case']} {kernel} 无有效数据")
                    n, ch, ih, iw, kh, kw, sh, sw, ph, pw = [int(r[key]) for key in
                        ['n', 'c', 'h', 'w', 'kh', 'kw', 'sh', 'sw', 'ph', 'pw']]
                    oh, ow = (ih + 2 * ph - kh) // sh + 1, (iw + 2 * pw - kw) // sw + 1
                    rate = 2 * n * ch * oh * ow * kh * kw / (float(r[kernel + '_us']) * 1e6)
                    max_rate = max(max_rate, rate)
                    xs.append(n)
                    ys.append(rate)
                    count += 1
                    if r['exceeds_physical_vram'] == '1':
                        ax.scatter(n, rate, marker='x', s=95, color='black', linewidths=1.5, zorder=5)
                ax.plot(xs, ys, color=color, marker='o', markersize=4, linewidth=1.7, label=label)
            ax.set_title(f'C={c}, H=W={h}')
            ax.grid(True, color='#dddddd', linewidth=0.7)
            ax.set_xticks([1, 8, 16, 32, 64])
            if hi == len(heights) - 1:
                ax.set_xlabel('Batch size (N)')
            if ci == 0:
                ax.set_ylabel('TFLOPs/s')
    axes[0, 0].set_ylim(0, max_rate * 1.08)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if any(r['exceeds_physical_vram'] == '1' for r in rows):
        handles.append(Line2D([], [], color='black', marker='x', linestyle='None'))
        labels.append('Input + output > physical VRAM')
    fig.suptitle(f'Depthwise Conv {args.dtype.upper()} | K={args.k}x{args.k} | NCHW | {device_label(args.csv)}', fontsize=18, y=0.99)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=len(handles), frameon=False)
    fig.text(0.5, 0.012, 'FLOPs = 2 x N x C x Hout x Wout x K x K (bias excluded); kernel time includes bias. '
             '20 warmup / 100 iterations; single sweep.', ha='center', fontsize=10, color='#555555')
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.09, top=0.87, hspace=0.38, wspace=0.12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor='white')
    plt.close(fig)
    logging.info('[SUCCESS] 已绘制 %d 个配置、%d 个数据点：%s', len(rows), count, args.output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
