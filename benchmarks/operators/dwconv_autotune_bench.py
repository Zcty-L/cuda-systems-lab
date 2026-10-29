"""重复比较 cuDNN heuristic 与 autotune，保存原始日志与 CSV。"""
import argparse
import csv
import logging
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='build/dwconv-autotune')
    parser.add_argument('--dtype', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--cases', type=int, nargs='+', default=[7, 5, 4, 401, 8])
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats 必须为正数')
    logging.basicConfig(level=logging.INFO)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    (out / 'device.txt').write_text(subprocess.check_output(
        ['./build/tools/device_query'], text=True))
    for case in args.cases:
        for rep in range(args.repeats):
            order = ['cudnn', 'cudnn-autotune']
            if rep % 2:
                order.reverse()
            for kernel in order:
                cmd = ['./build/tests/dwconv_test', '--kernel', kernel,
                       '--dtype', args.dtype, '--case', str(case),
                       '--warmup', '100', '--iterations', '500']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                (out / f'{case}-{rep}-{kernel}.log').write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                us = float(re.search(r'kernel launch：平均=([\d.]+)', result.stdout)[1])
                engine = re.findall(r'(?:selected_engine|engine)=(-?\d+)', result.stdout)[-1]
                rows.append(dict(case=case, repeat=rep, kernel=kernel, us=us, engine=engine))
                with (out / 'summary.csv').open('w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(rows)
                logging.info('%s', rows[-1])


if __name__ == '__main__':
    main()
