"""复用 dwconv_test，轮换 cuDNN 模式和布局，保存原始样本与中位数。"""

import argparse
import csv
import itertools
import logging
import re
import statistics
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--output", type=Path, default=Path("build/dwconv-cudnn-layout/perf"))
    parser.add_argument("--cases", type=int, nargs="+", default=[7, 5, 1, 0, 8, 9, 22])
    parser.add_argument("--kernels", nargs="+", choices=["cudnn", "cudnn-fused", "cudnn-separate"], default=["cudnn"])
    parser.add_argument("--layouts", nargs="+", choices=["nchw", "nhwc"], default=["nchw"],
                        help="默认仅 nchw；布局对比显式传入 nchw nhwc")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.rounds < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("rounds、iterations 必须为正，warmup 不得为负")
    args.output.mkdir(parents=True, exist_ok=True)
    device = subprocess.run([str(args.build / "tools/device_query")], check=True, capture_output=True, text=True)
    (args.output / "device.txt").write_text(device.stdout)
    combinations = list(itertools.product(args.kernels, args.layouts))
    rows = []
    fields = ["case", "dtype", "kernel", "layout", "round", "status", "us", "prepare_ms", "post_ms",
              "host_layout_ms", "algo", "engine", "workspace_bytes"]
    with (args.output / "samples.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for repeat in range(args.rounds):
            offset = repeat % len(combinations)
            order = combinations[offset:] + combinations[:offset]
            for case in args.cases:
                for dtype in ["fp32", "fp16"]:
                    for kernel, layout in order:
                        command = [str(args.build / "tests/dwconv_test"), "--quick", "--case", str(case),
                                   "--kernel", kernel, "--layout", layout, "--dtype", dtype,
                                   "--warmup", str(args.warmup), "--iterations", str(args.iterations)]
                        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
                        log = result.stdout + result.stderr
                        name = f"r{repeat}-case{case}-{dtype}-{kernel}-{layout}.log"
                        (args.output / name).write_text("命令：" + " ".join(command) + "\n" + log)
                        row = dict(case=case, dtype=dtype, kernel=kernel, layout=layout, round=repeat)
                        if "[SKIP]" in log:
                            row["status"] = "unsupported"
                            logging.info("round=%s case=%s %s %s %s 不支持，见 %s",
                                         repeat, case, dtype, kernel, layout, name)
                        else:
                            if result.returncode or "[SUCCESS]" not in log:
                                raise RuntimeError(f"执行失败，见 {args.output / name}")
                            info = re.search(r"algo=(-?\d+) engine=(-?\d+) workspace=(\d+)", log)
                            row.update(
                                status="ok",
                                us=float(re.search(r"kernel launch：平均=([\d.eE+-]+) us", log)[1]),
                                prepare_ms=float(re.search(r"GPU-Mem / preprocess：([\d.eE+-]+)", log)[1]),
                                post_ms=float(re.search(r"free-Mem / postprocess：([\d.eE+-]+)", log)[1]),
                                host_layout_ms=float(re.search(r"主机输入重排=([\d.eE+-]+)", log)[1]),
                                algo=info[1], engine=info[2], workspace_bytes=info[3],
                            )
                            logging.info("round=%s case=%s %s %s %s %.3f us",
                                         repeat, case, dtype, kernel, layout, row["us"])
                        rows.append(row)
                        writer.writerow(row)
                        file.flush()
    with (args.output / "summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["case", "dtype", "kernel", "layout", "samples", "unsupported", "median_us", "min_us",
                         "max_us", "speedup_vs_nchw", "speedup_vs_separate", "host_layout_ms"])
        for case, dtype in itertools.product(args.cases, ["fp32", "fp16"]):
            selected = [r for r in rows if r["case"] == case and r["dtype"] == dtype]
            values = {(k, layout): [r["us"] for r in selected if r["kernel"] == k and r["layout"] == layout
                                    and r["status"] == "ok"] for k, layout in combinations}
            for kernel, layout in combinations:
                samples = values[kernel, layout]
                failed = sum(r["status"] == "unsupported" for r in selected
                             if r["kernel"] == kernel and r["layout"] == layout)
                if not samples:
                    writer.writerow([case, dtype, kernel, layout, 0, failed, "", "", "", "", "", ""])
                    continue
                median = statistics.median(samples)
                nchw = values.get((kernel, "nchw"), [])
                separate = values.get(("cudnn-separate", layout), [])
                packing = [r["host_layout_ms"] for r in selected if r["kernel"] == kernel
                           and r["layout"] == layout and r["status"] == "ok"]
                writer.writerow([case, dtype, kernel, layout, len(samples), failed, median, min(samples), max(samples),
                                 statistics.median(nchw) / median if nchw else "",
                                 statistics.median(separate) / median if separate else "", statistics.median(packing)])
    logging.info("[SUCCESS] 原始样本和汇总：%s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
