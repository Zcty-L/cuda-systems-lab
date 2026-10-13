"""串行测量完整 FP32/FP16 shape 集合，以 cuDNN 耗时为 1.0，支持中断后续跑。"""

import argparse
import csv
import hashlib
import json
import logging
import re
import subprocess
import statistics
from pathlib import Path


def write_comparison(rows, output, dtype="fp32"):
    """每个 shape 一行；容量标记只计算输入输出，不代表实际驻留状态。"""
    total_bytes = int(re.search(r"Global memory.*\((\d+) B\)", (output / "device.txt").read_text())[1])
    groups = {}
    for row in rows:
        groups.setdefault(row["case"], {})[row["kernel"]] = row
    shape_fields = ["case", "n", "c", "h", "w", "kh", "kw", "sh", "sw", "ph", "pw", "bias"]
    fields = shape_fields + ["input_output_bytes", "exceeds_physical_vram"]
    for kernel in dict.fromkeys(row["kernel"] for row in rows):
        fields += [f"{kernel}_status", f"{kernel}_us", f"{kernel}_relative"]
    with (output / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for group in groups.values():
            first = next(iter(group.values()))
            row = {key: first[key] for key in shape_fields}
            n, c, h, w, kh, kw, sh, sw, ph, pw = (int(row[key]) for key in shape_fields[1:-1])
            oh, ow = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
            size = (2 if dtype == "fp16" else 4) * n * c * (h * w + oh * ow)
            row.update(input_output_bytes=size, exceeds_physical_vram=int(size > total_bytes))
            for kernel, sample in group.items():
                row.update({f"{kernel}_status": sample["status"], f"{kernel}_us": sample.get("us", ""),
                            f"{kernel}_relative": sample.get("relative_to_cudnn", "")})
            writer.writerow(row)


def write_aggregates(output):
    """只汇总规则网格中输入输出未超过物理显存的完整测量组合。"""
    with (output / "comparison.csv").open() as file:
        rows = list(csv.DictReader(file))
    kernels = [key.removesuffix("_status") for key in rows[0] if key.endswith("_status")]
    rows = [r for r in rows if int(r["case"]) >= 23 and r["exceeds_physical_vram"] == "0"
            and all(r[f"{k}_status"] == "ok" for k in kernels)]
    with (output / "aggregates.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["k", "shapes"] + kernels)
        writer.writeheader()
        for size in sorted({int(r["kh"]) for r in rows}):
            group = [r for r in rows if int(r["kh"]) == size]
            writer.writerow(dict(k=size, shapes=len(group), **{
                kernel: statistics.median(float(r[f"{kernel}_relative"]) for r in group)
                for kernel in kernels}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--include-opt", action="store_true", help="补测 opt，复用相同配置下已有的三种实现日志")
    args = parser.parse_args()
    if args.output is None:
        args.output = args.build / f"dwconv-full-{args.dtype}"
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup 不得为负，iterations 必须为正")
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.build / "tests/dwconv_test"
    config = dict(binary=str(binary.resolve()), warmup=args.warmup, iterations=args.iterations,
                  dtype=args.dtype, layout="nchw", kernels=["cudnn", "baseline", "pytorch"])
    # 防止续跑时混入另一份二进制或配置。
    config["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    manifest = args.output / "config.json"
    if args.include_opt:
        config["kernels"].append("opt")
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        expected = dict(config, kernels=previous["kernels"])
        if previous != expected or not set(previous["kernels"]).issubset(config["kernels"]):
            raise RuntimeError("输出目录已有不同配置，请指定新的 --output")
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    device = subprocess.run([str(args.build / "tools/device_query")], check=True, capture_output=True, text=True)
    (args.output / "device.txt").write_text(device.stdout)
    listing = subprocess.run([str(binary), "--list-cases"], check=True, capture_output=True, text=True).stdout
    (args.output / "cases.txt").write_text(listing)
    cases = re.findall(r"case=(\d+) NCHW=(\d+),(\d+),(\d+),(\d+) K=(\d+)x(\d+) stride=(\d+)x(\d+) padding=(\d+)x(\d+) bias=(\d+)", listing)
    shape_fields = ["case", "n", "c", "h", "w", "kh", "kw", "sh", "sw", "ph", "pw", "bias"]
    fields = shape_fields + ["kernel", "status", "us", "relative_to_cudnn", "prepare_ms", "post_ms", "engine", "workspace_bytes"]
    rows = []
    with (args.output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, shape in enumerate(cases):
            group = []
            # 轮换相邻 shape 的执行顺序，减少固定顺序偏差。
            kernels = config["kernels"]
            offset = index % len(kernels)
            for kernel in kernels[offset:] + kernels[:offset]:
                path = args.output / f"case-{shape[0]}-{kernel}.log"
                command = [str(binary), "--case", shape[0], "--kernel", kernel, "--dtype", args.dtype,
                           "--layout", "nchw", "--warmup", str(args.warmup), "--iterations", str(args.iterations)]
                log = path.read_text() if path.exists() else ""
                incompatible = "没有兼容的 kernel / dtype / shape 组合"
                if "[SUCCESS] dwconv_test" not in log and incompatible not in log:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
                    log = result.stdout + result.stderr
                    path.write_text("命令：" + " ".join(command) + "\n" + log)
                    if result.returncode and incompatible not in log:
                        logging.error("case=%s kernel=%s 执行失败，见 %s", shape[0], kernel, path)
                row = dict(zip(shape_fields, shape))
                row.update(kernel=kernel, status="error")
                timing = re.search(r"kernel launch：平均=([\d.eE+-]+) us", log)
                if "[SUCCESS] dwconv_test" in log and timing:
                    row.update(status="ok", us=float(timing[1]),
                               prepare_ms=float(re.search(r"GPU-Mem / preprocess：([\d.eE+-]+)", log)[1]),
                               post_ms=float(re.search(r"free-Mem / postprocess：([\d.eE+-]+)", log)[1]))
                    info = re.search(r"engine=(-?\d+) workspace=(\d+)", log)
                    if info:
                        row.update(engine=info[1], workspace_bytes=info[2])
                elif "[SKIP]" in log or incompatible in log:
                    row["status"] = "unsupported"
                group.append(row)
            reference = next((r["us"] for r in group if r["kernel"] == "cudnn" and r["status"] == "ok"), None)
            for row in group:
                if reference and row["status"] == "ok":
                    row["relative_to_cudnn"] = row["us"] / reference
                writer.writerow(row)
            file.flush()
            rows.extend(group)
            logging.info("完成 %d/%d shapes：case=%s %s", index + 1, len(cases), shape[0],
                         " ".join(f"{r['kernel']}={r.get('us', r['status'])}" for r in group))
    failed = sum(r["status"] == "error" for r in rows)
    skipped = sum(r["status"] == "unsupported" for r in rows)
    write_comparison(rows, args.output, args.dtype)
    write_aggregates(args.output)
    logging.info("%s %d 项，失败 %d 项，不支持 %d 项；结果：%s", "[SUCCESS]" if not failed else "[完成]", len(rows), failed, skipped, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
