"""串行测量 DWConv 多后端、布局与选优策略，支持子集、独立进程重复及续跑。"""

import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import re
import subprocess
from pathlib import Path

if __package__:
    from .reporting import SHAPE_FIELDS, annotate_references, write_reports
    from .runner import benchmark_binary, cached_returncode, canonical_kernel, invoke, parse_result, process_log
else:
    from reporting import SHAPE_FIELDS, annotate_references, write_reports
    from runner import benchmark_binary, cached_returncode, canonical_kernel, invoke, parse_result, process_log


KERNELS = [
    "baseline", "baseline-k32", "baseline-k64", "baseline-k128",
    "prefetch",
    "opt",
    "gmem",
    "gmem-l2-entry",
    "pytorch",        # PytorchForward 包装接口 → native launch，NCHW；NHWC 会先转换
    "pytorch-native", # 直接调用 launch_pytorch_forward_fp32/fp16，NCHW
    "cudnn", "cudnn-fused", "cudnn-separate", "cudnn-heuristic", "cudnn-autotune", "cudnn-graph",
    "tensorrt", "tensorrt-baseline", "tensorrt-opt"
]

SAMPLE_FIELDS = SHAPE_FIELDS + [
    "dtype", "kernel", "layout", "repeat", "status", "us", "relative_to_reference", "relative_to_cudnn",
    "reference_kernel", "reference_layout", "speedup_vs_nchw", "speedup_vs_separate", "warmup", "iterations",
    "samples", "samples_us", "min_us", "max_us", "prepare_ms", "post_ms", "host_layout_ms", "algo", "engine",
    "workspace_bytes", "timing_mode", "checked", "reason"
]


def memory_skip_reason(shape, dtype, max_io_gib=None):
    """启动子进程前预留显存；输入输出只是显存需求的下界。"""
    _, n, c, h, w, kh, kw, sh, sw, ph, pw, _ = map(int, shape)
    oh, ow = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
    size = (2 if dtype == "fp16" else 4) * n * c * (h * w + oh * ow)
    if max_io_gib is not None and size > max_io_gib * 2**30:
        return f"输入输出 {size / 2**30:.2f} GiB 超过预算 {max_io_gib:g} GiB"

    free = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True, timeout=10)

    available = float(free.strip()) * 2**20
    if not math.isfinite(available) or available < 0:
        raise ValueError("设备查询未返回有效的空闲显存容量")
    if size + 2 * 2**30 > available:
        return f"输入输出 {size / 2**30:.2f} GiB, 当前空闲 {available / 2**30:.2f} GiB, 需预留 2 GiB"
    return None


def is_oom(log):
    return bool(re.search(r"OutOfMemory|out of memory|cudaErrorMemoryAllocation|std::bad_alloc|could not be allocated",
                          log, re.IGNORECASE))


def normalize_config(config):
    config = dict(config)
    if "schema_version" not in config:
        config.update(schema_version=2, layouts=[config.pop("layout", "nchw")], cases=None, repeats=1,
                      reference=config.get("reference", "cudnn"), reference_layout=None)
    return config


def parse_options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "all"], default="fp32")
    parser.add_argument("--cases", type=int, nargs="+", help="指定原始 case 编号；默认测量全部用例")
    parser.add_argument("--layouts", choices=["nchw", "nhwc"], nargs="+", default=["nchw"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5, help="同一次后端准备完成后的 Event 采样数")
    parser.add_argument("--repeats", type=int, default=1, help="独立进程执行轮数，每轮重新初始化和选优")
    parser.add_argument("--check-cpu", action="store_true", help="在计时外启用 CPU 数值校验，默认关闭")
    parser.add_argument("--include-opt", action="store_true", help="将 opt 加入实现列表，兼容已有成功日志续跑")
    parser.add_argument("--kernels", nargs="+", choices=KERNELS, help="指定实现列表，每个 shape 和轮次轮换执行顺序")
    parser.add_argument("--reference", choices=KERNELS, default="cudnn", help="相对耗时的分母实现")
    parser.add_argument("--reference-layout", choices=["nchw", "nhwc"], help="分母布局，默认使用被比较项的布局")
    parser.add_argument("--timeout", type=float, default=1800, help="每个子进程超时秒数, 0 表示不限时")
    parser.add_argument("--max-io-gib", type=float, help="可选的输入输出容量上限；默认按当前空闲显存预留 2 GiB")
    args = parser.parse_args()

    if args.warmup < 0 or min(args.iterations, args.samples, args.repeats) < 1:
        parser.error("warmup 不得为负, iterations、samples、repeats 必须为正")
    if not math.isfinite(args.timeout) or args.timeout < 0:
        parser.error("timeout 必须是非负有限数")
    if args.max_io_gib is not None and (not math.isfinite(args.max_io_gib) or args.max_io_gib <= 0):
        parser.error("max-io-gib 必须是正的有限数")

    if args.cases is not None:
        args.cases = list(dict.fromkeys(args.cases))
        if any(case < 0 for case in args.cases):
            parser.error("case 编号不得为负")

    args.layouts = list(dict.fromkeys(args.layouts))
    if args.reference_layout and args.reference_layout not in args.layouts:
        parser.error("--reference-layout 必须包含在 --layouts 中")
    args.kernels = args.kernels or ["cudnn", "baseline", "pytorch"]

    if args.include_opt and "opt" not in args.kernels:
        args.kernels.append("opt")

    canonical = [canonical_kernel(kernel) for kernel in args.kernels]
    if len(set(canonical)) != len(canonical):
        parser.error("--kernels 含重复实现或等价的 cuDNN 别名")
    if canonical_kernel(args.reference) not in canonical:
        parser.error("--reference 必须包含在实现列表中")

    args.reference = args.kernels[canonical.index(canonical_kernel(args.reference))]
    args.output = args.output or args.build / f"dwconv-compare-full-{args.dtype}"
    return parser, args


def run_case(shape, dtype, kernel, layout, repeat, args, binary):
    """每次最多运行一个子进程; run() 等其结束后才返回。"""
    row = dict(zip(SHAPE_FIELDS, shape))
    row.update(dtype=dtype, kernel=kernel, layout=layout, repeat=repeat, status="error", checked=False,
               timing_mode="event", warmup=args.warmup, iterations=args.iterations, samples=args.samples)

    if layout == "nhwc" and kernel != "pytorch" and not kernel.startswith("cudnn"):
        row.update(status="unsupported", reason="该实现的基准入口仅支持 NCHW")
        return row

    path = args.output / f"case-{shape[0]}-{dtype}-{kernel}-{layout}-r{repeat}.log"
    legacy = args.output / f"case-{shape[0]}-{kernel}.log"
    source = path

    if not path.exists() and args.dtype != "all" and args.layouts == ["nchw"] and args.repeats == 1 and legacy.exists():
        source = legacy

    log = source.read_text() if source.exists() else ""
    record = None
    reason = "缺少有效的结构化结果"
    expected = dict(case=shape[0], kernel=kernel, dtype=dtype, layout=layout, warmup=args.warmup,
                    iterations=args.iterations, samples=args.samples, checked=args.check_cpu)

    if log:
        try:
            record = parse_result(log, cached_returncode(log), **expected)
        except (ValueError, TypeError, KeyError) as error:
            reason = str(error)

    # 成功和明确不支持可复用；失败重新检查，显存不足也可能在空闲容量恢复后成功。
    if record and record["status"] == "error":
        record = None

    if record is None:
        try:
            memory_reason = memory_skip_reason(shape, dtype, args.max_io_gib)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            row["reason"] = f"显存检查失败：{error}"
            return row
        if memory_reason:
            path.write_text(f"[MEMORY_SKIP] {memory_reason}\n")
            row.update(status="memory_skipped", reason=memory_reason)
            return row

        command = [str(binary), "--case", str(shape[0]), "--kernel", kernel, "--dtype", dtype, "--layout", layout,
                   "--warmup", str(args.warmup), "--iterations", str(args.iterations), "--samples", str(args.samples)]

        if args.check_cpu:
            command.append("--check-cpu")
        try:
            result = invoke(command, args.timeout or None)
        except OSError as error:
            result = subprocess.CompletedProcess(command, 127, "", str(error))

        log = process_log(command, result)
        path.write_text(log)

        try:
            record = parse_result(log, result.returncode, **expected)
        except (ValueError, TypeError, KeyError) as error:
            reason = str(error)

    if record and record["status"] == "pass":
        row.update(status="ok", **{key: record[key] for key in (
            "us", "samples_us", "min_us", "max_us", "prepare_ms", "post_ms", "host_layout_ms", "algo", "engine",
            "workspace_bytes", "checked")})
    elif record and record["status"] == "skip":
        row.update(status="unsupported", reason=record["reason"])
    elif is_oom(log):
        row.update(status="oom_skipped", reason=record.get("reason", "GPU 或主机内存不足") if record else "GPU 或主机内存不足")
    else:
        row["reason"] = record["reason"] if record else reason
        logging.error("case=%s %s %s %s repeat=%s 失败：%s，见 %s",
                      shape[0], dtype, kernel, layout, repeat, row["reason"], path)
    return row


def main():
    parser, args = parse_options()
    binary = benchmark_binary(args.build)
    if not binary.is_file():
        parser.error(f"未找到基准程序：{binary}；请先构建 dwconv_compare_bench")

    # --list-cases 仅输出主机元数据；非法编号在设备查询和测速之前拒绝。
    listing = subprocess.run([str(binary), "--list-cases"], check=True, capture_output=True, text=True).stdout
    shapes = re.findall(r"case=(\d+) NCHW=(\d+),(\d+),(\d+),(\d+) K=(\d+)x(\d+) stride=(\d+)x(\d+) padding=(\d+)x(\d+) bias=(\d+)", listing)
    if not shapes:
        parser.error("基准程序没有返回可解析的 shape 列表")

    by_id = {int(shape[0]): shape for shape in shapes}
    if args.cases is not None:
        unknown = [case for case in args.cases if case not in by_id]
        if unknown:
            parser.error(f"未知 case 编号：{unknown}")
        shapes = [by_id[case] for case in args.cases]

    config = dict(schema_version=2, binary=str(binary.resolve()), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  warmup=args.warmup, iterations=args.iterations, samples=args.samples, repeats=args.repeats,
                  timing_mode="event", checked=args.check_cpu, dtype=args.dtype, layouts=args.layouts,
                  cases=args.cases, kernels=args.kernels, reference=args.reference, reference_layout=args.reference_layout)
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = args.output / "config.json"
    if manifest.exists():
        previous = normalize_config(json.loads(manifest.read_text()))
        expected = dict(config, kernels=previous["kernels"])
        if previous != expected or not set(previous["kernels"]).issubset(config["kernels"]):
            raise RuntimeError("输出目录已有不同配置，请指定新的 --output")

    manifest.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    (args.output / "memory-policy.json").write_text(json.dumps(
        dict(mode="free_minus_reserve", max_io_gib=args.max_io_gib, reserve_gib=2), indent=2) + "\n")

    device = subprocess.run([str(args.build / "tools/device_query")], check=True, capture_output=True, text=True)

    (args.output / "device.txt").write_text(device.stdout)
    (args.output / "cases.txt").write_text(listing)

    logging.info("计时方式=event, 独立进程轮数=%s, 每进程采样=%s, CPU 校验=%s",
                 args.repeats, args.samples, args.check_cpu)

    dtypes = ["fp32", "fp16"] if args.dtype == "all" else [args.dtype]
    combinations = list(itertools.product(args.kernels, args.layouts))

    rows = []
    with (args.output / "samples.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()

        for index, shape in enumerate(shapes):
            for dtype in dtypes:
                for repeat in range(args.repeats):
                    offset = (index + repeat) % len(combinations)
                    order = combinations[offset:] + combinations[:offset]
                    group = [run_case(shape, dtype, kernel, layout, repeat, args, binary) for kernel, layout in order]
                    annotate_references(group, args.reference, args.reference_layout)

                    for row in group:
                        sample = {key: row[key] for key in SAMPLE_FIELDS if key in row}
                        if "samples_us" in sample:
                            sample["samples_us"] = json.dumps(sample["samples_us"])
                        writer.writerow(sample)

                    file.flush()
                    rows.extend(group)
                    logging.info("完成 %d/%d shapes：case=%s %s repeat=%s %s", index + 1, len(shapes), shape[0], dtype, repeat,
                                 " ".join(f"{row['kernel']}/{row['layout']}={row.get('us', row['status'])}" for row in group))

    summaries = write_reports(rows, args.output, repeats=args.repeats,
                              reference=args.reference, reference_layout=args.reference_layout)

    failed = sum(row["status"] == "error" for row in summaries)
    skipped = sum(row["status"] in {"unsupported", "oom_skipped", "memory_skipped"} for row in summaries)
    logging.info("%s %d 个配置，失败 %d 项，跳过 %d 项；结果：%s", "[失败]" if failed else "[SUCCESS]",
                 len(summaries), failed, skipped, args.output)

    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise SystemExit(main())
