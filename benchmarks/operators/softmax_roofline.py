"""采集 Softmax FP32 的 Nsight Compute 指标并汇总 Roofline，仅依赖 Python 标准库。"""

import argparse
import csv
import io
import itertools
import logging
import math
from pathlib import Path
import shlex
import subprocess

METRICS = [
    "gpu__time_duration.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
    "sm__cycles_elapsed.avg.per_second",
    "dram__bytes.sum",
    "dram__bytes.sum.peak_sustained",
    "dram__cycles_elapsed.avg.per_second",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "launch__waves_per_multiprocessor",
    "launch__registers_per_thread",
    "lts__t_sector_hit_rate.pct",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
]

# 可选诊断指标不可用时保留空值，不用零值冒充硬件测量。
DIAGNOSTICS = {
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_pct": "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "xu_pct": "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed",
    "fma_pipe_pct": "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "eligible_warps_per_scheduler": "smsp__warps_eligible.avg.per_cycle_active",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "registers_per_thread": "launch__registers_per_thread",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "long_scoreboard_cycles_per_issue": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "barrier_cycles_per_issue": "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "wait_cycles_per_issue": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
}


def read_capture(path):
    """读取 ncu --page raw --csv 的表头、单位行及唯一一次 kernel 记录。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
    except StopIteration as error:
        raise ValueError(f"{path} 没有 Nsight Compute CSV 表头，请检查权限或采样错误") from error
    records = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    if len(records) != 2 or records[0]["ID"] or not records[1]["ID"]:
        raise ValueError(f"{path} 应包含单位行和恰好一个 kernel 记录")
    return records[0], records[1]


def metric(record, key):
    value = float(record[key].replace(",", ""))
    if not math.isfinite(value):
        raise ValueError(f"无效指标 {key}={value}")
    return value


def summarize(units, record, cols, rows=512):
    time_units = {
        "second": 1.0, "msecond": 1e-3, "usecond": 1e-6, "nsecond": 1e-9,
        "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9,
    }
    seconds = metric(record, "gpu__time_duration.sum") * time_units[units["gpu__time_duration.sum"]]
    flops = sum(
        multiplier * metric(record, f"smsp__sass_thread_inst_executed_op_{op}_pred_on.sum")
        for op, multiplier in (("fadd", 1), ("fmul", 1), ("ffma", 2))
    )
    dram_bytes = metric(record, "dram__bytes.sum")
    peak_fp32 = (
        2 * metric(record, "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained")
        * metric(record, "sm__cycles_elapsed.avg.per_second")
    )
    peak_dram = (
        metric(record, "dram__bytes.sum.peak_sustained")
        * metric(record, "dram__cycles_elapsed.avg.per_second")
    )
    if min(seconds, flops, dram_bytes, peak_fp32, peak_dram) <= 0:
        raise ValueError("Roofline 核心计数必须为正数")
    ai = flops / dram_bytes
    achieved = flops / seconds
    roof = min(peak_fp32, ai * peak_dram)
    result = {
        "rows": rows,
        "cols": cols,
        "device": record.get("device__attribute_display_name", record.get("Device")),
        "cc": record.get("CC"),
        "kernel": record["Kernel Name"],
        "kernel_us": seconds * 1e6,
        "fp32_flops": flops,
        "dram_bytes": dram_bytes,
        "ai_dram_flop_per_byte": ai,
        "achieved_gflops": achieved / 1e9,
        "peak_fp32_gflops": peak_fp32 / 1e9,
        "peak_dram_gbytes_per_s": peak_dram / 1e9,
        "ridge_flop_per_byte": peak_fp32 / peak_dram,
        "roof_gflops": roof / 1e9,
        "roof_efficiency_pct": 100 * achieved / roof,
        "fp32_peak_pct": 100 * achieved / peak_fp32,
        "dram_byte_peak_pct": 100 * dram_bytes / seconds / peak_dram,
        "lower_bound_us": max(flops / peak_fp32, dram_bytes / peak_dram) * 1e6,
        "roof_headroom_factor": roof / achieved,
        "roof_side": "memory" if ai * peak_dram < peak_fp32 else "compute",
    }
    for name, key in DIAGNOSTICS.items():
        try:
            result[name] = metric(record, key)
        except (KeyError, ValueError):
            result[name] = None
            logging.warning("当前设备或工具未提供诊断指标 %s", key)
    return result


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, help="默认使用基础基准；指定 --variants 时使用 cluster 对照基准")
    parser.add_argument("--variants", nargs="+", choices=["online", "block512", "block1024", "cluster2", "cluster4"],
                        help="显式实现列表；所有实现必须支持所选 cols")
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--cols", type=int, nargs="+", default=[1024, 8192, 65536])
    parser.add_argument("--rows", type=int, nargs="+", default=[512])
    parser.add_argument("--output", type=Path, default=root / "build/softmax-roofline")
    parser.add_argument("--summarize-only", action="store_true", help="仅读取输出目录中已有的 ncu CSV")
    args = parser.parse_args()
    if args.bench is None:
        name = "softmax_cluster_bench" if args.variants else "softmax_bench"
        args.bench = root / "build/benchmarks/operators" / name
    if args.variants and len(set(args.variants)) != len(args.variants):
        parser.error("实现不能重复")
    if args.variants and any(cols > 32768 or (cols > 16384 and any(v in {"block512", "cluster2"} for v in args.variants)) for cols in args.cols):
        parser.error("显式实现不支持所选列数；block512/cluster2 最多 16384，其余对照最多 32768")
    if any(cols <= 0 for cols in args.cols) or len(set(args.cols)) != len(args.cols):
        parser.error("列数必须为不重复的正整数")
    if any(rows <= 0 or rows > 2147483647 for rows in args.rows) or len(set(args.rows)) != len(args.rows):
        parser.error("行数必须为不重复的正整数且不超过 INT_MAX")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for rows, cols, variant in itertools.product(args.rows, args.cols, args.variants or [None]):
        # 保持原 512 行结果的文件名兼容；其他形状包含完整尺寸。
        case_id = str(cols) if rows == 512 else f"{rows}x{cols}"
        if variant:
            case_id = f"{variant}-{rows}x{cols}"
        capture = args.output / f"ncu-{case_id}.csv"
        if not args.summarize_only:
            command = [
                args.ncu, "--metrics", ",".join(METRICS),
                "--cache-control", "all", "--clock-control", "base",
                # 显式基准先启动一次以完成惰性加载，采样此 kernel，避免图内采样差异。
                "--launch-skip", "0" if variant else "20", "--launch-count", "1",
                "--page", "raw", "--csv", "--print-units", "base", "--print-fp",
                "--log-file", str(capture.resolve()), str(args.bench.resolve()), "--cols", str(cols),
                "--rows", str(rows),
            ]
            if variant:
                command.extend(["--variant", variant, "--output", str((args.output / f"event-{case_id}.csv").resolve())])
            logging.info("[阶段] 采集 rows=%d cols=%d variant=%s", rows, cols, variant or "auto")
            (args.output / f"command-{case_id}.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
            with (args.output / f"application-{case_id}.log").open("w", encoding="utf-8") as output:
                subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
        summary = summarize(*read_capture(capture), cols, rows)
        if variant:
            summary["variant"] = variant
            _, record = read_capture(capture)
            summary["sm_clock_hz"] = metric(record, "sm__cycles_elapsed.avg.per_second")
            summary["dram_clock_hz"] = metric(record, "dram__cycles_elapsed.avg.per_second")
            # NCU 的 waves_per_sm 不能替代 cluster 驻留容量；旧版工具缺少字段时保留空值。
            summary["cluster_blocks"] = None
            summary["active_clusters"] = None
            summary["cluster_waves"] = None
            if variant.startswith("cluster"):
                try:
                    summary["cluster_blocks"] = metric(record, "launch__cluster_size")
                    summary["active_clusters"] = metric(record, "launch__cluster_max_active")
                    if summary["active_clusters"] > 0:
                        summary["cluster_waves"] = rows / summary["active_clusters"]
                except (KeyError, ValueError):
                    logging.warning("当前 NCU 未提供 cluster 驻留指标")
        summaries.append(summary)
        logging.info(
            "[结果] rows=%d cols=%d kernel=%.3f us FP32=%.3f GFLOP/s AI_DRAM=%.3f "
            "roof=%.3f GFLOP/s roof_efficiency=%.2f%% lower_bound=%.3f us headroom=%.2fx roof_side=%s",
            rows, cols, summary["kernel_us"], summary["achieved_gflops"], summary["ai_dram_flop_per_byte"],
            summary["roof_gflops"], summary["roof_efficiency_pct"], summary["lower_bound_us"],
            summary["roof_headroom_factor"], summary["roof_side"],
        )
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    logging.info("[SUCCESS] 汇总已保存到 %s；roof_side 是模型限制侧，不是实际瓶颈结论", args.output / "summary.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        logging.error("[FAILED] %s", error)
        raise SystemExit(1) from error
