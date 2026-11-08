"""串行记录 cuDNN 代表用例的实际 kernel 序列; trace 耗时不作为正式基准结果。"""

import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if __package__:
    from .dwconv_cudnn_profile import ROOT, KERNELS, invoke, parse_profile, write_json
else:
    from dwconv_cudnn_profile import ROOT, KERNELS, invoke, parse_profile, write_json

DEFAULT_CASES = [7, 5, 0, 401, 8, 529, 15, 16, 40, 22, 407, 410]
KEYS = ["case", "dtype", "layout", "kernel"]
PLAN_FIELDS = ["selected_candidate", "candidate_count", "cudnn_version", "math_type", "plan_workspace_bytes"]
FIELDS = KEYS + ["status", "reason", "captured_kernels", "prepare_ms", "algo", "engine", "workspace_bytes"] + PLAN_FIELDS + ["directory"]
TRACE_FIELDS = ["index", "name", "mangled_name", "short_name", "start_ns", "duration_ns", "correlation_id",
                "device_id", "context_id", "stream_id", "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z",
                "static_shared_bytes", "dynamic_shared_bytes"]
TRACE_OPTIONS = ["--trace=cuda", "--sample=none", "--cpuctxsw=none", "--capture-range=cudaProfilerApi",
                 "--capture-range-end=stop", "--cuda-memory-usage=false"]


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition):
    if not condition:
        raise ValueError("续跑缓存的日志、配置、校验值或 kernel trace 不一致")


def read_trace(path):
    """只读 SQLite, 保留逐 launch 顺序及完整名称，不采用按名称聚合的统计。"""
    query = """SELECT names.value AS name, mangled.value AS mangled_name, short.value AS short_name,
        k.start AS start_ns, k.end-k.start AS duration_ns, k.correlationId AS correlation_id,
        k.deviceId AS device_id, k.contextId AS context_id, k.streamId AS stream_id,
        k.gridX AS grid_x, k.gridY AS grid_y, k.gridZ AS grid_z,
        k.blockX AS block_x, k.blockY AS block_y, k.blockZ AS block_z,
        k.staticSharedMemory AS static_shared_bytes, k.dynamicSharedMemory AS dynamic_shared_bytes
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS names ON names.id=k.demangledName
        LEFT JOIN StringIds AS mangled ON mangled.id=k.mangledName
        LEFT JOIN StringIds AS short ON short.id=k.shortName
        ORDER BY k.start,k.correlationId"""
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as database:
        database.row_factory = sqlite3.Row
        kernels = [dict(index=index, **dict(row)) for index, row in enumerate(database.execute(query))]
        count = database.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    if not kernels or len(kernels) != count:
        raise ValueError("SQLite 中无 kernel, 或 StringIds 不完整")
    for kernel in kernels:
        if (not isinstance(kernel["name"], str) or not kernel["name"].strip() or kernel["duration_ns"] <= 0
                or any(kernel[key] <= 0 for key in ("grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z"))
                or min(kernel["static_shared_bytes"], kernel["dynamic_shared_bytes"]) < 0):
            raise ValueError("SQLite kernel 记录包含无效名称、尺寸或耗时")
    return kernels


def final_plan(log, record):
    lines = [line for line in log.splitlines() if line.startswith("[cuDNN 最终方案] ")]
    if len(lines) != 1:
        raise ValueError("日志缺少唯一的 cuDNN 最终方案")
    values = dict(re.findall(r"(\w+)=([^\s]+)", lines[0]))
    selected, count = map(int, values["candidate"].split("/"))
    if not 0 <= selected < count or any(str(record[key]) != values[key] for key in ("algo", "engine", "layout")):
        raise ValueError("最终方案候选或配置与 DWCONV_PROFILE 不一致")
    if int(values["workspace"]) != record["workspace_bytes"]:
        raise ValueError("最终方案 workspace 与 DWCONV_PROFILE 不一致")
    result = dict(selected_candidate=selected, candidate_count=count, cudnn_version=int(values["version"]),
                  math_type=int(values["math_type"]), plan_workspace_bytes=int(values["plan_workspace"]))
    if result["plan_workspace_bytes"] < 0 or result["plan_workspace_bytes"] > record["workspace_bytes"]:
        raise ValueError("最终方案 plan workspace 超出分配量")
    return result


def save_reports(output, rows):
    write_json(output / "summary.json", rows)
    inventory, unique = [], {}
    for row in rows:
        if row["status"] != "pass":
            continue
        context = {key: row[key] for key in KEYS + PLAN_FIELDS + ["directory"]}
        for kernel in row["kernels"]:
            inventory.append(dict(**{key: row[key] for key in KEYS}, **kernel))
            entry = unique.setdefault(kernel["name"], dict(name=kernel["name"], representative=dict(context, launch_index=kernel["index"]), configurations=[]))
            if context not in entry["configurations"]:
                entry["configurations"].append(context)
    write_json(output / "unique_kernels.json", list(unique.values()))
    for filename, fields, records in [("summary.csv", FIELDS, rows), ("kernel_inventory.csv", KEYS + TRACE_FIELDS, inventory)]:
        with (output / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)


def scan_one(args, expected):
    directory = args.output / "case-{case}-{dtype}-{kernel}-{layout}".format(**expected)
    command = [args.nsys, "profile", *TRACE_OPTIONS, "--output", str(directory / "profile"), str(args.binary),
               "--case", str(expected["case"]), "--kernel", expected["kernel"], "--dtype", expected["dtype"],
               "--layout", expected["layout"], "--warmup", str(args.warmup), "--profile"]
    if args.resume and (directory / "result.json").is_file():
        try:
            cached = json.loads((directory / "result.json").read_text())
            record = parse_profile((directory / "run.log").read_text(), expected, args.warmup)
            require(cached["status"] == record["status"] == "pass" and cached["capture_returncode"] == cached["export_returncode"] == 0)
            require(all(cached[key] == value for key, value in expected.items()))
            require(all(cached[key] == record[key] for key in ["prepare_ms", "algo", "engine", "workspace_bytes"]))
            require(json.loads((directory / "commands.json").read_text())["capture"] == command)
            require(set(cached["checksums"]) == {"run.log", "profile.nsys-rep", "profile.sqlite", "trace.json"})
            require(all((directory / name).stat().st_size > 0 and checksum(directory / name) == digest for name, digest in cached["checksums"].items()))
            trace = json.loads((directory / "trace.json").read_text())
            require(trace["configuration"] == expected and trace["kernels"] == cached["kernels"] == read_trace(directory / "profile.sqlite"))
            require(cached["captured_kernels"] == len(trace["kernels"]))
            require(all(cached[key] == value for key, value in final_plan((directory / "run.log").read_text(), record).items()))
            logging.info("复用已验证结果 %s", directory.name)
            return cached
        except (OSError, ValueError, KeyError, sqlite3.Error):
            logging.warning("缓存无效，重新抓取 %s", directory.name)
    if directory.exists():
        retired = args.output / "retired"
        retired.mkdir(exist_ok=True)
        directory.rename(retired / (directory.name + datetime.now().strftime("-%Y%m%d-%H%M%S-%f")))
    directory.mkdir()
    row = dict(expected, status="error", reason="", captured_kernels=0, directory=str(directory))
    commands = {"capture": command}
    write_json(directory / "commands.json", commands)
    try:
        row["capture_returncode"] = invoke(command, directory / "run.log", args.timeout)
        log = (directory / "run.log").read_text(encoding="utf-8", errors="replace")
        record = parse_profile(log, expected, args.warmup)
        errors = [line for line in log.splitlines() if ("==ERROR==" in line or line.startswith("Error:"))
                  and "application returned an error code" not in line]
        if errors:
            raise RuntimeError("Nsight Systems 报告错误：" + " | ".join(errors))
        if record["status"] == "skip":
            row.update(status="skip", reason=record["reason"])
        else:
            if record["status"] == "error" or row["capture_returncode"] != 0:
                raise RuntimeError(record.get("reason") or f"nsys 退出码 {row['capture_returncode']}")
            report = directory / "profile.nsys-rep"
            if not report.is_file() or report.stat().st_size == 0:
                raise RuntimeError("没有非空 .nsys-rep 报告")
            export = [args.nsys, "export", "--type", "sqlite", "--output", str(directory / "profile.sqlite"), str(report)]
            commands["sqlite"] = export
            write_json(directory / "commands.json", commands)
            row["export_returncode"] = invoke(export, directory / "export.log", args.timeout)
            if row["export_returncode"] != 0:
                raise RuntimeError("nsys SQLite 导出失败")
            kernels = read_trace(directory / "profile.sqlite")
            row.update({key: record[key] for key in ["prepare_ms", "algo", "engine", "workspace_bytes"]})
            row.update(final_plan(log, record), status="pass", captured_kernels=len(kernels), kernels=kernels)
            write_json(directory / "trace.json", dict(schema_version=1, configuration=expected, kernels=kernels))
            row["checksums"] = {name: checksum(directory / name) for name in ["run.log", "profile.nsys-rep", "profile.sqlite", "trace.json"]}
    except subprocess.TimeoutExpired as error:
        row.update(status="error", reason=f"超时 {error.timeout} 秒，停止后续抓取", stop=True)
    except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error) as error:
        row.update(status="error", reason=str(error))
    write_json(directory / "result.json", row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/benchmarks/operators/dwconv/dwconv_compare_bench")
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--cases", type=int, nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--dtypes", nargs="+", choices=["fp32", "fp16"], default=["fp32", "fp16"])
    parser.add_argument("--layouts", nargs="+", choices=["nchw", "nhwc"], default=["nchw", "nhwc"])
    parser.add_argument("--kernels", nargs="+", choices=KERNELS, default=KERNELS)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / datetime.now().strftime("dwconv-cudnn-scan-%Y%m%d-%H%M%S"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=1800, help="每项超时秒数; 0 表示不限时")
    parser.add_argument("--resume", action="store_true", help="同配置续跑, 仅复用经日志、SQLite 和 trace 核验的成功项")
    args = parser.parse_args()
    if args.warmup < 0 or any(case < 0 for case in args.cases) or not math.isfinite(args.timeout) or args.timeout < 0:
        parser.error("case/warmup/timeout 必须是非负有限数")
    if any(len(getattr(args, key)) != len(set(getattr(args, key))) for key in ("cases", "dtypes", "layouts", "kernels")):
        parser.error("参数列表不得包含重复项")
    args.binary, args.output = args.binary.resolve(), args.output.resolve()
    args.nsys = shutil.which(args.nsys)
    if not args.binary.is_file() or args.nsys is None:
        parser.error("找不到基准程序或 nsys")
    identity = dict(schema_version=1, binary=str(args.binary), binary_sha256=checksum(args.binary), nsys=args.nsys,
                    cases=args.cases, dtypes=args.dtypes, layouts=args.layouts, kernels=args.kernels,
                    warmup=args.warmup, trace_options=TRACE_OPTIONS)
    if args.resume:
        try:
            if json.loads((args.output / "config.json").read_text()) != identity:
                parser.error("续跑配置或二进制 SHA256 不一致；请使用新输出目录")
        except (OSError, ValueError) as error:
            parser.error(f"无法读取续跑配置：{error}")
    elif args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录非空；同配置续跑请显式传入 --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "config.json", identity)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "scan.log", encoding="utf-8")])
    environment = args.output / "environment" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    environment.mkdir(parents=True)
    write_json(environment / "invocation.json", dict(arguments=vars(args), argv=sys.argv))
    checks = {"nsys-version.txt": [args.nsys, "--version"], "nvidia-smi.txt": ["nvidia-smi"],
              "git-head.txt": ["git", "rev-parse", "HEAD"], "git-status.txt": ["git", "status", "--short", "--branch"],
              "device.txt": [str(args.binary.parents[3] / "tools/device_query")]}
    write_json(environment / "commands.json", checks)
    for filename, command in checks.items():
        try:
            if invoke(command, environment / filename, 60) != 0:
                raise RuntimeError(f"必要环境检查失败：{filename}")
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            write_json(environment / "failure.json", {"reason": str(error)})
            logging.error("%s; 未开始 GPU 抓取", error)
            return 1
    rows = []
    for case, dtype, layout, kernel in itertools.product(args.cases, args.dtypes, args.layouts, args.kernels):
        row = scan_one(args, dict(case=case, dtype=dtype, layout=layout, kernel=kernel))
        rows.append(row)
        save_reports(args.output, rows)
        logging.info("%s %s: %d kernels %s", row["status"], row["directory"], row["captured_kernels"], row["reason"])
        if row.get("stop"):
            break
    counts = {status: sum(row["status"] == status for row in rows) for status in ("pass", "skip", "error")}
    logging.info("%s 完成=%d 跳过=%d 失败=%d; trace 耗时仅用于观察执行结构", "[失败]" if counts["error"] else "[SUCCESS]", *counts.values())
    return int(bool(counts["error"]))


if __name__ == "__main__":
    sys.exit(main())
