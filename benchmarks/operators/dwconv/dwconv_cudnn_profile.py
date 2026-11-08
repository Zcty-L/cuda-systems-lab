"""串行抓取 DWConv cuDNN 配置的 Nsight Compute 完整指标与详细报告。"""

import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
KERNELS = ["cudnn", "cudnn-heuristic", "cudnn-separate", "cudnn-fused"]
FIELDS = ["case", "dtype", "layout", "kernel", "status", "reason", "captured_kernels",
          "prepare_ms", "algo", "engine", "workspace_bytes", "directory"]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def invoke(command, path, timeout):
    """同步等待子进程结束；超时保留日志，由调用方停止后续抓取。"""
    logging.info("执行 %s", shlex.join(map(str, command)))
    with path.open("w", encoding="utf-8") as output:
        try:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=output,
                                    stderr=subprocess.STDOUT, timeout=timeout or None, check=False)
        except subprocess.TimeoutExpired:
            output.write(f"\n[TIMEOUT] 单项超过 {timeout} 秒，停止后续抓取\n")
            raise
    return result.returncode


def parse_profile(log, expected, warmup):
    records = [json.loads(line[len("DWCONV_PROFILE "):]) for line in log.splitlines()
               if line.startswith("DWCONV_PROFILE ")]
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError("日志必须包含唯一的 DWCONV_PROFILE JSON object")
    record = records[0]
    if (type(record.get("schema_version")) is not int or record["schema_version"] != 1
            or record.get("timing_mode") != "profile"):
        raise ValueError("不支持的 profile schema 或 timing_mode")
    if (type(record.get("case")) is not int
            or any(record.get(key) != value for key, value in expected.items())
            or type(record.get("warmup")) is not int or record["warmup"] != warmup):
        raise ValueError("DWCONV_PROFILE 的 case/kernel/dtype/layout/warmup 与请求不一致")
    if record.get("status") not in {"pass", "skip", "error"} or "us" in record:
        raise ValueError("非法 profile 状态或混入测速结果")
    if record["status"] != "pass":
        if not isinstance(record.get("reason"), str) or not record["reason"].strip():
            raise ValueError("skip/error 缺少明确原因")
        return record
    value = record.get("prepare_ms")
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError("prepare_ms 必须是非负有限数")
    if any(type(record.get(key)) is not int for key in ("algo", "engine", "workspace_bytes")):
        raise ValueError("缺少整数类型的 algo/engine/workspace_bytes")
    if record["workspace_bytes"] < 0:
        raise ValueError("workspace_bytes 不得为负")
    return record


def kernel_inventory(path, expected):
    """raw 页按 kernel ID 去重，同时保留完整 kernel 名称。"""
    header, records = None, {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.reader(source):
            if "ID" in row and "Kernel Name" in row:
                header = row
                continue
            if header is None or len(row) != len(header):
                continue
            record = dict(zip(header, row))
            if record.get("ID", "").isdigit() and record.get("Kernel Name"):
                identifier = tuple(record.get(key, "") for key in ("Process ID", "ID"))
                records[identifier] = dict(expected, id=record["ID"], name=record["Kernel Name"],
                                           process_id=record.get("Process ID", ""))
    return list(records.values())


def split_sass(directory, inventory):
    """按实际 launch 顺序核对名称，并保存每个 kernel 的完整地址与指令。"""
    blocks = []
    with (directory / "sass.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.reader(source):
            if row and row[0] == "Kernel Name":
                if len(row) != 2 or not row[1].strip():
                    raise ValueError("SASS CSV 缺少完整 kernel 名称")
                blocks.append({"name": row[1], "instructions": []})
            elif row and row[0].startswith("0x"):
                if not blocks or len(row) != 2 or not row[1].strip():
                    raise ValueError("SASS CSV 包含无效指令")
                int(row[0], 16)
                blocks[-1]["instructions"].append(row)
    if len(blocks) != len(inventory) or any(block["name"] != item["name"] or not block["instructions"]
                                           for block, item in zip(blocks, inventory)):
        raise ValueError("SASS 分块的顺序、名称或指令与 raw kernel inventory 不一致")
    (directory / "sass").mkdir(exist_ok=True)
    for block, item in zip(blocks, inventory):
        target = directory / "sass" / f"kernel-{item['id']}.sass"
        target.write_text("// Kernel: " + block["name"] + "\n" + "".join(
            address + "  " + instruction.rstrip() + "\n" for address, instruction in block["instructions"]), encoding="utf-8")
        item["sass_file"] = str(target)


def save_summary(output, rows):
    write_json(output / "summary.json", rows)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    inventory = [kernel for row in rows for kernel in row.get("kernels", [])]
    write_json(output / "kernel_inventory.json", inventory)
    with (output / "kernel_inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "dtype", "layout", "kernel", "id", "name", "process_id", "sass_file"])
        writer.writeheader()
        writer.writerows(inventory)


def profile_one(args, expected):
    stem = "case-{case}-{dtype}-{kernel}-{layout}".format(**expected)
    directory = args.output / stem
    directory.mkdir()
    report = directory / "profile.ncu-rep"
    command = [args.ncu, "--set", "full", "--profile-from-start", "off", "--replay-mode", "kernel",
               "--cache-control", "all", "--clock-control", "none", "--target-processes", "application-only",
               "--rename-kernels", "off", "--print-kernel-base", "demangled",
               "--call-stack", "--import-source", "yes", "--import-sass", "yes", "--page", "details", "--print-details", "all",
               "--print-rule-details", "--export", report.with_suffix(""), args.binary,
               "--case", str(expected["case"]), "--kernel", expected["kernel"], "--dtype", expected["dtype"],
               "--layout", expected["layout"], "--warmup", str(args.warmup), "--profile"]
    if getattr(args, "cudnn_candidate", None) is not None:
        command += ["--cudnn-candidate", str(args.cudnn_candidate)]
    commands = {"capture": list(map(str, command))}
    write_json(directory / "command.json", commands)
    row = dict(expected, status="error", reason="", captured_kernels=0, directory=str(directory))
    try:
        returncode = invoke(command, directory / "run.log", args.timeout)
        log = (directory / "run.log").read_text(encoding="utf-8", errors="replace")
        record = parse_profile(log, expected, args.warmup)
        if record["status"] == "skip":
            errors = [line for line in log.splitlines() if "==ERROR==" in line
                      and "The application returned an error code" not in line]
            if errors:
                raise RuntimeError("ncu 报告错误，不能作为正常 unsupported 跳过")
            row.update(status="skip", reason=record["reason"])
            return row
        if record["status"] == "error":
            raise RuntimeError(record["reason"])
        if returncode != 0 or "==ERROR==" in log:
            raise RuntimeError(f"ncu 抓取失败，退出码 {returncode}")
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError("程序返回成功，但没有非空 .ncu-rep 报告")
        pages = {
            "details.txt": ["--page", "details", "--print-details", "all", "--print-rule-details"],
            "raw.csv": ["--page", "raw", "--csv", "--print-units", "base"],
            "sass-metrics.txt": ["--page", "source", "--print-source", "sass"],
            "sass.txt": ["--page", "source", "--print-source", "sass", "--metrics", "regex:^$"],
            "sass.csv": ["--page", "source", "--print-source", "sass", "--metrics", "regex:^$", "--csv"],
            "session.txt": ["--page", "session"],
        }
        for filename, options in pages.items():
            export = [args.ncu, "--import", report, "--rename-kernels", "off", "--print-kernel-base", "demangled", *options]
            commands[filename] = list(map(str, export))
            write_json(directory / "command.json", commands)
            if invoke(export, directory / filename, args.timeout) != 0:
                raise RuntimeError(f"报告页面导出失败：{filename}")
            if (directory / filename).stat().st_size == 0:
                raise RuntimeError(f"报告页面为空：{filename}")
        inventory = kernel_inventory(directory / "raw.csv", expected)
        if not inventory:
            raise RuntimeError("raw.csv 未包含任何可识别的 GPU kernel，不能视为抓取成功")
        split_sass(directory, inventory)
        row.update({key: record[key] for key in ("prepare_ms", "algo", "engine", "workspace_bytes")})
        row.update(status="pass", captured_kernels=len(inventory), kernels=inventory)
    except subprocess.TimeoutExpired as error:
        row.update(reason=f"超时：{error.timeout} 秒，停止后续抓取", stop=True)
    except (OSError, ValueError, RuntimeError, csv.Error) as error:
        row["reason"] = str(error)
    return row


def scan_helpers():
    # 延迟导入：scan 复用本文件的协议与命令执行函数。
    if __package__:
        from . import dwconv_cudnn_scan as scan
    else:
        import dwconv_cudnn_scan as scan
    return scan


def device_identity(directory):
    paths = sorted((directory / "environment").rglob("device.txt"))
    if not paths:
        raise ValueError(f"缺少设备查询记录：{directory}")
    identities = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        names = re.findall(r"^\s*Name\s+(.+)$", text, re.M)
        capabilities = re.findall(r"^\s*Compute capability\s+(\d+\.\d+)\s*$", text, re.M)
        if len(names) != 1 or len(capabilities) != 1:
            raise ValueError(f"设备记录须包含唯一的 GPU 名称及 SM：{path}")
        identities.append((names[0].strip(), capabilities[0]))
    if len(set(identities)) != 1:
        raise ValueError(f"设备记录发生变化：{directory}")
    return dict(name=identities[0][0], sm=identities[0][1])


def load_scan(directory, binary_sha256):
    scan = scan_helpers()
    config = json.loads((directory / "config.json").read_text())
    if config.get("schema_version") != 1 or config.get("binary_sha256") != binary_sha256:
        raise ValueError("scan schema 或基准二进制 SHA256 不一致")
    expected = [dict(zip(scan.KEYS, values)) for values in itertools.product(
        config["cases"], config["dtypes"], config["layouts"], config["kernels"])]
    rows = json.loads((directory / "summary.json").read_text())
    identities = [tuple(row[key] for key in scan.KEYS) for row in rows]
    if (len(rows) != len(expected) or len(set(identities)) != len(rows)
            or set(identities) != {tuple(item[key] for key in scan.KEYS) for item in expected}):
        raise ValueError("scan 未完成全部请求组合，或 summary 有重复/额外配置")
    for row in rows:
        if row["status"] not in {"pass", "skip"}:
            raise ValueError(f"scan 仍有失败项，不能开始补采：{row}")
        context = {key: row[key] for key in scan.KEYS}
        folder = Path(row["directory"])
        log = (folder / "run.log").read_text()
        record = parse_profile(log, context, config["warmup"])
        if record["status"] != row["status"]:
            raise ValueError(f"scan 日志状态与 summary 不一致：{folder}")
        if row["status"] == "skip":
            if row.get("reason") != record["reason"]:
                raise ValueError(f"scan skip 原因不一致：{folder}")
            continue
        if row.get("capture_returncode") != 0 or row.get("export_returncode") != 0:
            raise ValueError(f"scan 抓取/导出退出码不为零：{folder}")
        if set(row["checksums"]) != {"run.log", "profile.nsys-rep", "profile.sqlite", "trace.json"}:
            raise ValueError(f"scan 校验文件集合不完整：{folder}")
        for name, digest in row["checksums"].items():
            path = folder / name
            if path.stat().st_size == 0 or scan.checksum(path) != digest:
                raise ValueError(f"scan 原始文件校验失败：{path}")
        trace = json.loads((folder / "trace.json").read_text())
        kernels = scan.read_trace(folder / "profile.sqlite")
        if (trace.get("schema_version") != 1 or trace["configuration"] != context
                or trace["kernels"] != kernels or row["kernels"] != kernels
                or row["captured_kernels"] != len(kernels)):
            raise ValueError(f"scan trace/SQLite/summary 不一致：{folder}")
        for key, value in {**scan.final_plan(log, record), **{key: record[key] for key in
                           ("prepare_ms", "algo", "engine", "workspace_bytes")}}.items():
            if row[key] != value:
                raise ValueError(f"scan 最终方案 {key} 不一致：{folder}")
    return config, rows, device_identity(directory)


def load_reused_reports(directories, target_device, target_versions):
    scan = scan_helpers()
    reused, evidence = {}, []
    for directory in directories:
        directory = directory.resolve()
        if device_identity(directory) != target_device:
            raise ValueError(f"复用报告与 scan 的 GPU/SM 不同：{directory}")
        config = json.loads((directory / "config.json").read_text())
        rows = json.loads((directory / "summary.json").read_text())
        for row in rows:
            if row["status"] == "skip":
                continue
            if row["status"] != "pass":
                raise ValueError(f"复用报告包含失败项：{directory}")
            context = {key: row[key] for key in scan.KEYS}
            folder = Path(row["directory"])
            log = (folder / "run.log").read_text()
            record = parse_profile(log, context, config["arguments"]["warmup"])
            if record["status"] != "pass" or "==ERROR==" in log:
                raise ValueError(f"复用报告的原始日志未成功：{folder}")
            plan = scan.final_plan(log, record)
            if plan["cudnn_version"] not in target_versions:
                raise ValueError(f"复用报告的 cuDNN 版本与 scan 不同：{folder}")
            report = folder / "profile.ncu-rep"
            if not report.is_file() or report.stat().st_size == 0:
                raise ValueError(f"复用报告缺少非空 NCU 文件：{folder}")
            inventory = kernel_inventory(folder / "raw.csv", context)
            if not inventory or len(inventory) != row["captured_kernels"]:
                raise ValueError(f"复用报告 kernel 数不一致：{folder}")
            for item in inventory:
                path = folder / "sass" / f"kernel-{item['id']}.sass"
                lines = path.read_text().splitlines()
                if len(lines) < 2 or lines[0] != "// Kernel: " + item["name"] or not lines[1].startswith("0x"):
                    raise ValueError(f"复用 SASS 缺失或名称不一致：{path}")
                reused.setdefault(item["name"], dict(name=item["name"], sass_file=str(path),
                                  ncu_report=str(report), source_configuration=context,
                                  source="reused", cudnn_version=plan["cudnn_version"]))
        evidence.append(dict(directory=str(directory), binary_sha256=config["binary_sha256"],
                             device=target_device, cudnn_versions=sorted(target_versions),
                             basis="显式复用; GPU/SM、cuDNN 版本、原始 kernel 清单与 SASS 已核验。允许包装层变更造成基准二进制 SHA 不同，复用的是相同环境的 cuDNN 预编译 kernel。"))
    return reused, evidence


def capture_scan_results(args):
    """轻量扫描完成后，仅为尚无 SASS 的完整符号选择代表配置。"""
    scan = scan_helpers()
    config, scanned, device = load_scan(args.scan_results, hashlib.sha256(args.binary.read_bytes()).hexdigest())
    if device_identity(args.output) != device:
        raise ValueError("当前设备与 scan 不一致")
    versions = {row["cudnn_version"] for row in scanned if row["status"] == "pass"}
    covered, evidence = load_reused_reports(args.reuse_reports, device, versions)
    targets = {kernel["name"] for row in scanned if row["status"] == "pass" for kernel in row["kernels"]}
    initially_reused = targets & covered.keys()
    rows, attempted = [], set()

    def persist():
        missing = sorted(targets - covered.keys())
        save_summary(args.output, rows)
        write_json(args.output / "coverage.json", dict(schema_version=1, scan_results=str(args.scan_results),
                   scan_binary_sha256=config["binary_sha256"], device=device, cudnn_versions=sorted(versions),
                   target_count=len(targets), reused_count=len(initially_reused),
                   new_count=len((targets & covered.keys()) - initially_reused),
                   covered_count=len(targets & covered.keys()), missing=missing,
                   reused_reports=evidence, profiled_configs=[{key: row.get(key) for key in
                       scan.KEYS + ["status", "reason", "trace_match", "plan_match", "directory"]} for row in rows],
                   symbols=[covered[name] for name in sorted(targets & covered.keys())],
                   extra_symbols=sorted(covered.keys() - targets)))

    persist()
    while targets - covered.keys():
        missing = targets - covered.keys()
        candidates = [row for row in scanned if row["status"] == "pass"
                      and tuple(row[key] for key in scan.KEYS) not in attempted]
        if not candidates:
            break
        selected = max(candidates, key=lambda row: len(missing & {kernel["name"] for kernel in row["kernels"]}))
        if not missing & {kernel["name"] for kernel in selected["kernels"]}:
            break
        context = {key: selected[key] for key in scan.KEYS}
        attempted.add(tuple(context[key] for key in scan.KEYS))
        args.cudnn_candidate = selected["selected_candidate"]
        result = profile_one(args, context)
        result.update(scan_directory=selected["directory"], trace_match=False, plan_match=False)
        if result["status"] == "pass":
            try:
                folder = Path(result["directory"])
                log = (folder / "run.log").read_text()
                record = parse_profile(log, context, args.warmup)
                plan = scan.final_plan(log, record)
                result.update(plan)
                result["trace_match"] = [item["name"] for item in result["kernels"]] == [item["name"] for item in selected["kernels"]]
                result["plan_match"] = all(result[key] == selected[key] for key in
                                           scan.PLAN_FIELDS + ["algo", "engine", "workspace_bytes"])
                if not result["trace_match"] or not result["plan_match"]:
                    result["reason"] = "NCU 实际 kernel 序列或最终方案与 scan 不一致；仅将实际捕获的符号计入覆盖"
                    logging.error("%s：%s", folder.name, result["reason"])
                for item in result["kernels"]:
                    covered.setdefault(item["name"], dict(name=item["name"], sass_file=item["sass_file"],
                                       ncu_report=str(folder / "profile.ncu-rep"), source_configuration=context,
                                       source="new", cudnn_version=plan["cudnn_version"]))
            except (OSError, ValueError, KeyError) as error:
                result.update(status="error", reason=f"NCU 与 scan 对照失败：{error}")
        rows.append(result)
        persist()
        logging.info("补采 %s: 覆盖 %d/%d 个目标符号", result["status"], len(targets & covered.keys()), len(targets))
        if result.get("stop"):
            break
    failed = bool(targets - covered.keys()) or any(row["status"] != "pass" or not row["trace_match"] or not row["plan_match"] for row in rows)
    logging.info("%s 目标=%d, 复用=%d, 新增=%d, 缺失=%d", "[失败]" if failed else "[SUCCESS]",
                 len(targets), len(initially_reused), len((targets & covered.keys()) - initially_reused), len(targets - covered.keys()))
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/benchmarks/operators/dwconv/dwconv_compare_bench")
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--device-query", type=Path, help="默认使用同一 build 目录下的 tools/device_query")
    parser.add_argument("--cases", type=int, nargs="+", default=[7])
    parser.add_argument("--dtypes", choices=["fp32", "fp16"], nargs="+", default=["fp32", "fp16"])
    parser.add_argument("--layouts", choices=["nchw", "nhwc"], nargs="+", default=["nchw", "nhwc"])
    parser.add_argument("--kernels", choices=KERNELS, nargs="+", default=KERNELS)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / datetime.now().strftime("dwconv-cudnn-ncu-%Y%m%d-%H%M%S"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--cudnn-candidate", type=int, help="仅用于重放同配置的候选索引，跳过再次 autotune")
    parser.add_argument("--scan-results", type=Path, help="仅补采完整轻量扫描中尚无 SASS 的 kernel 符号")
    parser.add_argument("--reuse-reports", type=Path, nargs="+", default=[], help="显式复用同 GPU/SM/cuDNN 版本的已有 NCU 报告")
    parser.add_argument("--timeout", type=float, default=1800, help="每项抓取/导出超时秒数；0 表示不限时")
    args = parser.parse_args()
    if args.warmup < 0 or any(case < 0 for case in args.cases):
        parser.error("warmup 与 case 不得为负")
    if not math.isfinite(args.timeout) or args.timeout < 0:
        parser.error("timeout 必须是非负有限数")
    if args.scan_results is None and args.reuse_reports:
        parser.error("--reuse-reports 需要 --scan-results")
    if args.scan_results is not None and args.cudnn_candidate is not None:
        parser.error("--scan-results 自动使用各组候选，不能同时指定 --cudnn-candidate")
    if args.cudnn_candidate is not None and (args.cudnn_candidate < 0 or
            any(len(getattr(args, key)) != 1 for key in ("cases", "dtypes", "layouts", "kernels"))):
        parser.error("--cudnn-candidate 要求非负索引和唯一的 case/dtype/layout/kernel 组合")
    for key in ("cases", "dtypes", "layouts", "kernels"):
        values = getattr(args, key)
        if len(values) != len(set(values)):
            parser.error(f"--{key} 包含重复值")
    args.binary, args.output = args.binary.resolve(), args.output.resolve()
    if args.scan_results is not None:
        args.scan_results = args.scan_results.resolve()
    if not args.binary.is_file():
        parser.error(f"基准程序不存在：{args.binary}")
    executable = shutil.which(args.ncu)
    if executable is None:
        parser.error(f"找不到 ncu: {args.ncu}")
    args.ncu = str(Path(executable).resolve())
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录非空；请使用新目录，避免混用历史报告")
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "profile.log", encoding="utf-8")])
    write_json(args.output / "config.json", dict(schema_version=1, created=datetime.now().astimezone().isoformat(),
               binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(), arguments=vars(args), argv=sys.argv))
    environment = args.output / "environment"
    environment.mkdir()
    device_query = args.device_query or args.binary.parents[3] / "tools/device_query"
    checks = {"ncu-version.txt": [args.ncu, "--version"], "nvidia-smi.txt": ["nvidia-smi"],
              "git-head.txt": ["git", "rev-parse", "HEAD"], "git-status.txt": ["git", "status", "--short", "--branch"],
              "device.txt": [device_query]}
    write_json(environment / "commands.json", checks)
    for filename, command in checks.items():
        try:
            if invoke(command, environment / filename, 60) != 0:
                raise RuntimeError(f"必要环境检查失败：{filename}")
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            write_json(args.output / "failure.json", {"stage": "environment", "reason": str(error)})
            logging.error("%s; 未开始 ncu 抓取", error)
            return 1
    if args.scan_results is not None:
        try:
            return capture_scan_results(args)
        except (OSError, ValueError, KeyError, RuntimeError, csv.Error, sqlite3.Error) as error:
            write_json(args.output / "failure.json", {"stage": "scan-validation", "reason": str(error)})
            logging.error("%s; 停止补采", error)
            return 1
    rows = []
    for case, dtype, layout, kernel in itertools.product(args.cases, args.dtypes, args.layouts, args.kernels):
        row = profile_one(args, dict(case=case, dtype=dtype, layout=layout, kernel=kernel))
        rows.append(row)
        save_summary(args.output, rows)
        logging.info("%s %s: kernel 数=%s %s", row["status"], row["directory"], row["captured_kernels"], row["reason"])
        if row.get("stop"):
            break
    failed = any(row["status"] == "error" for row in rows)
    logging.info("%s 完成=%d 跳过=%d 失败=%d，输出 %s", "[失败]" if failed else "[SUCCESS]",
                 sum(row["status"] == "pass" for row in rows), sum(row["status"] == "skip" for row in rows),
                 sum(row["status"] == "error" for row in rows), args.output)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
