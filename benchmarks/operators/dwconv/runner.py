"""DWConv 基准程序的位置、结构化结果校验与原始日志归档。"""

import json
import math
import shlex
import subprocess
from pathlib import Path


RESULT_PREFIX = "DWCONV_RESULT "
PROCESS_PREFIX = "DWCONV_PROCESS "


def benchmark_binary(build):
    return Path(build) / "benchmarks/operators/dwconv/dwconv_compare_bench"


def canonical_kernel(kernel):
    return "cudnn" if kernel in {"cudnn-autotune", "cudnn-graph"} else kernel


def invoke(command, timeout):
    """超时也保留已有 stdout/stderr, 并以非零退出状态交给结果校验。"""
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
        return subprocess.CompletedProcess(
            command, 124, decode(error.stdout), decode(error.stderr) + f"\n[TIMEOUT] 单项超过 {timeout} 秒\n")


def process_log(command, result):
    return ("命令：" + shlex.join(map(str, command)) + "\n" + result.stdout + result.stderr + "\n"
            + PROCESS_PREFIX + json.dumps({"returncode": result.returncode}) + "\n")


def cached_returncode(log):
    records = [json.loads(line[len(PROCESS_PREFIX):]) for line in log.splitlines()
               if line.startswith(PROCESS_PREFIX)]
    if len(records) != 1 or not isinstance(records[0], dict) or type(records[0].get("returncode")) is not int:
        raise ValueError("原始日志缺少唯一的子进程退出状态")
    return records[0]["returncode"]


def parse_result(log, returncode, *, case, kernel, dtype, layout,
                 warmup=None, iterations=None, samples=None, checked=None):
    """只接受与请求完全对应的结果；执行失败时不得使用已打印的成功耗时。"""
    records = [json.loads(line[len(RESULT_PREFIX):]) for line in log.splitlines()
               if line.startswith(RESULT_PREFIX)]
    expected = (int(case), canonical_kernel(kernel), dtype, layout)

    if len(records) != 1:
        raise ValueError(f"预期恰好一条 DWCONV_RESULT, 实际 {len(records)} 条")
    record = records[0]

    if not isinstance(record, dict):
        raise ValueError("DWCONV_RESULT 必须为 JSON object")

    actual = tuple(record.get(key) for key in ("case", "kernel", "dtype", "layout"))

    if type(record.get("case")) is not int or actual != expected:
        raise ValueError(f"结果配置不匹配：期望 {expected}，实际 {actual}")
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1 or record.get("timing_mode") != "event":
        raise ValueError("不支持的结果 schema 或计时方式")
    if type(record.get("checked")) is not bool:
        raise ValueError("结果缺少 checked 校验标记")

    for key in ("warmup", "iterations", "samples"):
        if type(record.get(key)) is not int or record[key] < (0 if key == "warmup" else 1):
            raise ValueError(f"非法结果字段：{key}")
    for key, expected_value in (("warmup", warmup), ("iterations", iterations), ("samples", samples)):
        if expected_value is not None and record[key] != expected_value:
            raise ValueError(f"结果采样配置不匹配：{key}")

    status = record.get("status")
    if status not in {"pass", "skip", "error"}:
        raise ValueError(f"未知结果状态：{status}")
    if status != "pass":
        if not isinstance(record.get("reason"), str) or not record["reason"]:
            raise ValueError("跳过或失败结果缺少 reason")
        if any(key in record for key in ("us", "min_us", "max_us", "samples_us")):
            raise ValueError("跳过或失败结果不得包含成功耗时")
        return record

    if returncode != 0:
        raise ValueError(f"子进程退出码为 {returncode}, 不接受成功耗时")
    if checked is not None and record["checked"] != checked:
        raise ValueError("结果的 checked 与请求的 CPU 校验配置不一致")
    for key in ("us", "min_us", "max_us", "prepare_ms", "post_ms", "host_layout_ms"):
        value = record.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"非法或缺失的耗时字段：{key}")

    samples = record.get("samples_us")
    if (not isinstance(samples, list) or len(samples) != record["samples"]
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in samples)):
        raise ValueError("samples_us 与采样数不一致，或包含非法耗时")
    if record["us"] <= 0 or not record["min_us"] <= record["us"] <= record["max_us"]:
        raise ValueError("成功耗时必须为正，且处于最小/最大耗时之间")
    for key in ("algo", "engine"):
        if type(record.get(key)) is not int:
            raise ValueError(f"非法或缺失的后端字段：{key}")
    if "workspace_bytes" not in record:
        raise ValueError("缺失的后端字段: workspace_bytes")

    workspace = record["workspace_bytes"]
    if workspace is not None and (type(workspace) is not int or workspace < 0):
        raise ValueError("workspace_bytes 必须为非负整数，或 null 表示未报告")
    return record
