"""汇总 DWConv 独立进程重复测量，生成逐项、横向对比和规则网格统计。"""

import csv
import math
import re
import statistics
from pathlib import Path


SHAPE_FIELDS = ["case", "n", "c", "h", "w", "kh", "kw", "sh", "sw", "ph", "pw", "bias"]
REFERENCE_FIELDS = ["reference_kernel", "reference_layout", "relative_to_reference", "relative_to_cudnn",
                    "speedup_vs_nchw", "speedup_vs_separate"]
SUMMARY_FIELDS = SHAPE_FIELDS + [
    "dtype", "layout", "kernel", "status", "us", *REFERENCE_FIELDS,
    "repeats", "completed_repeats", "errors", "unsupported", "skipped", "samples", "warmup", "iterations",
    "min_us", "max_us", "prepare_ms", "post_ms", "host_layout_ms", "algo", "engine", "workspace_bytes",
    "timing_mode", "checked", "reason",
]


def _time(row):
    value = row.get("us") if row and row.get("status") == "ok" else None
    return value if type(value) in (int, float) and math.isfinite(value) and value > 0 else None


def _key(row, kernel=None, layout=None):
    # 原始样本只与同一次独立进程轮次的参考项比较；汇总项没有 repeat 字段。
    repeat = int(row["repeat"]) if "repeat" in row else None
    return int(row["case"]), row["dtype"], kernel or row["kernel"], layout or row["layout"], repeat


def annotate_references(rows, reference, reference_layout=None):
    """原地补充相对耗时；失败项、失败参考和重复参考均不生成比值。"""
    index = {}
    for row in rows:
        key = _key(row)
        index[key] = None if key in index else row
    for row in rows:
        layout = reference_layout or row["layout"]
        row.update(reference_kernel=reference, reference_layout=layout)
        for field in REFERENCE_FIELDS[2:]:
            row[field] = ""
        current = _time(row)
        if current is None:
            continue
        denominator = _time(index.get(_key(row, reference, layout)))
        if denominator is not None:
            row["relative_to_reference"] = row["relative_to_cudnn"] = current / denominator
        nchw = _time(index.get(_key(row, layout="nchw")))
        separate = _time(index.get(_key(row, kernel="cudnn-separate")))
        if nchw is not None:
            row["speedup_vs_nchw"] = nchw / current
        if separate is not None:
            row["speedup_vs_separate"] = separate / current
    return rows


def _common_value(rows, field):
    first = rows[0].get(field)
    return first if all(row.get(field) == first for row in rows) else None


def _summarize(group, repeats):
    first = group[0]
    successful = [row for row in group if row["status"] == "ok"]
    result = {key: first[key] for key in SHAPE_FIELDS + ["dtype", "layout", "kernel"]}
    result.update(repeats=repeats, completed_repeats=len(successful),
                  errors=sum(row["status"] == "error" for row in group),
                  unsupported=sum(row["status"] == "unsupported" for row in group),
                  skipped=sum(row["status"] not in {"ok", "error"} for row in group), checked=False)
    consistent = True
    for field in ("samples", "warmup", "iterations", "timing_mode"):
        present = [row for row in group if row.get(field) is not None]
        result[field] = _common_value(present, field) if present else None
        consistent = consistent and (not present or result[field] is not None)
    reasons = list(dict.fromkeys(row["reason"] for row in group if row.get("reason")))
    ids = [int(row["repeat"]) for row in group]
    complete = len(ids) == repeats and set(ids) == set(range(repeats))
    if not complete or not consistent:
        result["status"] = "error"
        result["errors"] = max(1, result["errors"])
        if not complete:
            reasons.append("独立进程轮次缺失、重复或超出预期范围")
        if not consistent:
            reasons.append("独立进程轮次的计时配置不一致")
    elif len(successful) == repeats:
        result.update(status="ok", us=statistics.median(row["us"] for row in group),
                      min_us=min(row["min_us"] for row in group), max_us=max(row["max_us"] for row in group),
                      checked=all(row.get("checked") is True for row in group))
        for field in ("prepare_ms", "post_ms", "host_layout_ms"):
            result[field] = statistics.median(row[field] for row in group)
        for field in ("algo", "engine"):
            result[field] = _common_value(group, field)
        workspace = [row.get("workspace_bytes") for row in group]
        result["workspace_bytes"] = None if any(value is None for value in workspace) else max(workspace)
    else:
        result["status"] = next(status for status in ("error", "oom_skipped", "memory_skipped", "unsupported")
                                if any(row["status"] == status for row in group))
    result["reason"] = "；".join(reasons)
    return result


def _write_csv(path, fields, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _comparison(summaries, kernels, total_bytes):
    groups = {}
    for row in summaries:
        groups.setdefault((int(row["case"]), row["dtype"], row["layout"]), {})[row["kernel"]] = row
    comparison = []
    for group in groups.values():
        first = next(iter(group.values()))
        row = {key: first[key] for key in SHAPE_FIELDS + ["dtype", "layout", "reference_kernel", "reference_layout"]}
        n, c, h, w, kh, kw, sh, sw, ph, pw = (int(row[key]) for key in SHAPE_FIELDS[1:-1])
        oh, ow = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
        size = (2 if row["dtype"] == "fp16" else 4) * n * c * (h * w + oh * ow)
        row.update(input_output_bytes=size, exceeds_physical_vram=int(size > total_bytes))
        for kernel in kernels:
            item = group.get(kernel)
            row.update({f"{kernel}_status": item["status"] if item else "missing",
                        f"{kernel}_us": item.get("us", "") if item else "",
                        f"{kernel}_relative": item.get("relative_to_reference", "") if item else ""})
        comparison.append(row)
    return comparison


def _aggregates(comparison, kernels):
    groups = {}
    for row in comparison:
        if int(row["case"]) < 23 or row["exceeds_physical_vram"]:
            continue
        if not all(row[f"{kernel}_status"] == "ok" and row[f"{kernel}_relative"] != "" for kernel in kernels):
            continue
        groups.setdefault((row["dtype"], row["layout"], int(row["kh"])), []).append(row)
    aggregates = []
    for (dtype, layout, size), group in sorted(groups.items()):
        aggregates.append(dict(dtype=dtype, layout=layout, k=size, shapes=len(group), **{
            kernel: statistics.median(row[f"{kernel}_relative"] for row in group) for kernel in kernels}))
    return aggregates


def write_reports(rows, output, *, repeats, reference, reference_layout=None):
    """写汇总 CSV 并返回汇总行；各轮全部成功后才发布聚合耗时。"""
    if repeats < 1:
        raise ValueError("repeats 必须为正数")
    output = Path(output)
    device = (output / "device.txt").read_text()
    match = re.search(r"Global memory.*\((\d+) B\)", device)
    if match is None:
        raise ValueError("设备记录缺少 Global memory 字节数")
    total_bytes = int(match[1])
    groups = {}
    for row in rows:
        groups.setdefault((int(row["case"]), row["dtype"], row["kernel"], row["layout"]), []).append(row)
    summaries = [_summarize(group, repeats) for group in groups.values()]
    annotate_references(summaries, reference, reference_layout)
    _write_csv(output / "summary.csv", SUMMARY_FIELDS, summaries)
    kernels = list(dict.fromkeys(row["kernel"] for row in rows))
    comparison = _comparison(summaries, kernels, total_bytes)
    fields = SHAPE_FIELDS + ["dtype", "layout", "reference_kernel", "reference_layout",
                            "input_output_bytes", "exceeds_physical_vram"]
    for kernel in kernels:
        fields += [f"{kernel}_status", f"{kernel}_us", f"{kernel}_relative"]
    _write_csv(output / "comparison.csv", fields, comparison)
    _write_csv(output / "aggregates.csv", ["dtype", "layout", "k", "shapes"] + kernels,
               _aggregates(comparison, kernels))
    return summaries
