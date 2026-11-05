"""python -m benchmarks.operators.vector_add.benchmark --help"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import logging
from pathlib import Path
import subprocess
import sys

import torch

from benchmarks.common.environment import collect_environment
from benchmarks.common.timing import TimingConfig, measure_api_batch, measure_cuda
from .backends import prepare

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list) or not cases:
        raise ValueError("用例文件必须是非空 JSON 数组")
    identifiers = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]:
            raise ValueError("每个用例必须有非空字符串 id")
        if case["id"] in identifiers:
            raise ValueError(f"用例 id 重复：{case['id']}")
        identifiers.add(case["id"])
        shape = case.get("shape")
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError(f"用例 {case['id']} 的 shape 必须是非负整数数组")
        if type(case.get("seed", 42)) is not int:
            raise ValueError(f"用例 {case['id']} 的 seed 必须是整数")
    return cases


def parse_args():
    parser = argparse.ArgumentParser(description="比较 Vector Add 与 torch.add，输出 JSONL、CSV 和环境记录")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--backends", nargs="+", choices=["cuda_lab", "torch"], default=["cuda_lab", "torch"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--mode", choices=["event", "graph"], default="event")
    parser.add_argument("--api-time", action="store_true", help="另测含批末同步的 API 批量摊销耗时")
    parser.add_argument("--device-query", type=Path, default=ROOT / "build/tools/device_query")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/results/vector_add" / stamp)
    return parser.parse_args()


def save_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args) -> int:
    config = TimingConfig(args.warmup, args.iterations, args.samples, args.mode)

    cases = load_cases(args.cases)

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU")

    # 先执行项目规定的设备查询；失败时停止测量，保留原始错误。
    query = subprocess.run([str(args.device_query.resolve())], check=True, capture_output=True, text=True)
    LOGGER.info("%s", query.stdout.rstrip())

    torch.cuda.set_device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "device-query.log").write_text(query.stdout, encoding="utf-8")
    environment = collect_environment(ROOT)

    save_json(args.output_dir / "environment.json", environment)
    save_json(args.output_dir / "config.json", {
        "timing": asdict(config), "api_time": args.api_time,
        "backends": args.backends, "cases": cases,
        "command": [sys.executable, "-m", "benchmarks.operators.vector_add.benchmark", *sys.argv[1:]],
        "scope": "预分配输出，排除分配、传输与输入生成；同一输入和 stream",
    })

    records = []
    failed = False
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as output:
        for case in cases:
            generator = torch.Generator(device="cuda").manual_seed(case.get("seed", 42))
            a = torch.randn(case["shape"], device="cuda", dtype=torch.float32, generator=generator)
            b = torch.randn(case["shape"], device="cuda", dtype=torch.float32, generator=generator)
            reference = (a.double() + b.double()).float()

            LOGGER.info("\n[配置] case=%s shape=%s mode=%s", case["id"], case["shape"], config.mode)

            for backend in dict.fromkeys(args.backends):
                record = {
                    "case_id": case["id"], "op": "vector_add", "backend": backend,
                    "shape": case["shape"], "dtype": "float32", "layout": "contiguous",
                    "seed": case.get("seed", 42), "mode": config.mode,
                    "iterations": config.iterations, "samples": config.samples,
                }
                try:
                    if a.numel() == 0:
                        record.update(status="skip", reason="空输入仅做正确性测试，不测 GPU 耗时")
                    else:
                        prepared = prepare(backend, a, b)
                        record.update(prepared.metadata)
                        LOGGER.info("\n[阶段] backend=%s 正确性验证", backend)

                        prepared.call()

                        torch.testing.assert_close(prepared.output, reference, rtol=0, atol=0)

                        record.update(max_abs_error=(prepared.output - reference).abs().max().item())
                        LOGGER.info("[阶段] backend=%s 预热与计时", backend)

                        measurements = measure_cuda(prepared.call, config)
                        if args.api_time:
                            measurements["api_batch_us"] = measure_api_batch(prepared.call, config)
                        record.update(measurements)
                        record["status"] = "pass"
                except Exception as error:
                    failed = True
                    record.update(status="error", reason=f"{type(error).__name__}: {error}")

                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                LOGGER.info("[结果] backend=%s status=%s gpu_us=%s %s", backend, record["status"],
                            record.get("gpu_us"), record.get("reason", ""))

    baselines = {row["case_id"]: row["gpu_us"] for row in records
                 if row["backend"] == "torch" and row["status"] == "pass"}
    fields = ["case_id", "backend", "shape", "dtype", "mode", "status", "gpu_us", "gpu_us_min",
              "gpu_us_max", "api_batch_us", "speedup_vs_torch", "max_abs_error", "reason"]

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            if row["status"] == "pass" and row["case_id"] in baselines and row["gpu_us"] > 0:
                row["speedup_vs_torch"] = baselines[row["case_id"]] / row["gpu_us"]
            writer.writerow(row)

    LOGGER.info("\n[%s] 结果目录：%s", "ERROR" if failed else "SUCCESS", args.output_dir)

    return 1 if failed else 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        with torch.inference_mode():
            return run(parse_args())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        LOGGER.error("[ERROR] %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
