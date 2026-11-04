"""预分配输出的 CUDA Event / CUDA Graph 计时与 API 批量计时。"""

from dataclasses import dataclass
from statistics import mean, median, pvariance
from time import perf_counter
from typing import Callable

import torch


@dataclass(frozen=True)
class TimingConfig:
    warmup: int = 20
    iterations: int = 100
    samples: int = 5
    mode: str = "event"

    def __post_init__(self):
        if self.warmup < 0 or self.iterations <= 0 or self.samples <= 0:
            raise ValueError("warmup 必须非负，iterations 和 samples 必须为正数")
        if self.mode not in {"event", "graph"}:
            raise ValueError("mode 必须是 event 或 graph")


def measure_cuda(call: Callable, config: TimingConfig) -> dict:
    """返回每次调用的 GPU 微秒数；event 模式可能包含主机提交间隙。"""
    for _ in range(config.warmup):
        call()
    torch.cuda.synchronize()

    graph = None
    if config.mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(config.iterations):
                call()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(config.samples):
        start.record()
        if graph is not None:
            graph.replay()
        else:
            for _ in range(config.iterations):
                call()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000 / config.iterations)
    return {
        "gpu_us": median(samples),
        "gpu_us_mean": mean(samples),
        "gpu_us_variance": pvariance(samples),
        "gpu_us_min": min(samples),
        "gpu_us_max": max(samples),
        "gpu_us_samples": samples,
    }


def measure_api_batch(call: Callable, config: TimingConfig) -> float:
    """主机计时的批量调用摊销耗时，含批末同步；不是单次请求延迟。"""
    samples = []
    for _ in range(config.samples):
        torch.cuda.synchronize()
        start = perf_counter()
        for _ in range(config.iterations):
            call()
        torch.cuda.synchronize()
        samples.append((perf_counter() - start) * 1e6 / config.iterations)
    return median(samples)
