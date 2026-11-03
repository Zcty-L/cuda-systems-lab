"""后端只负责准备输出与待测调用，输入生成和计时由统一入口负责。"""

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class PreparedCall:
    call: Callable
    output: torch.Tensor
    metadata: dict


def prepare(name: str, a: torch.Tensor, b: torch.Tensor) -> PreparedCall:
    out = torch.empty_like(a)
    if name == "torch":
        return PreparedCall(lambda: torch.add(a, b, out=out), out, {"implementation": "torch.add(out=)"})
    if name == "cuda_lab":
        import cuda_lab

        return PreparedCall(
            lambda: cuda_lab.vector_add_out(a, b, out),
            out,
            {"implementation": "cuda_lab.vector_add_out", "build_info": cuda_lab.build_info()},
        )
    raise ValueError(f"未知后端：{name}")
