"""连续 FP32 CUDA 张量的逐元素加法，不执行广播或隐式复制。"""

import torch
from torch import Tensor


def vector_add(a: Tensor, b: Tensor) -> Tensor:
    """返回与输入同 shape 的逐元素加法结果。"""
    return torch.ops.cuda_lab.vector_add.default(a, b)


def vector_add_out(a: Tensor, b: Tensor, out: Tensor) -> Tensor:
    """将结果写入独立预分配的 out 并返回 out。"""
    torch.ops.cuda_lab.vector_add_out.default(a, b, out)
    return out
