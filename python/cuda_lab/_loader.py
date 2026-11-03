"""加载当前安装包中的算子库，不在导入时查询或初始化 CUDA 设备。"""

import json
from importlib.util import find_spec
from pathlib import Path

import torch

_PACKAGE_DIR = Path(__file__).resolve().parent
_spec = find_spec("cuda_lab._C")
if _spec is None or _spec.origin is None:
    raise ImportError("未找到 cuda_lab 的 _C 算子库；请按构建指南安装当前 worktree 的包")

# 只查找路径，不 import 扩展：此库通过 torch.ops 加载，不提供 PyInit__C。
# editable 安装时源码与二进制可位于不同目录。
_LIBRARY = Path(_spec.origin).resolve()
torch.ops.load_library(str(_LIBRARY))


def build_info() -> dict:
    """返回构建环境、提交状态和实际加载路径，供实验记录使用。"""
    with _LIBRARY.with_name("_build_info.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {**metadata, "package_path": str(_PACKAGE_DIR), "library_path": str(_LIBRARY)}
