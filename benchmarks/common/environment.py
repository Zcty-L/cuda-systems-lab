"""采集实际运行环境和工作树状态。"""

from pathlib import Path
import os
import platform
import subprocess
import sys

import torch


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_environment(root: Path) -> dict:
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    status = _git(root, "status", "--porcelain")
    return {
        "worktree": str(root),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_branch": _git(root, "branch", "--show-current"),
        "git_dirty": None if status is None else bool(status),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_index": device,
        "device_name": properties.name,
        "sm": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
    }
