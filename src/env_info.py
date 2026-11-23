"""Hardware and code-version info written into every results row.

OWNER: C (Minh). Section 4.2 of the assignment requires recording hardware,
training time and number of parameters for each run.
"""
from __future__ import annotations

import functools
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=1)
def cpu_name() -> str:
    name = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    name = name or platform.processor() or platform.machine() or "unknown"
    return f"CPU: {name} ({os.cpu_count()} cores)"


@functools.lru_cache(maxsize=1)
def gpu_name() -> str | None:
    try:
        import torch  # noqa: WPS433 (optional dependency)
    except ImportError:
        return None
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return None


def hardware_for(model) -> str:
    """GPU name if the model says it trained on CUDA, otherwise the CPU name."""
    device = str(getattr(model, "device", "cpu"))
    if device.startswith("cuda"):
        return gpu_name() or "GPU: unknown"
    return cpu_name()


@functools.lru_cache(maxsize=1)
def git_commit() -> str:
    """Short commit hash, with "-dirty" if code or configs have uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs", "scripts"],
                                cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if status else sha
