import os
import platform
import subprocess
import sys

import torch


def _run(*cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def cpu_model() -> str:
    if sys.platform == "darwin":
        return _run("sysctl", "-n", "machdep.cpu.brand_string")
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def ram_gb() -> float:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError):
        return 0.0


def hardware() -> dict:
    return {
        "cpu": cpu_model(),
        "cores": os.cpu_count(),
        "ram_gb": ram_gb(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "go": _run("go", "version"),
    }


def git_commit() -> str:
    sha = _run("git", "rev-parse", "--short=12", "HEAD")
    dirty = _run("git", "status", "--porcelain", "--untracked-files=no")
    return sha + ("-dirty" if dirty else "")
