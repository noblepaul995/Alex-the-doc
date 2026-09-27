"""
Best-effort hardware detection for `alex setup`'s local-model
recommendation step — how much RAM and GPU VRAM this machine has,
so `config.model_recommendations` can suggest models that will actually
run at a usable speed rather than thrash swap or fail to load.

Deliberately zero new dependencies (no `psutil`): every provider stage
here is either stdlib or a subprocess call to a tool that's already
expected to exist for the thing it's detecting (`nvidia-smi` only
matters if there's an NVIDIA GPU to ask about in the first place).
Every detection path degrades to `None` on failure rather than raising —
a laptop with no GPU, a container without `/proc`, a locked-down `wmic`
call are all real, unremarkable situations this needs to handle
silently, not treat as errors.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class HardwareInfo:
    total_ram_gb: float | None
    gpu_name: str | None
    gpu_vram_gb: float | None


def detect_hardware() -> HardwareInfo:
    """Best-effort snapshot of this machine's RAM and (NVIDIA) GPU VRAM. Any field may be `None` if it couldn't be determined."""
    return HardwareInfo(
        total_ram_gb=_detect_ram_gb(),
        gpu_name=_detect_nvidia_gpu_name(),
        gpu_vram_gb=_detect_nvidia_vram_gb(),
    )


def _detect_ram_gb() -> float | None:
    system = platform.system()
    try:
        if system == "Linux":
            return _ram_from_proc_meminfo()
        if system == "Darwin":
            return _ram_from_sysctl()
        if system == "Windows":
            return _ram_from_wmic()
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return None


def _ram_from_proc_meminfo() -> float | None:
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                kib = int(line.split()[1])
                return round(kib / (1024 * 1024), 1)
    return None


def _ram_from_sysctl() -> float | None:
    result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
    if result.returncode != 0:
        return None
    return round(int(result.stdout.strip()) / (1024**3), 1)


def _ram_from_wmic() -> float | None:
    # `wmic` is deprecated but still present on most Windows installs this
    # would run on; PowerShell's Get-CimInstance would be the modern
    # replacement but adds a slower startup — falling back to `None` (and
    # the manual-entry prompt in the setup flow) is an acceptable outcome
    # either way, so this stays simple rather than trying both.
    result = subprocess.run(
        ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        return None
    match = re.search(r"\d+", result.stdout)
    if not match:
        return None
    return round(int(match.group()) / (1024**3), 1)


def _run_nvidia_smi(query: str) -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    # Multiple GPUs would each get their own line — take the first, since
    # the recommendation logic only needs a rough capability tier, not an
    # exact multi-GPU inventory.
    return result.stdout.strip().splitlines()[0].strip()


def _detect_nvidia_gpu_name() -> str | None:
    return _run_nvidia_smi("name")


def _detect_nvidia_vram_gb() -> float | None:
    raw = _run_nvidia_smi("memory.total")
    if raw is None:
        return None
    # nvidia-smi reports this as e.g. "24576 MiB".
    match = re.search(r"[\d.]+", raw)
    if not match:
        return None
    return round(float(match.group()) / 1024, 1)