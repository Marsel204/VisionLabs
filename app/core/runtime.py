"""Runtime capability detection for CPU/GPU deployments."""

from __future__ import annotations

import functools
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GpuStatus:
    """Detected accelerator state suitable for display in the UI."""

    available: bool
    device: str
    name: str
    memory_gb: float | None = None


@functools.cache
def detect_gpu() -> GpuStatus:
    """Detect CUDA, MPS, or CPU without importing torch at application import time."""
    try:
        import torch
    except ImportError:
        return GpuStatus(False, "cpu", "PyTorch unavailable")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        capability = torch.cuda.get_device_capability(0)
        supported_architectures = set(torch.cuda.get_arch_list())
        architecture = f"sm_{capability[0]}{capability[1]}"
        if supported_architectures and architecture not in supported_architectures:
            return GpuStatus(
                False,
                "cpu",
                f"{properties.name} (unsupported by installed PyTorch; using CPU)",
                properties.total_memory / 1024**3,
            )
        return GpuStatus(True, "cuda", properties.name, properties.total_memory / 1024**3)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return GpuStatus(True, "mps", "Apple Metal", None)
    return GpuStatus(False, "cpu", "CPU")
