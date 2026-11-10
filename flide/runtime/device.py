"""Device and dtype selection for CUDA and Apple Silicon (MPS)."""
from __future__ import annotations

import platform
import sys


def is_darwin() -> bool:
    return sys.platform == "darwin"


def mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def resolve_device(requested: str | None = None) -> str:
    """Pick a device string. `auto` / None: CUDA if present, else MPS on Apple Silicon, else error."""
    import torch

    req = (requested or "auto").strip().lower()
    if req in ("", "auto"):
        if torch.cuda.is_available():
            return "cuda:0"
        if mps_available():
            return "mps"
        raise RuntimeError(
            "No usable GPU: install a CUDA build of PyTorch with an NVIDIA driver, "
            "or run on Apple Silicon with the macOS MPS build of PyTorch."
        )
    if req == "mps":
        if not mps_available():
            raise RuntimeError("MPS is not available (need Apple Silicon and a macOS PyTorch build).")
        return "mps"
    if req == "cpu":
        return "cpu"
    if req.startswith("cuda"):
        return requested if ":" in requested else f"{requested}:0"
    return requested


def resolve_dtype(device: str, requested: str | None = None) -> str:
    """Default dtype: bfloat16 on CUDA, float16 on MPS/CPU (bf16 is uneven on Metal)."""
    if requested and requested.strip().lower() not in ("", "auto"):
        return requested
    kind = device.split(":", 1)[0]
    return "bfloat16" if kind == "cuda" else "float16"


def device_kind(device) -> str:
    import torch

    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


def is_cuda(device) -> bool:
    return device_kind(device) == "cuda"


def synchronize(device=None) -> None:
    import torch

    kind = device_kind(device) if device is not None else ("cuda" if torch.cuda.is_available()
                                                          else "mps" if mps_available() else "cpu")
    if kind == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif kind == "mps" and mps_available():
        torch.mps.synchronize()


def empty_cache(device=None) -> None:
    import torch

    kind = device_kind(device) if device is not None else ("cuda" if torch.cuda.is_available()
                                                          else "mps" if mps_available() else "cpu")
    if kind == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif kind == "mps" and mps_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def set_device(device) -> None:
    import torch

    if is_cuda(device):
        torch.cuda.set_device(device)


def mem_gib(device=None) -> tuple[float, float, float]:
    """(allocated, reserved, used) in GiB. MPS reports process allocations only; used matches allocated."""
    import torch

    g = 2 ** 30
    kind = device_kind(device) if device is not None else ("cuda" if torch.cuda.is_available()
                                                          else "mps" if mps_available() else "cpu")
    if kind == "cuda" and torch.cuda.is_available():
        synchronize("cuda")
        empty_cache("cuda")
        free, total = torch.cuda.mem_get_info()
        return torch.cuda.memory_allocated() / g, torch.cuda.memory_reserved() / g, (total - free) / g
    if kind == "mps" and mps_available():
        synchronize("mps")
        alloc = torch.mps.current_allocated_memory() / g if hasattr(torch.mps, "current_allocated_memory") else 0.0
        driver = torch.mps.driver_allocated_memory() / g if hasattr(torch.mps, "driver_allocated_memory") else alloc
        return alloc, driver, driver
    return 0.0, 0.0, 0.0


def device_label(device=None) -> str:
    import torch

    kind = device_kind(device) if device is not None else None
    if kind == "cuda" or (kind is None and torch.cuda.is_available()):
        idx = int(str(device).split(":")[-1]) if device is not None and ":" in str(device) else 0
        return torch.cuda.get_device_name(idx)
    if kind == "mps" or (kind is None and mps_available()):
        return f"Apple MPS ({platform.machine()})"
    return "cpu"
