"""Which kernel path to use: Triton (CUDA) or eager PyTorch (MPS/CPU)."""
from __future__ import annotations

_TRITON = None


def triton_available() -> bool:
    global _TRITON
    if _TRITON is None:
        try:
            import triton  # noqa: F401

            _TRITON = True
        except Exception:
            _TRITON = False
    return _TRITON


def use_triton(device=None) -> bool:
    """Triton kernels only on CUDA devices when Triton imports."""
    if not triton_available():
        return False
    if device is None:
        return True
    import torch

    kind = device.type if isinstance(device, torch.device) else str(device).split(":", 1)[0]
    return kind == "cuda"
