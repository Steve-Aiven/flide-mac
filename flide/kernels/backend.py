"""Which kernel path to use: Triton (CUDA), Metal (MPS), or eager PyTorch."""
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


def _kind(device) -> str | None:
    if device is None:
        return None
    import torch

    return device.type if isinstance(device, torch.device) else str(device).split(":", 1)[0]


def use_triton(device=None) -> bool:
    """Triton kernels only on CUDA devices when Triton imports."""
    if not triton_available():
        return False
    kind = _kind(device)
    return kind is None or kind == "cuda"


def use_metal(device=None) -> bool:
    """Metal shaders on MPS when torch.mps.compile_shader is available."""
    kind = _kind(device)
    if kind is not None and kind != "mps":
        return False
    try:
        from .metal.runtime import metal_available

        return metal_available()
    except Exception:
        return False
