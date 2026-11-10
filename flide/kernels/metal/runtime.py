"""Compile and cache Metal shader libraries."""
from __future__ import annotations

_LIB = None
_FAILED = False


def metal_available() -> bool:
    if _FAILED:
        return False
    try:
        import torch
    except ImportError:
        return False
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                and hasattr(torch.mps, "compile_shader"))


def shader_lib():
    """Lazy-compile the Flide Metal shader library once per process."""
    global _LIB, _FAILED
    if _LIB is not None:
        return _LIB
    if _FAILED or not metal_available():
        raise RuntimeError("Metal shaders are not available on this PyTorch build")
    import torch

    from .shaders import SOURCE

    try:
        _LIB = torch.mps.compile_shader(SOURCE)
    except Exception as exc:
        _FAILED = True
        raise RuntimeError(f"failed to compile Flide Metal shaders: {exc}") from exc
    return _LIB
