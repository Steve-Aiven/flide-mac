"""Kernels. On CUDA, route Triton's run-time C builds through prebuilt modules before compile."""
from .backend import triton_available

if triton_available():
    from .launcher import install as _install

    _install()
