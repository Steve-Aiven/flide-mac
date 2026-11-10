"""Row-wise kernels: Triton on CUDA, eager PyTorch on MPS/CPU."""
from __future__ import annotations

from .backend import use_triton


def rms_norm(x, w, eps: float, residual=None, offset: bool = True):
    if not use_triton(x.device):
        from . import eager

        return eager.rms_norm(x, w, eps, residual=residual, offset=offset)
    from . import pointwise_triton as t

    return t.rms_norm(x, w, eps, residual=residual, offset=offset)


def swiglu(gu, n: int):
    if not use_triton(gu.device):
        from . import eager

        return eager.swiglu(gu, n)
    from . import pointwise_triton as t

    return t.swiglu(gu, n)


def gated_rms(o, z, w, eps: float):
    if not use_triton(o.device):
        from . import eager

        return eager.gated_rms(o, z, w, eps)
    from . import pointwise_triton as t

    return t.gated_rms(o, z, w, eps)


def qk_norm_rope(q, k, cos, sin, wq, wk, eps: float):
    if not use_triton(q.device):
        from . import eager

        return eager.qk_norm_rope(q, k, cos, sin, wq, wk, eps)
    from . import pointwise_triton as t

    return t.qk_norm_rope(q, k, cos, sin, wq, wk, eps)


def norm_rope(x, cos, sin, w, eps: float):
    if not use_triton(x.device):
        from . import eager

        return eager.norm_rope(x, cos, sin, w, eps)
    from . import pointwise_triton as t

    return t.norm_rope(x, cos, sin, w, eps)


def gate(o, g):
    if not use_triton(o.device):
        from . import eager

        return eager.gate(o, g)
    from . import pointwise_triton as t

    return t.gate(o, g)


def gemma_norm_rope(x, cos, sin, w, eps: float):
    if not use_triton(x.device):
        from . import eager

        return eager.gemma_norm_rope(x, cos, sin, w, eps)
    from . import pointwise_triton as t

    return t.gemma_norm_rope(x, cos, sin, w, eps)


def sandwich(d, h, w1, eps: float, scale=None, w2=None):
    if not use_triton(d.device):
        from . import eager

        return eager.sandwich(d, h, w1, eps, scale=scale, w2=w2)
    from . import pointwise_triton as t

    return t.sandwich(d, h, w1, eps, scale=scale, w2=w2)


def geglu(a, b):
    if not use_triton(a.device):
        from . import eager

        return eager.geglu(a, b)
    from . import pointwise_triton as t

    return t.geglu(a, b)
