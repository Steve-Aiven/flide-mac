"""Metal pointwise kernels for the Qwen3.5 / OneJev MPS hot path."""
from __future__ import annotations

import torch

from .runtime import metal_available, shader_lib


def _ok(t: torch.Tensor) -> bool:
    return bool(metal_available() and t.is_mps and t.dtype == torch.float16)


def rms_norm(x, w, eps: float, residual=None, offset: bool = True):
    if not _ok(x):
        from .. import eager

        return eager.rms_norm(x, w, eps, residual=residual, offset=offset)
    T, N = x.shape
    y = torch.empty_like(x)
    has_res = residual is not None
    if T == 0:
        return (torch.empty_like(x), y) if has_res else y
    h = torch.empty_like(x) if has_res else y  # unused when no residual
    res = residual.contiguous() if has_res else x
    kn = shader_lib().rms_norm_f16
    # One threadgroup per row; parallelize the N-reduction (tg must be a power of two).
    tg = 256 if N >= 256 else 128 if N >= 128 else 64 if N >= 64 else 32
    tg = min(tg, 256)
    kn(x.contiguous(), res, w.contiguous().half(), h, y,
       T, N, float(eps), int(has_res), int(offset),
       threads=[T * tg, 1, 1], group_size=[tg, 1, 1])
    return (h, y) if has_res else y


def swiglu(gu, n: int):
    if not _ok(gu):
        from .. import eager

        return eager.swiglu(gu, n)
    T = gu.shape[0]
    out = torch.empty(T, n, device=gu.device, dtype=gu.dtype)
    if T == 0:
        return out
    kn = shader_lib().swiglu_f16
    tg = min(256, max(n, 1))
    kn(gu.contiguous(), out, T, n,
       threads=[n, T, 1], group_size=[tg, 1, 1])
    return out


def gated_rms(o, z, w, eps: float):
    if not _ok(o):
        from .. import eager

        return eager.gated_rms(o, z, w, eps)
    T, H, DV = o.shape
    out = torch.empty(T, H * DV, device=o.device, dtype=o.dtype)
    if T == 0:
        return out
    kn = shader_lib().gated_rms_f16
    tg = min(256, max(H, 1))
    kn(o.contiguous(), z.contiguous(), w.contiguous().half(), out,
       T, H, DV, float(eps),
       threads=[H, T, 1], group_size=[tg, 1, 1])
    return out


def norm_rope(x, cos, sin, w, eps: float):
    if not _ok(x):
        from .. import eager

        return eager.norm_rope(x, cos, sin, w, eps)
    T, Heads, HD = x.shape
    R = int(cos.shape[-1])
    out = torch.empty(T, Heads, HD, device=x.device, dtype=x.dtype)
    if T == 0:
        return out
    kn = shader_lib().norm_rope_f16
    tg = min(256, max(Heads, 1))
    kn(x.contiguous(), cos.contiguous().half(), sin.contiguous().half(), w.contiguous().half(), out,
       T, Heads, HD, R, float(eps),
       threads=[Heads, T, 1], group_size=[tg, 1, 1])
    return out


def qk_norm_rope(q, k, cos, sin, wq, wk, eps: float):
    return norm_rope(q, cos, sin, wq, eps), norm_rope(k, cos, sin, wk, eps)


def gate(o, g):
    if not (_ok(o) and g.dtype == torch.float16):
        from .. import eager

        return eager.gate(o, g)
    T, H, HD = o.shape
    out = torch.empty(T, H * HD, device=o.device, dtype=g.dtype)
    if T == 0:
        return out
    # Contiguous pack: last dim must be HD-contiguous for the simple kernel layout.
    oc = o.contiguous()
    gc = g.contiguous()
    kn = shader_lib().gate_f16
    tg = min(256, max(H, 1))
    kn(oc, gc, out, T, H, HD,
       threads=[H, T, 1], group_size=[tg, 1, 1])
    return out
