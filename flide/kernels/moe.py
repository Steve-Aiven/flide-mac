"""Mixture-of-experts: Triton on CUDA, eager PyTorch on MPS/CPU."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .backend import use_triton


def moe(x: torch.Tensor, w: dict, mm, swiglu) -> torch.Tensor:
    if use_triton(x.device):
        from . import moe_triton as t

        return t.moe(x, w, mm, swiglu)
    return _moe_eager(x, w, mm, swiglu)


def quantize_experts(w: torch.Tensor):
    from . import moe_triton as t

    return t.quantize_experts(w)


def combine_norm(y: torch.Tensor, dense: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, k: int,
                 eps: float) -> torch.Tensor:
    if use_triton(dense.device):
        from . import moe_triton as t

        return t.combine_norm(y, dense, w1, w2, k, eps)
    T, H = dense.shape
    y = y.view(T, k, H).float().sum(1)
    df = dense.float()
    inv_d = torch.rsqrt(df.pow(2).mean(-1, keepdim=True) + eps)
    inv_y = torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return (df * inv_d * w1.float() + y * inv_y * w2.float()).to(dense.dtype)


def _moe_eager(x: torch.Tensor, w: dict, mm, swiglu) -> torch.Tensor:
    """Packed experts [E, N, K] as in Flide's fused store: gate_up [E, 2I, H], down [E, H, I]."""
    if "experts_scale" in w:
        raise RuntimeError("FP8 MoE is CUDA-only; omit --quantize fp8 on Apple Silicon.")
    K = w["topk"]
    wr, wg = w["router"], w["shared_gate"]
    logits = x @ wr.T if isinstance(wr, torch.Tensor) else mm(x, wr)
    gate = x @ wg.T if isinstance(wg, torch.Tensor) else mm(x, wg)
    probs = torch.softmax(logits.float(), dim=-1)
    top, ids = torch.topk(probs, K, dim=-1)
    top = (top / top.sum(-1, keepdim=True).clamp_min(1e-9)).to(x.dtype)
    T, H = x.shape
    gu_w = w["experts"]       # [E, 2I, H]
    dn_w = w["experts_down"]  # [E, H, I]
    I = w["inter"]
    out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
    for t in range(T):
        xt = x[t]
        acc = torch.zeros(H, device=x.device, dtype=torch.float32)
        for j in range(K):
            e = int(ids[t, j])
            gu = gu_w[e] @ xt  # [2I]
            h = swiglu(gu.unsqueeze(0), I)[0]
            y = dn_w[e] @ h    # [H]
            acc = acc + y.float() * float(top[t, j])
        out[t] = acc
    shared = mm(swiglu(mm(x, w["shared_gate_up"]), w["shared_inter"]), w["shared_down"])
    return (out + shared.float() * torch.sigmoid(gate.float())).to(x.dtype)
