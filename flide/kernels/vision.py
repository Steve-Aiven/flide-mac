"""Vision tower helpers: Triton on CUDA, eager PyTorch on MPS/CPU."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .backend import use_triton


def layer_norm(x, w, b, eps: float, residual=None):
    if use_triton(x.device):
        from . import vision_triton as t

        return t.layer_norm(x, w, b, eps, residual=residual)
    if residual is not None:
        h = (x.float() + residual.float()).to(x.dtype)
        xf = h.float()
    else:
        h = x
        xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = ((xf - mean) ** 2).mean(-1, keepdim=True)
    y = ((xf - mean) * torch.rsqrt(var + eps) * w.float() + b.float()).to(x.dtype)
    return (h, y) if residual is not None else y


def rope(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """qkv [n, 3, H, D], cos/sin [n, D] -> rotated q, k [n, H, D]."""
    if use_triton(qkv.device):
        from . import vision_triton as t

        return t.rope(qkv, cos, sin)
    n, _, H, D = qkv.shape
    half = D // 2
    c1, c2 = cos[:, :half].float(), cos[:, half:].float()
    s1, s2 = sin[:, :half].float(), sin[:, half:].float()

    def rot(x):
        x1, x2 = x[..., :half].float(), x[..., half:].float()
        y1 = x1 * c1[:, None, :] - x2 * s1[:, None, :]
        y2 = x2 * c2[:, None, :] + x1 * s2[:, None, :]
        return torch.cat([y1, y2], -1).to(x.dtype)

    return rot(qkv[:, 0]), rot(qkv[:, 1])


def resize_u8(img: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if use_triton(img.device):
        from . import vision_triton as t

        return t.resize_u8(img, height, width)
    x = img
    if x.dim() == 3 and x.shape[-1] in (1, 3, 4):
        x = x.permute(2, 0, 1).unsqueeze(0).float()
    else:
        x = x.unsqueeze(0).float() if x.dim() == 3 else x.float()
    y = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return y.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
