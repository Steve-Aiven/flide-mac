"""Eager PyTorch kernels for MPS/CPU (and as Triton fallbacks). Arithmetic matches the Triton paths in float32."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def rms_norm(x, w, eps: float, residual=None, offset: bool = True):
    weight = (1.0 + w.float()) if offset else w.float()
    if residual is not None:
        h = (x.float() + residual.float()).to(x.dtype)
        xf = h.float()
        inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        y = (xf * inv * weight).to(x.dtype)
        return h, y
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * inv * weight).to(x.dtype)


def swiglu(gu, n: int):
    a, b = gu[:, :n].float(), gu[:, n:].float()
    return (a * torch.sigmoid(a) * b).to(gu.dtype)


def gated_rms(o, z, w, eps: float):
    """Match pointwise_triton: RMSNorm(o) * silu(z)."""
    T, H, DV = o.shape
    of = o.float()
    inv = torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    y = of * inv * w.float()
    zf = z.float().view(T, H, DV)
    return (y * zf * torch.sigmoid(zf)).to(o.dtype).view(T, H * DV)


def norm_rope(x, cos, sin, w, eps: float):
    """Match kernels/pointwise_triton.norm_rope: RMSNorm then partial RoPE on the first R dims."""
    T, N, HD = x.shape
    R = cos.shape[-1]
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = xf * inv * (1.0 + w.float())
    if R:
        half = R // 2
        c = cos.float().unsqueeze(1)  # [T, 1, R]
        s = sin.float().unsqueeze(1)
        yr = y[..., :R]
        partner = torch.cat([yr[..., half:R], yr[..., :half]], dim=-1)
        sign = y.new_empty(R)
        sign[:half] = -1.0
        sign[half:] = 1.0
        y = y.clone()
        y[..., :R] = yr * c + sign * partner * s
    return y.to(x.dtype).contiguous()


def qk_norm_rope(q, k, cos, sin, wq, wk, eps: float):
    return norm_rope(q, cos, sin, wq, eps), norm_rope(k, cos, sin, wk, eps)


def gate(o, g):
    T, H, HD = o.shape
    return (o.float() * torch.sigmoid(g.float())).to(g.dtype).reshape(T, H * HD)


def gemma_norm_rope(x, cos, sin, w, eps: float):
    T, N, HD = x.shape
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = xf * inv
    if w is not None:
        y = y * w.float()
    if cos is not None:
        half = HD // 2
        y1, y2 = y[..., :half], y[..., half:]
        c, s = cos.float(), sin.float()
        # rotate-half with cos/sin laid out as in Gemma (full head)
        c1, c2 = c[..., :half], c[..., half:]
        s1, s2 = s[..., :half], s[..., half:]
        y = torch.cat([y1 * c1 - y2 * s1, y1 * s2 + y2 * c2], dim=-1)
    return y.to(x.dtype).contiguous()


def sandwich(d, h, w1, eps: float, scale=None, w2=None):
    df = d.float()
    inv = torch.rsqrt(df.pow(2).mean(-1, keepdim=True) + eps)
    out_h = h.float() + df * inv * w1.float()
    if scale is not None:
        out_h = out_h * scale.float()
    hb = out_h.to(h.dtype)
    if w2 is None:
        return hb, None
    hf = hb.float()
    inv2 = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps)
    return hb, (hf * inv2 * w2.float()).to(h.dtype)


def geglu(a, b):
    af, bf = a.float(), b.float()
    z = 0.7978845608028654 * (af + 0.044715 * af * af * af)
    g = 0.5 * af * (1.0 + torch.tanh(z))
    return (g * bf).to(a.dtype)


def attention_config(head_dim: int, dtype) -> dict:
    return {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1, "SPLIT": 1}


def block_ranges_ref(lo1, hi1, lo2, hi2, block_m: int, block_n: int = 0, exact: bool = False,
                     groups: int = 0, tiles_from=None) -> torch.Tensor:
    """Same as kernels/attention_triton.block_ranges_ref without a Triton dependency."""
    from .. import settings

    T = lo1.shape[0]
    dev = lo1.device
    big = torch.iinfo(torch.int32).max
    ar = torch.arange(T, device=dev)
    if groups:
        gsrc = tiles_from or (lo1, hi1, lo2, hi2)
        if settings.batch_invariant:
            g = torch.where(gsrc[0] < gsrc[1],
                            gsrc[0].long() * 2 ** 32 + torch.where(gsrc[2] < gsrc[3], gsrc[2].long() + 1, 0), -1)
        else:
            g = torch.where(gsrc[0] < gsrc[1], 2 * gsrc[0].long() + (gsrc[2] < gsrc[3]).long(), -1)
        new = torch.ones(T, dtype=torch.bool, device=dev)
        new[1:] = g[1:] != g[:-1]
        first = torch.cummax(torch.where(new, ar, 0), 0).values
        tid = torch.cumsum((new | ((ar - first) % block_m == 0)).long(), 0) - 1
        n = _cdiv(T, block_m) + groups
    else:
        tid = ar // block_m
        n = _cdiv(T, block_m)

    def red(x, how, fill):
        return torch.full((n,), fill, dtype=x.dtype, device=dev).scatter_reduce(0, tid, x, how, include_self=True)

    row_hi = red(ar + 1, "amax", 0)
    row_lo = torch.where(row_hi > 0, red(ar, "amin", T), 0)

    def tile(lo, hi):
        lo, hi = lo.int(), hi.int()
        empty = lo >= hi
        ulo = red(torch.where(empty, big, lo), "amin", big)
        uhi = red(torch.where(empty, 0, hi), "amax", 0)
        has = ulo < uhi
        ulo, uhi = torch.where(has, ulo, 0), torch.where(has, uhi, 0)
        if not exact or not block_n:
            return ulo, uhi, uhi, uhi
        any_empty = red(empty.int(), "amax", 0) > 0
        ilo = red(lo, "amax", -big)
        ihi = red(hi, "amin", big)
        f0 = ulo + (ilo - ulo + block_n - 1).clamp(min=0) // block_n * block_n
        f1 = f0 + (ihi - f0).clamp(min=0) // block_n * block_n
        ok = has & ~any_empty & (f1 > f0)
        return ulo, torch.where(ok, f0, uhi), torch.where(ok, f1, uhi), uhi

    return torch.stack([row_lo.int(), row_hi.int(), *tile(lo1, hi1), *tile(lo2, hi2)], 1).to(torch.int32).contiguous()


def attention_runs(ranges, by_row: bool = False) -> int:
    import numpy as np

    from .. import settings

    r = np.asarray(ranges)
    if by_row:
        r = r.T
    if settings.batch_invariant:
        g = np.where(r[0] < r[1],
                     r[0].astype(np.int64) * 2 ** 32 + np.where(r[2] < r[3], r[2].astype(np.int64) + 1, 0), -1)
    else:
        g = np.where(r[0] < r[1], 2 * r[0].astype(np.int64) + (r[2] < r[3]), -1)
    return 1 + int(np.count_nonzero(g[1:] != g[:-1])) if g.size else 0


def attention(q, k1, v1, k2, v2, q_meta, k1_meta, k2_meta, ranges, scale: float, window: int = 0,
              cfg: dict | None = None, out=None, qranges=None):
    """Packed multi-source attention in PyTorch (see tests/test_attention.py dense reference)."""
    Tq, H, D = q.shape
    if out is None:
        out = torch.empty(Tq, H, D, device=q.device, dtype=q.dtype)
    if Tq == 0:
        return out

    def expand(k, v):
        g = H // k.shape[1]
        return k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)

    k1e, v1e = expand(k1, v1)
    k2e, v2e = expand(k2, v2)
    # Prefer exact per-query ranges when provided (packed fork path).
    if qranges is not None:
        lo1, hi1, lo2, hi2 = qranges.long()
        same = k1.data_ptr() == k2.data_ptr() and v1.data_ptr() == v2.data_ptr()
        if same:
            kk, vv = k1e, v1e
            Tk = kk.shape[0]
            idx = torch.arange(Tk, device=q.device)
            mask = ((idx[None, :] >= lo1[:, None]) & (idx[None, :] < hi1[:, None])) | (
                (idx[None, :] >= lo2[:, None]) & (idx[None, :] < hi2[:, None]))
        else:
            Tk1, Tk2 = k1e.shape[0], k2e.shape[0]
            kk = torch.cat([k1e, k2e], 0)
            vv = torch.cat([v1e, v2e], 0)
            idx1 = torch.arange(Tk1, device=q.device)
            idx2 = torch.arange(Tk2, device=q.device)
            m1 = (idx1[None, :] >= lo1[:, None]) & (idx1[None, :] < hi1[:, None])
            m2 = (idx2[None, :] >= lo2[:, None]) & (idx2[None, :] < hi2[:, None])
            mask = torch.cat([m1, m2], 1)
        # [1, H, Tq, D] SDPA with bool mask (True = keep).
        qh = q.transpose(0, 1).unsqueeze(0)
        kh = kk.transpose(0, 1).unsqueeze(0)
        vh = vv.transpose(0, 1).unsqueeze(0)
        # Broadcast mask to [1, 1, Tq, Tk]
        attn_mask = mask.unsqueeze(0).unsqueeze(0)
        y = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_mask, scale=scale)
        out.copy_(y.squeeze(0).transpose(0, 1).contiguous().to(q.dtype))
        return out

    qj, qb, qp = (x.long() for x in q_meta[:3])
    blocks = len(q_meta) == 4
    qx = q_meta[3].long() if blocks else None

    def score_source(k, v, km):
        kj, kb, kp = (x.long() for x in km[:3])
        seen = kp[None, :] <= qp[:, None]
        if blocks:
            kx = km[3].long()
            seen = seen | ((qx[:, None] == kx[None, :]) & (qx[:, None] >= 0))
        vis = (qj[:, None] == kj[None, :]) & ((kb[None, :] == -1) | (kb[None, :] == qb[:, None])) & seen
        if window:
            vis = vis & ((qp[:, None] - kp[None, :]) < window)
        s = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
        s = s.masked_fill(~vis[None], float("-inf"))
        p = torch.nan_to_num(torch.softmax(s, -1), nan=0.0)
        return torch.einsum("hqk,khd->qhd", p, v.float())

    # Two sources: when identical tensors, one pass is enough.
    if k1.data_ptr() == k2.data_ptr() and v1.data_ptr() == v2.data_ptr():
        y = score_source(k1e, v1e, k1_meta)
    else:
        # Softmax across both sources requires a joint pass; fall back to concatenating keys.
        Tk1, Tk2 = k1e.shape[0], k2e.shape[0]
        kk = torch.cat([k1e, k2e], 0)
        vv = torch.cat([v1e, v2e], 0)
        # Build joint metadata by shifting source-2 positions into a fake job split via ranges is hard;
        # use sequential masking from each meta with renormalization via log-sum-exp merge.
        y1 = score_source(k1e, v1e, k1_meta)
        y2 = score_source(k2e, v2e, k2_meta)
        # Approximate when both sources contribute: average is wrong; recompute with cat metas.
        kj1, kb1, kp1 = (x.long() for x in k1_meta[:3])
        kj2, kb2, kp2 = (x.long() for x in k2_meta[:3])
        kj = torch.cat([kj1, kj2])
        kb = torch.cat([kb1, kb2])
        kp = torch.cat([kp1, kp2])
        km = (kj, kb, kp) + ((torch.cat([k1_meta[3].long(), k2_meta[3].long()]),) if blocks else ())
        y = score_source(kk, vv, km)
        del y1, y2
    out.copy_(y.to(q.dtype))
    return out


def causal_conv1d(x, weight, bias=None, activation=None, initial_state=None, output_final_state=False,
                  cu_seqlens=None, cu_seqlens_cpu=None, **_):
    """Depthwise causal conv1d over a packed sequence batch (FLA-compatible subset).

    x: [1, T, C], weight: [C, W] or [C, 1, W]
    """
    if x.dim() == 3 and x.shape[0] == 1:
        x = x[0]
    T, C = x.shape
    w = weight.squeeze(1) if weight.dim() == 3 else weight
    W = w.shape[-1]
    # Depthwise: each channel convolves with its kernel.
    xt = x.T.unsqueeze(0)  # [1, C, T]
    wt = w.unsqueeze(1)  # [C, 1, W]
    pad = W - 1
    if initial_state is not None:
        # initial_state [N, C, W-1] per sequence; apply per cu_seqlens segment.
        pass
    y = F.conv1d(F.pad(xt, (pad, 0)), wt, groups=C)
    y = y.squeeze(0).T  # [T, C]
    if bias is not None:
        y = y + bias
    if activation in ("silu", "swish"):
        y = F.silu(y)
    elif activation == "relu":
        y = F.relu(y)

    if cu_seqlens is None:
        state = y[-pad:].T.contiguous() if output_final_state and pad else None
        if output_final_state:
            return y.unsqueeze(0), (state.unsqueeze(0) if state is not None else None)
        return y.unsqueeze(0), None

    cu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    cu = cu.tolist() if hasattr(cu, "tolist") else list(cu)
    outs = []
    states = []
    for i in range(len(cu) - 1):
        a, b = int(cu[i]), int(cu[i + 1])
        seg = x[a:b]
        if seg.numel() == 0:
            outs.append(seg)
            if output_final_state:
                states.append(torch.zeros(C, pad, device=x.device, dtype=x.dtype) if pad else None)
            continue
        init = None
        if initial_state is not None:
            init = initial_state[i]
        st = F.pad(seg.T.unsqueeze(0), (pad, 0))
        if init is not None and pad:
            # init [C, W-1]
            st[..., :pad] = init.unsqueeze(0)
        yi = F.conv1d(st, wt, groups=C).squeeze(0).T
        if bias is not None:
            yi = yi + bias
        if activation in ("silu", "swish"):
            yi = F.silu(yi)
        elif activation == "relu":
            yi = F.relu(yi)
        outs.append(yi)
        if output_final_state:
            # last W-1 inputs after activation path: use padded input window
            window = st.squeeze(0)[..., -pad:] if pad else None
            states.append(window.contiguous() if window is not None else None)
    y = torch.cat(outs, 0) if outs else x.new_zeros(0, C)
    if output_final_state:
        S = torch.stack(states, 0) if states and states[0] is not None else None
        return y.unsqueeze(0), S
    return y.unsqueeze(0), None


def chunk_gated_delta_rule(q, k, v, g=None, beta=None, initial_state=None, output_final_state=False,
                           cu_seqlens=None, cu_seqlens_cpu=None, use_qk_l2norm_in_kernel=False,
                           use_gate_in_kernel=False, A_log=None, dt_bias=None,
                           use_beta_sigmoid_in_kernel=False, **_):
    """Recurrent Gated DeltaNet over packed sequences (correct, not CUDA-fast).

    q,k: [1, T, Hk, Dk], v: [1, T, Hv, Dv]. State layout S [Hv, Dv, Dk].
    Update: S = exp(g) * S + beta * (v - S k) k^T ; o = S q
    """
    q, k, v = q[0], k[0], v[0]
    T, Hk, Dk = q.shape
    Hv, Dv = v.shape[1], v.shape[2]
    group = max(Hv // Hk, 1)
    if use_qk_l2norm_in_kernel:
        q = F.normalize(q.float(), dim=-1).to(q.dtype)
        k = F.normalize(k.float(), dim=-1).to(k.dtype)
    if beta is None:
        beta_t = torch.ones(T, Hv, device=q.device, dtype=torch.float32)
    else:
        beta_t = beta[0] if beta.dim() == 3 else beta
        beta_t = torch.sigmoid(beta_t.float()) if use_beta_sigmoid_in_kernel else beta_t.float()
        if beta_t.dim() == 1:
            beta_t = beta_t[:, None].expand(T, Hv)
        if beta_t.shape[-1] == Hk and Hv != Hk:
            beta_t = beta_t.repeat_interleave(group, -1)
    if g is None:
        g_t = torch.zeros(T, Hv, device=q.device, dtype=torch.float32)
    else:
        g_t = (g[0] if g.dim() == 3 else g).float()
        if g_t.dim() == 1:
            g_t = g_t[:, None].expand(T, Hv)
        if g_t.shape[-1] == Hk and Hv != Hk:
            g_t = g_t.repeat_interleave(group, -1)
    if use_gate_in_kernel and A_log is not None:
        a = A_log.float()
        if a.numel() == Hk and Hv != Hk:
            a = a.repeat_interleave(group, 0)
        dt = g_t if dt_bias is None else g_t + dt_bias.float().repeat_interleave(group, 0)[:Hv]
        g_t = -a.exp() * F.softplus(dt)

    cu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    bounds = [0, T] if cu is None else (cu.tolist() if hasattr(cu, "tolist") else list(cu))
    outs, final_states = [], []
    for si in range(len(bounds) - 1):
        a, b = int(bounds[si]), int(bounds[si + 1])
        S = (initial_state[si].float().clone() if initial_state is not None
             else torch.zeros(Hv, Dv, Dk, device=q.device, dtype=torch.float32))
        o_seg = []
        for t in range(a, b):
            qt = q[t].float().repeat_interleave(group, 0)[:Hv]
            kt = k[t].float().repeat_interleave(group, 0)[:Hv]
            vt = v[t].float()
            decay = torch.exp(g_t[t]).view(Hv, 1, 1)
            S = S * decay
            sk = torch.einsum("hdk,hk->hd", S, kt)
            delta = (vt - sk) * beta_t[t].view(Hv, 1)
            S = S + torch.einsum("hd,hk->hdk", delta, kt)
            o_seg.append(torch.einsum("hdk,hk->hd", S, qt))
        outs.append(torch.stack(o_seg, 0) if o_seg else q.new_zeros(0, Hv, Dv))
        if output_final_state:
            final_states.append(S)
    o = torch.cat(outs, 0).unsqueeze(0).to(v.dtype)
    return o, (torch.stack(final_states, 0) if output_final_state else None)
