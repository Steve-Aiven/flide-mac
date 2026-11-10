"""Metal Gated DeltaNet (recurrent) matching Flide's FLA call shape."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .runtime import metal_available, shader_lib


def chunk_gated_delta_rule(q, k, v, g=None, beta=None, initial_state=None, output_final_state=False,
                           cu_seqlens=None, cu_seqlens_cpu=None, use_qk_l2norm_in_kernel=False,
                           use_gate_in_kernel=False, A_log=None, dt_bias=None,
                           use_beta_sigmoid_in_kernel=False, **_):
    """q,k: [1, T, Hk, Dk], v: [1, T, Hv, Dv]. Returns o [1, T, Hv, Dv], state [N, Hv, Dv, Dk]|None."""
    if not (metal_available() and q.is_mps):
        from .. import eager

        return eager.chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=initial_state, output_final_state=output_final_state,
            cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log, dt_bias=dt_bias, use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        )

    q0, k0, v0 = q[0], k[0], v[0]
    T, Hk, Dk = q0.shape
    Hv, Dv = v0.shape[1], v0.shape[2]
    group = max(Hv // Hk, 1)
    device = q.device

    if use_qk_l2norm_in_kernel:
        q0 = F.normalize(q0.float(), dim=-1).to(q0.dtype)
        k0 = F.normalize(k0.float(), dim=-1).to(k0.dtype)

    # Expand Q/K heads to V heads.
    if group > 1:
        q0 = q0.repeat_interleave(group, 1)[:, :Hv]
        k0 = k0.repeat_interleave(group, 1)[:, :Hv]

    if beta is None:
        beta_t = torch.ones(T, Hv, device=device, dtype=torch.float16)
    else:
        beta_t = beta[0] if beta.dim() == 3 else beta
        beta_t = torch.sigmoid(beta_t.float()) if use_beta_sigmoid_in_kernel else beta_t.float()
        if beta_t.dim() == 1:
            beta_t = beta_t[:, None].expand(T, Hv)
        if beta_t.shape[-1] == Hk and Hv != Hk:
            beta_t = beta_t.repeat_interleave(group, -1)[:, :Hv]
        beta_t = beta_t.to(torch.float16)

    if g is None:
        g_t = torch.zeros(T, Hv, device=device, dtype=torch.float32)
    else:
        g_t = (g[0] if g.dim() == 3 else g).float()
        if g_t.dim() == 1:
            g_t = g_t[:, None].expand(T, Hv)
        if g_t.shape[-1] == Hk and Hv != Hk:
            g_t = g_t.repeat_interleave(group, -1)[:, :Hv]
    if use_gate_in_kernel and A_log is not None:
        a = A_log.float()
        if a.numel() == Hk and Hv != Hk:
            a = a.repeat_interleave(group, 0)[:Hv]
        dt = g_t if dt_bias is None else g_t + dt_bias.float().repeat_interleave(group, 0)[:Hv]
        g_t = -a.exp() * F.softplus(dt)

    cu_src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    bounds = [0, T] if cu_src is None else (cu_src.tolist() if hasattr(cu_src, "tolist") else list(cu_src))
    N = len(bounds) - 1

    # Packed layouts for the varlen kernel: [T, H, K/V], g/beta [T, H], state [N, H, V, K].
    qh = q0.contiguous().half()
    kh = k0.contiguous().half()
    vh = v0.contiguous().half()
    gh = g_t.contiguous().float()
    bh = beta_t.contiguous().half()
    cu = torch.tensor(bounds, device=device, dtype=torch.int32)

    if initial_state is not None:
        st = initial_state.float().clone().contiguous()
    else:
        st = torch.zeros(N, Hv, Dv, Dk, device=device, dtype=torch.float32)
    out = torch.empty(T, Hv, Dv, device=device, dtype=torch.float16)

    lib = shader_lib()
    tg = min(256, max(Dv, 1))
    # One threadgroup per (head, segment).
    lib.gdn_recurrent_varlen(
        qh, kh, vh, gh, bh, cu, st, out,
        N, Hv, Dk, Dv,
        threads=[Hv * tg, N, 1], group_size=[tg, 1, 1],
    )

    o = out.unsqueeze(0).to(v.dtype)
    state = st if output_final_state else None
    return o, state
