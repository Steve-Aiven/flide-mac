"""Metal causal_conv1d matching the FLA-compatible subset used by Flide's decoder."""
from __future__ import annotations

import torch

from .runtime import metal_available, shader_lib


def causal_conv1d(x, weight, bias=None, activation=None, initial_state=None, output_final_state=False,
                  cu_seqlens=None, cu_seqlens_cpu=None, **_):
    """x: [1, T, C]; weight: [C, W] or [C, 1, W]. Returns ([1, T, C], state|None)."""
    if not (metal_available() and x.is_mps):
        from .. import eager

        return eager.causal_conv1d(x, weight, bias, activation, initial_state, output_final_state,
                                   cu_seqlens, cu_seqlens_cpu)

    if x.dim() != 3 or x.shape[0] != 1:
        from .. import eager

        return eager.causal_conv1d(x, weight, bias, activation, initial_state, output_final_state,
                                   cu_seqlens, cu_seqlens_cpu)

    x_tc = x[0]  # [T, C]
    w = weight.squeeze(1) if weight.dim() == 3 else weight
    W = int(w.shape[-1])
    if W > 16:
        from .. import eager

        return eager.causal_conv1d(x, weight, bias, activation, initial_state, output_final_state,
                                   cu_seqlens, cu_seqlens_cpu)

    pad = W - 1
    T, C = x_tc.shape
    cu_src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    if cu_src is None:
        bounds = [0, T]
        N = 1
    else:
        bounds = cu_src.tolist() if hasattr(cu_src, "tolist") else list(cu_src)
        N = len(bounds) - 1

    # Prefer the single-launch varlen kernel for float16 (decoder hot path).
    if x_tc.dtype == torch.float16:
        return _varlen_f16(x_tc, w, bias, activation, initial_state, output_final_state, bounds, N, C, W, pad)

    # Float32 / other: keep the existing per-segment fwd path.
    return _per_segment(x_tc, w, bias, activation, initial_state, output_final_state, bounds, C, pad)


def _varlen_f16(x_tc, w, bias, activation, initial_state, output_final_state, bounds, N, C, W, pad):
    device = x_tc.device
    T = x_tc.shape[0]
    cu = torch.tensor(bounds, device=device, dtype=torch.int32)
    out = torch.empty_like(x_tc)
    wt = w.contiguous().half()
    has_bias = bias is not None
    bias_t = bias.contiguous().half() if has_bias else torch.zeros(C, device=device, dtype=torch.float16)
    apply_silu = 1 if activation in ("silu", "swish") else 0
    has_init = initial_state is not None and pad > 0
    if has_init:
        init = initial_state.contiguous().half()
    else:
        init = torch.zeros(N, C, max(pad, 1), device=device, dtype=torch.float16)
    write_state = bool(output_final_state and pad > 0)
    if write_state:
        out_state = torch.empty(N, C, pad, device=device, dtype=torch.float16)
    else:
        out_state = torch.empty(N, C, max(pad, 1), device=device, dtype=torch.float16)

    kn = shader_lib().causal_conv1d_varlen_f16
    tg = min(256, max(C, 1))
    kn(x_tc.contiguous(), wt, bias_t, cu, init, out, out_state,
       N, C, W, int(has_bias), apply_silu, int(has_init), int(write_state),
       threads=[C, N, 1], group_size=[tg, 1, 1])
    return out.unsqueeze(0), (out_state if write_state else None)


def _per_segment(x, w, bias, activation, initial_state, output_final_state, bounds, C, pad):
    outs, states = [], []
    for i in range(len(bounds) - 1):
        a, b = int(bounds[i]), int(bounds[i + 1])
        seg = x[a:b]
        if seg.shape[0] == 0:
            outs.append(seg)
            if output_final_state:
                states.append(torch.zeros(C, pad, device=x.device, dtype=x.dtype) if pad else None)
            continue
        init = initial_state[i] if initial_state is not None else None
        if init is not None and pad:
            stream = torch.cat([init.transpose(0, 1).contiguous(), seg], 0)
            y_full = _fwd_btc(stream.unsqueeze(0), w, bias, activation)[0]
            yi = y_full[pad:]
            st = stream[-pad:].transpose(0, 1).contiguous() if output_final_state else None
        else:
            yi = _fwd_btc(seg.unsqueeze(0), w, bias, activation)[0]
            if output_final_state and pad:
                if seg.shape[0] >= pad:
                    st = seg[-pad:].transpose(0, 1).contiguous()
                else:
                    st = torch.zeros(C, pad, device=x.device, dtype=x.dtype)
                    st[:, -seg.shape[0]:] = seg.transpose(0, 1)
            else:
                st = None
        outs.append(yi)
        if output_final_state:
            states.append(st)

    y = torch.cat(outs, 0) if outs else x.new_zeros(0, C)
    S = torch.stack(states, 0) if output_final_state and states and states[0] is not None else None
    return y.unsqueeze(0), S


def _fwd_btc(x_btc: torch.Tensor, weight: torch.Tensor, bias, activation) -> torch.Tensor:
    """x: [B, T, C] -> [B, T, C] via Metal depthwise causal conv."""
    B, T, C = x_btc.shape
    w = weight.squeeze(1) if weight.dim() == 3 else weight
    W = int(w.shape[-1])
    x = x_btc.transpose(1, 2).contiguous()  # [B, C, T]
    if x_btc.dtype == torch.float16:
        name, xt, wt = "causal_conv1d_fwd_f16", x.half(), w.contiguous().half()
    else:
        name, xt, wt = "causal_conv1d_fwd_f32", x.float(), w.contiguous().float()
    out = torch.empty_like(xt)
    has_bias = bias is not None
    bias_t = bias.contiguous().to(xt.dtype) if has_bias else torch.zeros(C, device=x.device, dtype=xt.dtype)
    apply_silu = 1 if activation in ("silu", "swish") else 0
    kn = getattr(shader_lib(), name)
    tg = min(256, max(T, 1))
    kn(xt, wt, bias_t, out, B, C, T, W, int(has_bias), apply_silu,
       threads=[max(T, 1), C, B], group_size=[tg, 1, 1])
    return out.transpose(1, 2).contiguous().to(x_btc.dtype)
