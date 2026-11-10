"""Parity and smoke for Metal GDN / causal_conv vs eager."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.kernels import eager
from flide.kernels.metal.runtime import metal_available


def _skip():
    if not metal_available():
        print("skip: metal unavailable")
        return True
    return False


def test_compile():
    if _skip():
        return
    from flide.kernels.metal.runtime import shader_lib

    lib = shader_lib()
    assert hasattr(lib, "gdn_recurrent_fused")
    assert hasattr(lib, "gdn_recurrent_varlen")
    assert hasattr(lib, "causal_conv1d_fwd_f16")
    assert hasattr(lib, "causal_conv1d_varlen_f16")
    print("compile ok")


def test_conv_parity():
    if _skip():
        return
    from flide.kernels import metal

    torch.manual_seed(0)
    B, T, C, W = 1, 64, 32, 4
    x = torch.randn(B, T, C, device="mps", dtype=torch.float16)
    w = torch.randn(C, W, device="mps", dtype=torch.float16)
    y_m, _ = metal.causal_conv1d(x, w, activation="silu", output_final_state=False)
    y_e, _ = eager.causal_conv1d(x, w, activation="silu", output_final_state=False)
    err = (y_m.float() - y_e.float()).abs().max().item()
    print(f"conv max abs err {err:.6f}")
    assert err < 2e-2, err


def test_conv_varlen_parity():
    if _skip():
        return
    from flide.kernels import metal

    torch.manual_seed(2)
    C, W = 48, 4
    lengths = [7, 1, 13, 5, 1]
    T = sum(lengths)
    cu = torch.tensor([0, *torch.cumsum(torch.tensor(lengths), 0).tolist()], dtype=torch.int32)
    x = torch.randn(1, T, C, device="mps", dtype=torch.float16)
    w = torch.randn(C, W, device="mps", dtype=torch.float16)
    init = torch.randn(len(lengths), C, W - 1, device="mps", dtype=torch.float16)
    kwargs = dict(activation="silu", initial_state=init, output_final_state=True,
                  cu_seqlens=cu.to("mps"), cu_seqlens_cpu=cu)
    y_m, s_m = metal.causal_conv1d(x, w, **kwargs)
    y_e, s_e = eager.causal_conv1d(x, w, **kwargs)
    err_y = (y_m.float() - y_e.float()).abs().max().item()
    err_s = (s_m.float() - s_e.float()).abs().max().item()
    print(f"conv varlen N={len(lengths)} out {err_y:.6f}, state {err_s:.6f}")
    assert err_y < 2e-2, err_y
    assert err_s < 2e-2, err_s


def test_gdn_parity():
    if _skip():
        return
    from flide.kernels import metal

    torch.manual_seed(1)
    T, Hk, Hv, Dk, Dv = 48, 2, 4, 16, 32
    q = torch.randn(1, T, Hk, Dk, device="mps", dtype=torch.float16)
    k = torch.randn(1, T, Hk, Dk, device="mps", dtype=torch.float16)
    v = torch.randn(1, T, Hv, Dv, device="mps", dtype=torch.float16)
    g = torch.randn(1, T, Hv, device="mps", dtype=torch.float32) * 0.1 - 1.0
    beta = torch.rand(1, T, Hv, device="mps", dtype=torch.float16)
    kwargs = dict(g=g, beta=beta, output_final_state=True, use_qk_l2norm_in_kernel=True,
                  use_beta_sigmoid_in_kernel=True)
    o_m, s_m = metal.chunk_gated_delta_rule(q, k, v, **kwargs)
    o_e, s_e = eager.chunk_gated_delta_rule(q, k, v, **kwargs)
    err_o = (o_m.float() - o_e.float()).abs().max().item()
    err_s = (s_m.float() - s_e.float()).abs().max().item()
    print(f"gdn out max abs err {err_o:.6f}, state {err_s:.6f}")
    assert err_o < 5e-2, err_o
    assert err_s < 5e-2, err_s


def test_gdn_varlen_parity():
    if _skip():
        return
    from flide.kernels import metal

    torch.manual_seed(3)
    Hk, Hv, Dk, Dv = 2, 4, 16, 32
    lengths = [11, 1, 9, 4, 1, 6]
    T = sum(lengths)
    cu = torch.tensor([0, *torch.cumsum(torch.tensor(lengths), 0).tolist()], dtype=torch.int32)
    q = torch.randn(1, T, Hk, Dk, device="mps", dtype=torch.float16)
    k = torch.randn(1, T, Hk, Dk, device="mps", dtype=torch.float16)
    v = torch.randn(1, T, Hv, Dv, device="mps", dtype=torch.float16)
    g = torch.randn(1, T, Hv, device="mps", dtype=torch.float32) * 0.05 - 0.5
    beta = torch.rand(1, T, Hv, device="mps", dtype=torch.float16)
    init = torch.randn(len(lengths), Hv, Dv, Dk, device="mps", dtype=torch.float32) * 0.01
    kwargs = dict(g=g, beta=beta, initial_state=init, output_final_state=True,
                  cu_seqlens=cu.to("mps"), cu_seqlens_cpu=cu,
                  use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True)
    o_m, s_m = metal.chunk_gated_delta_rule(q, k, v, **kwargs)
    o_e, s_e = eager.chunk_gated_delta_rule(q, k, v, **kwargs)
    err_o = (o_m.float() - o_e.float()).abs().max().item()
    err_s = (s_m.float() - s_e.float()).abs().max().item()
    print(f"gdn varlen N={len(lengths)} out {err_o:.6f}, state {err_s:.6f}")
    assert err_o < 5e-2, err_o
    assert err_s < 5e-2, err_s


def test_gdn_speed():
    if _skip():
        return
    from flide.kernels import metal

    torch.manual_seed(0)
    T, Hk, Hv, Dk, Dv = 256, 4, 8, 64, 64
    q = torch.randn(1, T, Hk, Dk, device="mps", dtype=torch.float16)
    k = torch.randn(1, T, Hk, Dk, device="mps", dtype=torch.float16)
    v = torch.randn(1, T, Hv, Dv, device="mps", dtype=torch.float16)
    g = torch.zeros(1, T, Hv, device="mps", dtype=torch.float32)
    beta = torch.ones(1, T, Hv, device="mps", dtype=torch.float16)
    kwargs = dict(g=g, beta=beta, output_final_state=False, use_qk_l2norm_in_kernel=True)

    metal.chunk_gated_delta_rule(q, k, v, **kwargs)
    eager.chunk_gated_delta_rule(q, k, v, **kwargs)
    torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(5):
        metal.chunk_gated_delta_rule(q, k, v, **kwargs)
    torch.mps.synchronize()
    metal_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(5):
        eager.chunk_gated_delta_rule(q, k, v, **kwargs)
    torch.mps.synchronize()
    eager_s = time.perf_counter() - t0
    print(f"gdn 5x T={T}: metal {metal_s:.3f}s, eager {eager_s:.3f}s, speedup {eager_s / metal_s:.2f}x")


def test_varlen_multi_seg_speed():
    """Many short segments: the workload that used to mean one Metal launch per segment."""
    if _skip():
        return
    from flide.kernels import metal

    torch.manual_seed(0)
    N, L, Hk, Hv, Dk, Dv = 32, 8, 4, 8, 64, 64
    T = N * L
    cu = torch.arange(0, T + 1, L, dtype=torch.int32)
    q = torch.randn(1, T, Hk, Dk, device="mps", dtype=torch.float16)
    k = torch.randn(1, T, Hk, Dk, device="mps", dtype=torch.float16)
    v = torch.randn(1, T, Hv, Dv, device="mps", dtype=torch.float16)
    g = torch.zeros(1, T, Hv, device="mps", dtype=torch.float32)
    beta = torch.ones(1, T, Hv, device="mps", dtype=torch.float16)
    kwargs = dict(g=g, beta=beta, output_final_state=False, use_qk_l2norm_in_kernel=True,
                  cu_seqlens=cu.to("mps"), cu_seqlens_cpu=cu)

    metal.chunk_gated_delta_rule(q, k, v, **kwargs)
    eager.chunk_gated_delta_rule(q, k, v, **kwargs)
    torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(10):
        metal.chunk_gated_delta_rule(q, k, v, **kwargs)
    torch.mps.synchronize()
    metal_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(10):
        eager.chunk_gated_delta_rule(q, k, v, **kwargs)
    torch.mps.synchronize()
    eager_s = time.perf_counter() - t0
    print(f"gdn varlen 10x N={N} L={L}: metal {metal_s:.3f}s, eager {eager_s:.3f}s, "
          f"speedup {eager_s / metal_s:.2f}x")


def main():
    test_compile()
    test_conv_parity()
    test_conv_varlen_parity()
    test_gdn_parity()
    test_gdn_varlen_parity()
    test_gdn_speed()
    test_varlen_multi_seg_speed()
    print("ok")


if __name__ == "__main__":
    main()
