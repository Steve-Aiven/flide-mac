"""Parity and smoke for Metal pointwise vs eager."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.kernels import eager
from flide.kernels.metal import pointwise as metal_pw
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
    for name in ("rms_norm_f16", "swiglu_f16", "gated_rms_f16", "norm_rope_f16", "gate_f16"):
        assert hasattr(lib, name), name
    print("compile ok")


def test_rms_parity():
    if _skip():
        return
    torch.manual_seed(0)
    T, N, eps = 64, 512, 1e-6
    x = torch.randn(T, N, device="mps", dtype=torch.float16)
    w = torch.randn(N, device="mps", dtype=torch.float16) * 0.1
    y_m = metal_pw.rms_norm(x, w, eps, offset=True)
    y_e = eager.rms_norm(x, w, eps, offset=True)
    err = (y_m.float() - y_e.float()).abs().max().item()
    print(f"rms err {err:.6f}")
    assert err < 2e-2, err

    res = torch.randn(T, N, device="mps", dtype=torch.float16)
    h_m, y_m = metal_pw.rms_norm(x, w, eps, residual=res, offset=True)
    h_e, y_e = eager.rms_norm(x, w, eps, residual=res, offset=True)
    err_h = (h_m.float() - h_e.float()).abs().max().item()
    err_y = (y_m.float() - y_e.float()).abs().max().item()
    print(f"rms+res h {err_h:.6f}, y {err_y:.6f}")
    assert err_h < 2e-2 and err_y < 2e-2


def test_swiglu_parity():
    if _skip():
        return
    torch.manual_seed(1)
    T, n = 48, 768
    gu = torch.randn(T, 2 * n, device="mps", dtype=torch.float16)
    y_m = metal_pw.swiglu(gu, n)
    y_e = eager.swiglu(gu, n)
    err = (y_m.float() - y_e.float()).abs().max().item()
    print(f"swiglu err {err:.6f}")
    assert err < 2e-2, err


def test_gated_rms_parity():
    if _skip():
        return
    torch.manual_seed(2)
    T, H, DV, eps = 32, 8, 64, 1e-6
    o = torch.randn(T, H, DV, device="mps", dtype=torch.float16)
    z = torch.randn(T, H * DV, device="mps", dtype=torch.float16)
    w = torch.randn(DV, device="mps", dtype=torch.float16)
    y_m = metal_pw.gated_rms(o, z, w, eps)
    y_e = eager.gated_rms(o, z, w, eps)
    err = (y_m.float() - y_e.float()).abs().max().item()
    print(f"gated_rms err {err:.6f}")
    assert err < 2e-2, err


def test_norm_rope_parity():
    if _skip():
        return
    torch.manual_seed(3)
    T, Heads, HD, R, eps = 40, 8, 64, 32, 1e-6
    x = torch.randn(T, Heads, HD, device="mps", dtype=torch.float16)
    cos = torch.randn(T, R, device="mps", dtype=torch.float16)
    sin = torch.randn(T, R, device="mps", dtype=torch.float16)
    w = torch.randn(HD, device="mps", dtype=torch.float16) * 0.1
    y_m = metal_pw.norm_rope(x, cos, sin, w, eps)
    y_e = eager.norm_rope(x, cos, sin, w, eps)
    err = (y_m.float() - y_e.float()).abs().max().item()
    print(f"norm_rope err {err:.6f}")
    assert err < 3e-2, err


def test_gate_parity():
    if _skip():
        return
    torch.manual_seed(4)
    T, H, HD = 36, 8, 64
    o = torch.randn(T, H, HD, device="mps", dtype=torch.float16)
    g = torch.randn(T, H, HD, device="mps", dtype=torch.float16)
    y_m = metal_pw.gate(o, g)
    y_e = eager.gate(o, g)
    err = (y_m.float() - y_e.float()).abs().max().item()
    print(f"gate err {err:.6f}")
    assert err < 2e-2, err


def test_pointwise_speed():
    if _skip():
        return
    torch.manual_seed(0)
    T, N, eps = 256, 2048, 1e-6
    x = torch.randn(T, N, device="mps", dtype=torch.float16)
    w = torch.randn(N, device="mps", dtype=torch.float16) * 0.1
    res = torch.randn(T, N, device="mps", dtype=torch.float16)
    gu = torch.randn(T, 2 * N, device="mps", dtype=torch.float16)

    for fn in (
        lambda: metal_pw.rms_norm(x, w, eps, residual=res),
        lambda: eager.rms_norm(x, w, eps, residual=res),
        lambda: metal_pw.swiglu(gu, N),
        lambda: eager.swiglu(gu, N),
    ):
        fn()
    torch.mps.synchronize()

    def bench(fn, reps=20):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.mps.synchronize()
        return time.perf_counter() - t0

    rms_m = bench(lambda: metal_pw.rms_norm(x, w, eps, residual=res))
    rms_e = bench(lambda: eager.rms_norm(x, w, eps, residual=res))
    sw_m = bench(lambda: metal_pw.swiglu(gu, N))
    sw_e = bench(lambda: eager.swiglu(gu, N))
    print(f"rms+res 20x: metal {rms_m:.3f}s, eager {rms_e:.3f}s, speedup {rms_e / rms_m:.2f}x")
    print(f"swiglu 20x: metal {sw_m:.3f}s, eager {sw_e:.3f}s, speedup {sw_e / sw_m:.2f}x")


def main():
    test_compile()
    test_rms_parity()
    test_swiglu_parity()
    test_gated_rms_parity()
    test_norm_rope_parity()
    test_gate_parity()
    test_pointwise_speed()
    print("ok")


if __name__ == "__main__":
    main()
