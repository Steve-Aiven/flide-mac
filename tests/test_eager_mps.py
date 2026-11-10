"""Smoke tests for eager kernels used on Apple Silicon (CPU-safe)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.kernels import eager, pointwise
from flide.runtime.device import resolve_device, resolve_dtype


def test_resolve_auto_prefers_available_backend():
    d = resolve_device("auto")
    assert d.startswith("cuda") or d in ("mps", "cpu") or d.startswith("mps")
    assert resolve_dtype(d, "auto") in ("bfloat16", "float16")


def test_pointwise_rms_swiglu():
    x = torch.randn(4, 32)
    w = torch.zeros(32)
    y = pointwise.rms_norm(x, w, 1e-6)
    assert y.shape == x.shape
    gu = torch.randn(4, 64)
    assert pointwise.swiglu(gu, 32).shape == (4, 32)


def test_eager_gdn_shapes():
    T, Hk, Hv, Dk, Dv = 8, 2, 4, 16, 16
    q = torch.randn(1, T, Hk, Dk)
    k = torch.randn(1, T, Hk, Dk)
    v = torch.randn(1, T, Hv, Dv)
    g = torch.zeros(1, T, Hv)
    beta = torch.ones(1, T, Hv)
    o, st = eager.chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=False,
    )
    assert o.shape == (1, T, Hv, Dv)
    assert st is not None and st.shape[0] == 1


def test_qranges_attention_matches_loop():
    torch.manual_seed(0)
    Tq, H, Hk, D = 5, 4, 2, 8
    q = torch.randn(Tq, H, D)
    k = torch.randn(Tq, Hk, D)
    v = torch.randn(Tq, Hk, D)
    meta = (torch.zeros(Tq, dtype=torch.int32), torch.full((Tq,), -1, dtype=torch.int32),
            torch.arange(Tq, dtype=torch.int32))
    lo1 = torch.zeros(Tq, dtype=torch.int32)
    hi1 = torch.arange(1, Tq + 1, dtype=torch.int32)
    lo2 = torch.zeros(Tq, dtype=torch.int32)
    hi2 = torch.zeros(Tq, dtype=torch.int32)
    qranges = torch.stack([lo1, hi1, lo2, hi2])
    R = torch.zeros(1, 10, dtype=torch.int32)
    y = eager.attention(q, k, v, k, v, meta, meta, meta, R, scale=D ** -0.5, qranges=qranges)
    assert y.shape == (Tq, H, D)
    assert torch.isfinite(y).all()


def main():
    test_resolve_auto_prefers_available_backend()
    test_pointwise_rms_swiglu()
    test_eager_gdn_shapes()
    test_qranges_attention_matches_loop()
    print("ok")


if __name__ == "__main__":
    main()
