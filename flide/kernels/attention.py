"""Packed attention: Triton on CUDA, eager PyTorch on MPS/CPU."""
from __future__ import annotations

from .backend import use_triton


def config(device, dtype, head_dim: int) -> dict:
    if not use_triton(device):
        from . import eager

        return eager.attention_config(head_dim, dtype)
    from . import attention_triton as t

    return t.config(device, dtype, head_dim)


def block_ranges(lo1, hi1, lo2, hi2, block_m: int, block_n: int = 0, exact: bool = False,
                 groups: int = 0, tiles_from=None):
    if not use_triton(lo1.device):
        from . import eager

        return eager.block_ranges_ref(lo1, hi1, lo2, hi2, block_m, block_n, exact, groups, tiles_from)
    from . import attention_triton as t

    return t.block_ranges(lo1, hi1, lo2, hi2, block_m, block_n, exact, groups, tiles_from)


def block_ranges_ref(lo1, hi1, lo2, hi2, block_m: int, block_n: int = 0, exact: bool = False,
                     groups: int = 0, tiles_from=None):
    from . import eager

    return eager.block_ranges_ref(lo1, hi1, lo2, hi2, block_m, block_n, exact, groups, tiles_from)


def runs(ranges, by_row: bool = False) -> int:
    from . import eager

    return eager.attention_runs(ranges, by_row=by_row)


def attention(q, k1, v1, k2, v2, q_meta, k1_meta, k2_meta, ranges, scale: float, window: int = 0,
              cfg: dict | None = None, out=None, qranges=None):
    if not use_triton(q.device):
        from . import eager

        return eager.attention(q, k1, v1, k2, v2, q_meta, k1_meta, k2_meta, ranges, scale, window, cfg, out,
                               qranges)
    from . import attention_triton as t

    return t.attention(q, k1, v1, k2, v2, q_meta, k1_meta, k2_meta, ranges, scale, window, cfg, out, qranges)
