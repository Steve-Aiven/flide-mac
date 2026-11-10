"""Metal kernels for Apple Silicon via torch.mps.compile_shader."""
from __future__ import annotations

from .runtime import metal_available
from .gdn import chunk_gated_delta_rule
from .conv import causal_conv1d
from . import pointwise

__all__ = ["metal_available", "chunk_gated_delta_rule", "causal_conv1d", "pointwise"]
