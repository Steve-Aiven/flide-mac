"""flide-mac: Flide fork with Apple Silicon (MPS) support for decision models."""
from __future__ import annotations

__version__ = "0.1.12+mac"


def __getattr__(name):
    if name == "Engine":
        from .runtime.engine import Engine

        return Engine
    raise AttributeError(name)


__all__ = ["Engine", "__version__"]
