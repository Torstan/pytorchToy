"""
Compatibility re-exports for the FX tracer implementation.
"""

from torch.fx.proxy import (
    Proxy,
    Tracer,
    UnsupportedTraceError,
    current_tracer,
    is_tracing,
)

__all__ = [
    "Proxy",
    "Tracer",
    "UnsupportedTraceError",
    "current_tracer",
    "is_tracing",
]
