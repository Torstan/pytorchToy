"""
最小 torch.fx 兼容层。
"""

from .graph import Graph, GraphModule, Node
from .meta import propagate_meta
from .proxy import (
    Proxy,
    Tracer,
    UnsupportedTraceError,
    current_tracer,
    is_tracing,
)

__all__ = [
    "Graph",
    "GraphModule",
    "Node",
    "Proxy",
    "Tracer",
    "UnsupportedTraceError",
    "current_tracer",
    "is_tracing",
    "propagate_meta",
]
