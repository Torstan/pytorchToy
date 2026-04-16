"""
最小 torch._dynamo 兼容层。

当前阶段先提供公共 API 形状：
- optimize
- list_backends
- disable
- reset
"""

from . import config
from .eval_frame import OptimizedFunction, optimize, disable, reset
from .backends.registry import list_backends, lookup_backend

__all__ = [
    "OptimizedFunction",
    "config",
    "optimize",
    "disable",
    "reset",
    "list_backends",
    "lookup_backend",
]
