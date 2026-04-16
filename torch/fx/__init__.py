"""
最小 torch.fx 兼容层。

当前阶段直接复用 torch._compile.graph 中已经存在的图数据结构，
避免在迁移早期维护两套 IR。
"""

from torch._compile.graph import Graph, GraphModule, Node

from .meta import propagate_meta

__all__ = [
    "Graph",
    "GraphModule",
    "Node",
    "propagate_meta",
]
