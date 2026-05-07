"""
Compatibility re-exports for the FX graph implementation.
"""

from torch.fx.graph import (
    EAGER_OP_TABLE,
    Graph,
    GraphModule,
    Node,
    interpret,
    normalize_shape_args,
    register_eager_op,
    resolve,
    target_name,
)

_OP_TABLE = EAGER_OP_TABLE
_interpret = interpret
_normalize_shape_args = normalize_shape_args
_register_op = register_eager_op
_resolve = resolve
_target_name = target_name

__all__ = [
    "EAGER_OP_TABLE",
    "Graph",
    "GraphModule",
    "Node",
    "interpret",
    "normalize_shape_args",
    "register_eager_op",
    "resolve",
    "target_name",
    "_OP_TABLE",
    "_interpret",
    "_normalize_shape_args",
    "_register_op",
    "_resolve",
    "_target_name",
]
