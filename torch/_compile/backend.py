"""
Backend 编译器 -- 将 GraphModule 编译为可执行 callable

对应 PyTorch: torch._dynamo.backends.registry + torch._inductor.compile_fx

在教学实现中提供两个 backend:
  - eager_backend: 直接解释执行 GraphModule (无优化，但完整保留 compile 流水线)
  - fuse_backend: 简单的算子融合 (将连续逐元素操作合并)
"""

from torch._compile.graph import GraphModule
from torch._compile.pointwise import (
    PointwiseLoweringError,
    lower_pointwise_graph,
)


# ---- Backend Registry ----

_BACKENDS = {}


def register_backend(name):
    """注册 backend 的装饰器"""
    def decorator(fn):
        _BACKENDS[name] = fn
        return fn
    return decorator


def lookup_backend(backend):
    """按名称查找 backend，或直接返回 callable"""
    if callable(backend) and not isinstance(backend, str):
        return backend
    if isinstance(backend, str):
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. "
                           f"Available: {list(_BACKENDS.keys())}")
        return _BACKENDS[backend]
    raise TypeError(f"backend must be str or callable, got {type(backend)}")


# ---- 内置 Backends ----

@register_backend("eager")
def eager_backend(graph_module, example_inputs):
    """
    Eager backend: 直接解释执行 GraphModule

    不做任何优化，但保留完整的 compile 流水线。
    对应 PyTorch 的 "eager" debug backend。
    """
    def compiled_fn(*args):
        return graph_module(*args)
    return compiled_fn


@register_backend("inductor")
def inductor_backend(graph_module, example_inputs):
    """
    Inductor backend: 优先走 pointwise fused fast path

    第一阶段只覆盖 inference-only 的纯逐元素图。
    不满足条件时回退到 GraphModule 解释执行，保留完整语义。
    """
    try:
        program = lower_pointwise_graph(graph_module, example_inputs)
        compiled_program = program.compile()
    except PointwiseLoweringError:
        def compiled_fn(*args):
            return graph_module(*args)
        return compiled_fn

    def compiled_fn(*args):
        return compiled_program.run(args)
    return compiled_fn


@register_backend("default")
def default_backend(graph_module, example_inputs):
    """
    默认 backend: 与 eager 相同

    在真实 PyTorch 中，默认 backend 是 "inductor"，
    这里简化为直接解释执行。
    """
    def compiled_fn(*args):
        return graph_module(*args)
    return compiled_fn
