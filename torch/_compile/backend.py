"""
Backend 编译器 -- 将 GraphModule 编译为可执行 callable

对应 PyTorch: torch._dynamo.backends.registry + torch._inductor.compile_fx
"""


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


def list_backends():
    """返回当前已注册 backend 名称。"""
    return sorted(_BACKENDS.keys())


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
    Inductor backend: 走 mini torch._inductor.compile_fx 总控。
    """
    from torch._inductor.compile_fx import compile_fx

    return compile_fx(graph_module, example_inputs)


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
