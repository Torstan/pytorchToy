"""
最小 TorchDynamo 入口。

这一阶段还没有真正的 frame-eval/bytecode symbolic execution。
先把公共 API 和 backend 接口收敛到 torch._dynamo 上，
内部暂时仍复用现有 proxy tracing 路径。
"""

from torch._compile.tracer import Tracer, UnsupportedTraceError
from torch._logging import get_log_settings

from .backends.registry import lookup_backend

_ACTIVE_WRAPPERS = []


def _value_signature(value):
    from torch.tensor import Tensor

    if isinstance(value, Tensor):
        dtype = getattr(getattr(value, "_dtype", None), "name", repr(getattr(value, "_dtype", None)))
        return (
            "tensor",
            tuple(value.shape),
            dtype,
            value.requires_grad,
            value.is_contiguous(),
        )
    if isinstance(value, (int, float, str, bool, type(None))):
        return (type(value).__name__, value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_value_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_value_signature(item) for item in value))
    if isinstance(value, dict):
        items = tuple((key, _value_signature(item)) for key, item in sorted(value.items()))
        return ("dict", items)
    return (type(value).__name__, id(value))


def _call_signature(args, kwargs):
    return (
        tuple(_value_signature(arg) for arg in args),
        tuple((key, _value_signature(value)) for key, value in sorted(kwargs.items())),
    )


class OptimizedFunction:
    """
    最小 Dynamo 包装对象。

    当前仍然按调用签名缓存，但公共入口已经切换到 torch._dynamo.optimize。
    """

    def __init__(self, fn, backend="inductor", *, nopython=False, dynamic=None):
        self._original_fn = fn
        self._backend_name = backend
        self._backend = lookup_backend(backend)
        self._nopython = nopython
        self._dynamic = dynamic
        self._cache = {}
        _ACTIVE_WRAPPERS.append(self)

    def __call__(self, *args, **kwargs):
        key = _call_signature(args, kwargs)
        compiled = self._cache.get(key)
        if compiled is None:
            compiled = self._compile_for_signature(*args, **kwargs)
            self._cache[key] = compiled
        return compiled(*args, **kwargs)

    def _compile_for_signature(self, *args, **kwargs):
        tracer = Tracer()
        try:
            graph_module = tracer.trace(self._original_fn, args)
        except UnsupportedTraceError:
            if self._nopython:
                raise
            return self._original_fn

        self._log_graph(graph_module)
        return self._backend(graph_module, list(args))

    def _log_graph(self, graph_module):
        settings = get_log_settings()
        if settings.get("graph_code", False):
            print("=== GRAPH CODE ===")
            graph_module.print_readable()
            print("==================")

    def reset(self):
        self._cache.clear()


class OptimizeContext:
    def __init__(self, backend="inductor", *, nopython=False, dynamic=None, disable=False):
        self.backend = backend
        self.nopython = nopython
        self.dynamic = dynamic
        self.disable_compile = disable

    def __call__(self, fn):
        if self.disable_compile:
            return fn
        return OptimizedFunction(
            fn,
            backend=self.backend,
            nopython=self.nopython,
            dynamic=self.dynamic,
        )


def optimize(
    backend="inductor",
    *,
    nopython=False,
    dynamic=None,
    disable=False,
):
    """
    最小 torch._dynamo.optimize 入口。
    """

    return OptimizeContext(
        backend=backend,
        nopython=nopython,
        dynamic=dynamic,
        disable=disable,
    )


def disable(fn=None, *, reason=None):
    del reason
    if fn is not None:
        return fn
    return lambda inner: inner


def reset():
    for wrapper in list(_ACTIVE_WRAPPERS):
        wrapper.reset()
