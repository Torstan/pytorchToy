"""
最小 TorchDynamo 入口。

这一阶段还没有真正的 frame-eval/bytecode symbolic execution。
先把公共 API 和 backend 接口收敛到 torch._dynamo 上，
内部暂时仍复用现有 proxy tracing 路径。
"""

from dataclasses import dataclass, field

from torch._compile.tracer import Tracer, UnsupportedTraceError
from torch._logging import get_log_settings

from .backends.registry import lookup_backend
from . import config
from .guards import callable_guard_signature, describe_callable_guards

_ACTIVE_WRAPPERS = []
_CODE_CACHE = {}


@dataclass
class _CodeCacheEntry:
    compiled_by_signature: dict = field(default_factory=dict)
    eager_fallback_signatures: set = field(default_factory=set)


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


def _describe_value(value):
    from torch.tensor import Tensor

    if isinstance(value, Tensor):
        dtype = getattr(getattr(value, "_dtype", None), "name", repr(getattr(value, "_dtype", None)))
        return (
            f"tensor shape={tuple(value.shape)} dtype={dtype} "
            f"requires_grad={value.requires_grad} contiguous={value.is_contiguous()}"
        )
    if isinstance(value, (int, float, str, bool, type(None))):
        return f"{type(value).__name__} value={value!r}"
    if isinstance(value, tuple):
        return f"tuple len={len(value)}"
    if isinstance(value, list):
        return f"list len={len(value)}"
    if isinstance(value, dict):
        return f"dict keys={sorted(value.keys())}"
    return type(value).__name__


def _log_value_descriptions(args, kwargs):
    lines = []
    for index, value in enumerate(args):
        lines.append(f"arg{index}: {_describe_value(value)}")
    for key, value in sorted(kwargs.items()):
        lines.append(f"kwarg[{key!r}]: {_describe_value(value)}")
    return lines


def _callable_code_key(fn):
    code = getattr(fn, "__code__", None)
    if code is not None:
        return code

    forward = getattr(fn, "forward", None)
    if callable(forward):
        forward_code = getattr(forward, "__code__", None)
        if forward_code is not None:
            return (forward_code, id(fn))

    call = getattr(type(fn), "__call__", None)
    call_code = getattr(call, "__code__", None)
    if call_code is not None:
        return (call_code, id(fn))

    return id(fn)


def _code_cache_key(fn, backend_name, backend, nopython, dynamic):
    backend_key = backend_name if isinstance(backend_name, str) else backend
    return (_callable_code_key(fn), backend_key, bool(nopython), dynamic)


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
        self._cache_key = _code_cache_key(
            fn,
            backend,
            self._backend,
            nopython,
            dynamic,
        )
        _ACTIVE_WRAPPERS.append(self)

    def __call__(self, *args, **kwargs):
        key = (
            _call_signature(args, kwargs),
            callable_guard_signature(self._original_fn),
        )
        cache_entry = _CODE_CACHE.setdefault(self._cache_key, _CodeCacheEntry())
        compiled = cache_entry.compiled_by_signature.get(key)
        if compiled is not None:
            return compiled(*args, **kwargs)
        if key in cache_entry.eager_fallback_signatures:
            return self._original_fn(*args, **kwargs)
        if len(cache_entry.compiled_by_signature) >= config.recompile_limit:
            self._log_recompile(args, kwargs, cached_variants=len(cache_entry.compiled_by_signature))
            cache_entry.eager_fallback_signatures.add(key)
            return self._original_fn(*args, **kwargs)

        if cache_entry.compiled_by_signature:
            self._log_recompile(args, kwargs, cached_variants=len(cache_entry.compiled_by_signature))

        compiled = self._compile_for_signature(*args, **kwargs)
        cache_entry.compiled_by_signature[key] = compiled
        return compiled(*args, **kwargs)

    def _compile_for_signature(self, *args, **kwargs):
        tracer = Tracer()
        try:
            graph_module = tracer.trace(self._original_fn, args)
        except UnsupportedTraceError:
            if self._nopython:
                raise
            return self._original_fn

        self._log_guards(args, kwargs)
        self._log_graph(graph_module)
        return self._backend(graph_module, list(args))

    def _log_graph(self, graph_module):
        settings = get_log_settings()
        if settings.get("graph_code", False):
            print("=== GRAPH CODE ===")
            graph_module.print_readable()
            print("==================")

    def _log_guards(self, args, kwargs):
        settings = get_log_settings()
        if not settings.get("guards", False):
            return
        print("=== GUARDS ===")
        for line in _log_value_descriptions(args, kwargs):
            print(line)
        for line in describe_callable_guards(self._original_fn):
            print(line)
        print("==============")

    def _log_recompile(self, args, kwargs, *, cached_variants):
        settings = get_log_settings()
        if not settings.get("recompiles", False):
            return
        print("=== RECOMPILE ===")
        print(f"cached_variants={cached_variants}")
        for line in _log_value_descriptions(args, kwargs):
            print(line)
        print("=================")

    def reset(self):
        _CODE_CACHE.pop(self._cache_key, None)


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
    _CODE_CACHE.clear()
    for wrapper in list(_ACTIVE_WRAPPERS):
        wrapper.reset()
