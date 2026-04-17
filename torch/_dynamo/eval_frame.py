"""
最小 TorchDynamo 入口。

这一阶段还没有真正的 frame-eval/bytecode symbolic execution。
先把公共 API 和 backend 接口收敛到 torch._dynamo 上，
内部暂时仍复用现有 proxy tracing 路径。
"""

import inspect
from dataclasses import dataclass, field

from torch._compile.tracer import Tracer, UnsupportedTraceError
from torch._logging import get_log_settings

from .backends.registry import lookup_backend
from . import config
from .convert_frame import convert_frame
from .guards import GuardManager, build_guard_manager
from .resume_execution import build_resume_plan

_ACTIVE_WRAPPERS = []
_CODE_CACHE = {}


@dataclass
class GuardedCode:
    guard_manager: GuardManager
    compiled: object


@dataclass
class _CodeCacheEntry:
    compiled_variants: list[GuardedCode] = field(default_factory=list)
    eager_fallback_variants: list[GuardManager] = field(default_factory=list)


def _find_compiled_variant(cache_entry, args, kwargs, fn):
    for variant in cache_entry.compiled_variants:
        if variant.guard_manager.matches(args, kwargs, fn):
            return variant
    return None


def _has_eager_fallback(cache_entry, args, kwargs, fn):
    for guard_manager in cache_entry.eager_fallback_variants:
        if guard_manager.matches(args, kwargs, fn):
            return True
    return False


def _callable_code_key(fn):
    code = getattr(fn, "__code__", None)
    if code is not None:
        return code

    forward = getattr(fn, "forward", None)
    if callable(forward):
        forward_code = getattr(forward, "__code__", None)
        if forward_code is not None:
            if hasattr(fn, "__dict__") and "_modules" in fn.__dict__:
                return forward_code
            return (forward_code, id(fn))

    call = getattr(type(fn), "__call__", None)
    call_code = getattr(call, "__code__", None)
    if call_code is not None:
        return (call_code, id(fn))

    return id(fn)


def _code_cache_key(fn, backend_name, backend, nopython, dynamic):
    backend_key = backend_name if isinstance(backend_name, str) else backend
    return (_callable_code_key(fn), backend_key, bool(nopython), dynamic)


def _callable_signature(fn):
    forward = getattr(fn, "forward", None)
    if callable(forward) and hasattr(fn, "__dict__") and "_modules" in fn.__dict__:
        try:
            return inspect.signature(forward)
        except (TypeError, ValueError):
            return None

    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def _flatten_runtime_inputs(fn, args, kwargs, *, signature=None):
    if not kwargs:
        return tuple(args), None

    signature = signature or _callable_signature(fn)
    if signature is None:
        raise NotImplementedError("torch.compile kwargs are not supported for this callable")

    unsupported = (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    )
    if any(param.kind in unsupported for param in signature.parameters.values()):
        raise NotImplementedError("torch.compile kwargs are not supported for varargs callables")

    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    flat_args = tuple(
        bound.arguments[name]
        for name in signature.parameters
        if name in bound.arguments
    )
    return flat_args, signature


def _rebuild_runtime_call(signature, flat_args):
    call_args = []
    call_kwargs = {}
    for (name, param), value in zip(signature.parameters.items(), flat_args):
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            call_args.append(value)
            continue
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            call_kwargs[name] = value
            continue
        raise NotImplementedError("torch.compile kwargs are not supported for varargs callables")
    return tuple(call_args), call_kwargs


def _wrap_compiled_with_runtime_binding(fn, compiled, signature):
    def compiled_with_binding(*args, **kwargs):
        flat_args, _ = _flatten_runtime_inputs(
            fn,
            args,
            kwargs,
            signature=signature,
        )
        return compiled(*flat_args)

    return compiled_with_binding


class OptimizedFunction:
    """
    最小 Dynamo 包装对象。

    当前基于显式 guard manager 复用 compiled variants，
    但公共入口已经切换到 torch._dynamo.optimize。
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
        guard_manager = build_guard_manager(self._original_fn, args, kwargs)
        cache_entry = _CODE_CACHE.setdefault(self._cache_key, _CodeCacheEntry())
        variant = _find_compiled_variant(cache_entry, args, kwargs, self._original_fn)
        if variant is not None:
            return variant.compiled(*args, **kwargs)
        if _has_eager_fallback(cache_entry, args, kwargs, self._original_fn):
            return self._original_fn(*args, **kwargs)
        if len(cache_entry.compiled_variants) >= config.recompile_limit:
            self._log_recompile(guard_manager, cached_variants=len(cache_entry.compiled_variants))
            cache_entry.eager_fallback_variants.append(guard_manager)
            return self._original_fn(*args, **kwargs)

        if cache_entry.compiled_variants:
            self._log_recompile(guard_manager, cached_variants=len(cache_entry.compiled_variants))

        compiled = self._compile_for_signature(*args, **kwargs)
        cache_entry.compiled_variants.append(
            GuardedCode(
                guard_manager=guard_manager,
                compiled=compiled,
            )
        )
        return compiled(*args, **kwargs)

    def _compile_for_signature(self, *args, **kwargs):
        flat_args, signature = _flatten_runtime_inputs(self._original_fn, args, kwargs)
        try:
            graph_module = convert_frame(self._original_fn, flat_args)
        except UnsupportedTraceError:
            trace_fn = self._original_fn
            if signature is not None:
                def trace_fn(*runtime_args):
                    call_args, call_kwargs = _rebuild_runtime_call(signature, runtime_args)
                    return self._original_fn(*call_args, **call_kwargs)

            tracer = Tracer()
            try:
                graph_module = tracer.trace(trace_fn, flat_args)
            except (UnsupportedTraceError, TypeError):
                if self._nopython:
                    raise
                resume_plan = build_resume_plan(
                    self._original_fn,
                    self._backend,
                    args,
                    kwargs,
                )
                if resume_plan is not None:
                    return resume_plan
                return self._original_fn

        self._log_guards(build_guard_manager(self._original_fn, args, kwargs))
        self._log_graph(graph_module)
        compiled = self._backend(graph_module, list(flat_args))
        if signature is None:
            return compiled
        return _wrap_compiled_with_runtime_binding(self._original_fn, compiled, signature)

    def _log_graph(self, graph_module):
        settings = get_log_settings()
        if settings.get("graph_code", False):
            print("=== GRAPH CODE ===")
            graph_module.print_readable()
            print("==================")

    def _log_guards(self, guard_manager):
        settings = get_log_settings()
        if not settings.get("guards", False):
            return
        print("=== GUARDS ===")
        for line in guard_manager.describe():
            print(line)
        print("==============")

    def _log_recompile(self, guard_manager, *, cached_variants):
        settings = get_log_settings()
        if not settings.get("recompiles", False):
            return
        print("=== RECOMPILE ===")
        print(f"cached_variants={cached_variants}")
        for line in guard_manager.describe():
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
