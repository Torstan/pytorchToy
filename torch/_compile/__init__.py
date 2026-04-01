"""
torch.compile API -- pytorchToy 的编译入口

对应 PyTorch:
  - torch.compile()             (torch/__init__.py)
  - torch._dynamo.optimize()    (torch/_dynamo/eval_frame.py)
  - OptimizedModule             (torch/_dynamo/eval_frame.py)
  - _TorchDynamoContext         (torch/_dynamo/eval_frame.py)

核心流程:
  torch.compile(fn) -> OptimizedFunction
    -> 每次调用时: 检查 guard cache
      -> cache hit: 直接使用 compiled_fn
      -> cache miss: Tracer.trace(fn) -> GraphModule -> backend(gm) -> compiled_fn
"""

import inspect
from functools import wraps

from torch._compile.tracer import Tracer, UnsupportedTraceError
from torch._compile.backend import lookup_backend


def _value_signature(value):
    """
    提取值的编译签名 (shape, dtype, requires_grad 等)

    对应 PyTorch Guard 系统的简化版:
    PyTorch 用 Guard 检查输入是否满足已编译图的前提条件,
    这里简化为基于签名的 hash key。
    """
    from torch.tensor import Tensor
    if isinstance(value, Tensor):
        dtype = getattr(getattr(value, "_dtype", None), "name",
                        repr(getattr(value, "_dtype", None)))
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
        items = tuple((key, _value_signature(item))
                      for key, item in sorted(value.items()))
        return ("dict", items)
    return (type(value).__name__, id(value))


def _call_signature(args, kwargs):
    """计算一次调用的完整签名，作为 cache key"""
    return (
        tuple(_value_signature(arg) for arg in args),
        tuple((key, _value_signature(value))
              for key, value in sorted(kwargs.items())),
    )


class OptimizedFunction:
    """
    torch.compile 返回的包装对象

    对应 PyTorch 的 OptimizedModule / _TorchDynamoContext.__call__ 返回值。

    带 Guard + Cache 机制:
    - 每次调用时提取输入签名 (shape, dtype, requires_grad)
    - 签名命中缓存时直接复用已编译结果
    - 签名变化时重新 trace + compile (recompilation)
    """

    def __init__(self, fn, backend="default", fullgraph=False, dynamic=None):
        self._original_fn = fn
        self._backend_name = backend
        self._backend = lookup_backend(backend)
        self._fullgraph = fullgraph
        self._dynamic = dynamic
        self._cache = {}

    def __call__(self, *args, **kwargs):
        key = _call_signature(args, kwargs)
        compiled = self._cache.get(key)
        if compiled is None:
            compiled = self._compile_for_signature(*args, **kwargs)
            self._cache[key] = compiled
        return compiled(*args, **kwargs)

    def _compile_for_signature(self, *args, **kwargs):
        """对当前输入签名执行 trace + compile"""
        tracer = Tracer()
        try:
            graph_module = tracer.trace(self._original_fn, args)
        except UnsupportedTraceError:
            if self._fullgraph:
                raise
            # fullgraph=False 时，graph break 后 fallback 到 eager
            return self._original_fn

        self._log_graph(graph_module)
        return self._backend(graph_module, list(args))

    def _log_graph(self, graph_module):
        """如果启用了 graph_code 日志，打印 Graph"""
        from torch._logging import get_log_settings
        settings = get_log_settings()
        if settings.get('graph_code', False):
            print("=== GRAPH CODE ===")
            graph_module.print_readable()
            print("==================")


def compile(model=None, *, fullgraph=False, dynamic=None, backend="default",
            mode=None, options=None, disable=False):
    """
    torch.compile -- 编译优化函数或模块

    对应 PyTorch torch.compile()

    支持三种用法:
      1. opt_fn = torch.compile(fn)         # 直接包装
      2. @torch.compile                     # 装饰器
      3. @torch.compile(backend="eager")    # 带参数的装饰器

    Args:
        model: 要编译的函数或 Module (None 时返回装饰器)
        fullgraph: 是否要求整个函数为单一图
        dynamic: 动态形状 (简化实现中未使用)
        backend: 后端名称或 callable
        mode: 编译模式 (简化实现中未使用)
        options: 后端选项 (简化实现中未使用)
        disable: 是否禁用编译 (True 时返回原函数)
    """
    if disable:
        if model is not None:
            return model
        return lambda fn: fn

    # @torch.compile (无括号装饰器 -- model 就是被装饰的函数)
    if model is not None and callable(model):
        return OptimizedFunction(model, backend=backend, fullgraph=fullgraph,
                                 dynamic=dynamic)

    # @torch.compile(...) (带参数装饰器 -- 返回装饰器工厂)
    def decorator(fn):
        return OptimizedFunction(fn, backend=backend, fullgraph=fullgraph,
                                 dynamic=dynamic)
    return decorator
