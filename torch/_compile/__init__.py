"""
torch.compile API -- pytorchToy 的编译入口

对应 PyTorch:
  - torch.compile()             (torch/__init__.py)
  - torch._dynamo.optimize()    (torch/_dynamo/eval_frame.py)
  - OptimizedModule             (torch/_dynamo/eval_frame.py)
  - _TorchDynamoContext         (torch/_dynamo/eval_frame.py)

核心流程:
  torch.compile(fn) -> OptimizedFunction
    -> 首次调用时: Tracer.trace(fn) -> GraphModule -> backend(gm) -> compiled_fn
    -> 后续调用: 直接使用 compiled_fn (缓存复用)
"""

from torch._compile.tracer import Tracer
from torch._compile.backend import lookup_backend


class OptimizedFunction:
    """
    torch.compile 返回的包装对象

    对应 PyTorch 的 OptimizedModule / _TorchDynamoContext.__call__ 返回值。
    延迟到首次调用时进行 tracing 和编译，之后缓存编译结果。
    """

    def __init__(self, fn, backend="default", fullgraph=False, dynamic=None):
        self._original_fn = fn
        self._backend_name = backend
        self._backend = lookup_backend(backend)
        self._fullgraph = fullgraph
        self._dynamic = dynamic
        self._compiled_fn = None
        self._graph_module = None

    def __call__(self, *args, **kwargs):
        if self._compiled_fn is None:
            # 首次调用: trace + compile
            self._graph_module = self._trace(args)
            self._log_graph()
            self._compiled_fn = self._backend(self._graph_module, list(args))
        return self._compiled_fn(*args, **kwargs)

    def _trace(self, example_inputs):
        """执行 tracing，构建 GraphModule"""
        tracer = Tracer()
        return tracer.trace(self._original_fn, example_inputs)

    def _log_graph(self):
        """如果启用了 graph_code 日志，打印 Graph"""
        from torch._logging import get_log_settings
        settings = get_log_settings()
        if settings.get('graph_code', False):
            print("=== GRAPH CODE ===")
            self._graph_module.print_readable()
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
        fullgraph: 是否要求整个函数为单一图 (简化实现中未强制)
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

    # @torch.compile (无括号装饰器 — model 就是被装饰的函数)
    if model is not None and callable(model):
        return OptimizedFunction(model, backend=backend, fullgraph=fullgraph,
                                 dynamic=dynamic)

    # @torch.compile(...) (带参数装饰器 — 返回装饰器工厂)
    def decorator(fn):
        return OptimizedFunction(fn, backend=backend, fullgraph=fullgraph,
                                 dynamic=dynamic)
    return decorator
