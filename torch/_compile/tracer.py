"""
Proxy-based Tracer -- 通过代理对象拦截操作，构建 FX Graph

对应 PyTorch: torch.fx.Tracer + torch.fx.Proxy
简化了 TorchDynamo 的字节码符号执行为 proxy tracing。

工作原理:
1. 为每个函数输入创建 Proxy 对象
2. Proxy 重载所有运算符，每次操作记录到 Graph 中
3. torch.sin 等函数通过 duck typing 调用 Proxy.sin()，自动走 tracing 路径
4. 最终收集输出，构建完整的 GraphModule
"""

import threading

from torch._compile.graph import Graph, GraphModule, Node


class UnsupportedTraceError(RuntimeError):
    """
    tracing 遇到不支持的操作时抛出

    对应 PyTorch torch._dynamo.exc.Unsupported。
    在 fullgraph=False 时会被捕获并 fallback 到 eager；
    在 fullgraph=True 时会传播给用户。
    """
    pass


# ---- Tracing 状态管理 (线程安全) ----

_TRACE_STATE = threading.local()


def current_tracer():
    """获取当前线程的 tracer"""
    return getattr(_TRACE_STATE, "tracer", None)


def is_tracing():
    """当前是否处于 tracing 模式"""
    return current_tracer() is not None


class _TraceContext:
    """tracing 上下文管理器，进入/退出 tracing 模式"""

    def __init__(self, tracer):
        self.tracer = tracer
        self.previous = None

    def __enter__(self):
        self.previous = current_tracer()
        _TRACE_STATE.tracer = self.tracer
        return self.tracer

    def __exit__(self, exc_type, exc, tb):
        _TRACE_STATE.tracer = self.previous
        return False


class Proxy:
    """
    代理对象，对应 PyTorch torch.fx.Proxy

    包装一个 Graph Node，拦截所有操作并记录到 Graph。
    当对 Proxy 做运算时，产生新的 Proxy (新的 Node)。

    显式列举支持的方法 (sin, cos 等)，
    同时通过 __getattr__ 泛化处理其他方法调用。
    """

    def __init__(self, node, tracer):
        object.__setattr__(self, '_node', node)
        object.__setattr__(self, '_tracer', tracer)

    @property
    def node(self):
        return object.__getattribute__(self, '_node')

    @property
    def tracer(self):
        return object.__getattribute__(self, '_tracer')

    def __repr__(self):
        return f"Proxy({self.node})"

    # ---- 显式方法 (确保 duck typing 的 hasattr 检查通过) ----

    def sin(self):
        return self.tracer.create_proxy("sin", (self,))

    def cos(self):
        return self.tracer.create_proxy("cos", (self,))

    def relu(self):
        return self.tracer.create_proxy("relu", (self,))

    def tanh(self):
        return self.tracer.create_proxy("tanh", (self,))

    def sum(self, dim=None, keepdim=False):
        return self.tracer.create_proxy("sum", (self,),
                                        {"dim": dim, "keepdim": keepdim})

    # ---- 算术运算符 ----

    def __add__(self, other):
        return self.tracer.create_proxy("add", (self, other))

    def __radd__(self, other):
        return self.tracer.create_proxy("add", (other, self))

    def __sub__(self, other):
        return self.tracer.create_proxy("sub", (self, other))

    def __rsub__(self, other):
        return self.tracer.create_proxy("sub", (other, self))

    def __mul__(self, other):
        return self.tracer.create_proxy("mul", (self, other))

    def __rmul__(self, other):
        return self.tracer.create_proxy("mul", (other, self))

    def __truediv__(self, other):
        return self.tracer.create_proxy("div", (self, other))

    def __neg__(self):
        return self.tracer.create_proxy("neg", (self,))

    # ---- Graph Break 触发器 (data-dependent 操作不可 trace) ----

    def __bool__(self):
        raise UnsupportedTraceError(
            "data-dependent control flow is not supported: "
            "cannot convert Proxy to bool during tracing"
        )

    def __len__(self):
        raise UnsupportedTraceError(
            "len() on Proxy is not supported during tracing"
        )

    def item(self):
        raise UnsupportedTraceError(
            "item() on Proxy is not supported during tracing: "
            "data-dependent value access"
        )

    # ---- 泛化方法拦截 (fallback) ----

    def __getattr__(self, name):
        def method_proxy(*args, **kwargs):
            return self.tracer.create_proxy(name, (self, *args), kwargs)
        return method_proxy


class Tracer:
    """
    图追踪器，对应 PyTorch torch.fx.Tracer / TorchDynamo

    通过 proxy-based tracing 把一个 Python 函数转换为 GraphModule:
    1. 为每个输入创建 Proxy
    2. 用 Proxy 执行原函数 (操作被拦截记录)
    3. 收集输出，构建 GraphModule
    """

    def __init__(self):
        self.graph = Graph()

    def trace(self, fn, example_inputs):
        """
        追踪函数 fn，返回 GraphModule

        Args:
            fn: 要追踪的函数
            example_inputs: 示例输入 (用于确定输入数量和命名)

        Raises:
            UnsupportedTraceError: 遇到不可 trace 的操作时抛出
        """
        # 获取参数名
        try:
            sig = inspect.signature(fn)
            param_names = list(sig.parameters.keys())
        except (ValueError, TypeError):
            param_names = [f"arg_{i}" for i in range(len(example_inputs))]

        # 为每个输入创建 placeholder + Proxy
        proxies = []
        for i, inp in enumerate(example_inputs):
            name = param_names[i] if i < len(param_names) else f"arg_{i}"
            node = self.graph.placeholder(name)
            proxies.append(Proxy(node, self))

        # 在 tracing 上下文中执行函数
        with _TraceContext(self):
            result = fn(*proxies)

        # 收集输出
        if isinstance(result, Proxy):
            self.graph.output(result.node)
        elif isinstance(result, (tuple, list)):
            output_nodes = tuple(
                o.node if isinstance(o, Proxy) else o for o in result
            )
            self.graph.output(output_nodes)
        else:
            self.graph.output(result)

        return GraphModule(self.graph)

    def create_proxy(self, target, args, kwargs=None):
        """创建新的 Proxy 节点"""
        node = self.graph.call_function(
            target,
            self._unwrap(args),
            self._unwrap(kwargs) if kwargs else None,
        )
        return Proxy(node, self)

    def _unwrap(self, value):
        """将 Proxy 解包为 Node 引用"""
        if isinstance(value, Proxy):
            return value.node
        if isinstance(value, tuple):
            return tuple(self._unwrap(item) for item in value)
        if isinstance(value, list):
            return [self._unwrap(item) for item in value]
        if isinstance(value, dict):
            return {key: self._unwrap(item) for key, item in value.items()}
        return value


import inspect
