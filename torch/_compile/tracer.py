"""
Proxy-based Tracer -- 通过代理对象拦截操作，构建 FX Graph

对应 PyTorch: torch.fx.Tracer + torch.fx.Proxy
简化了 TorchDynamo 的字节码符号执行为 proxy tracing。

工作原理:
1. 为每个函数输入创建 Proxy 对象
2. Proxy 重载所有运算符，每次操作记录到 Graph 中
3. torch.sin 等函数检查参数是否为 Proxy，如果是则走 tracing 路径
4. 最终收集输出，构建完整的 GraphModule
"""

from torch._compile.graph import Graph, GraphModule, Node

# 全局 tracing 状态
_current_tracer = None


def is_tracing():
    """当前是否处于 tracing 模式"""
    return _current_tracer is not None


def get_current_tracer():
    return _current_tracer


class Proxy:
    """
    代理对象，对应 PyTorch torch.fx.Proxy

    包装一个 Graph Node，拦截所有操作并记录到 Graph。
    当对 Proxy 做运算时，产生新的 Proxy (新的 Node)。
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

    # ---- 算术运算符 ----

    def __add__(self, other):
        return self._binary_op('__add__', other)

    def __radd__(self, other):
        return self._binary_op('__radd__', other)

    def __sub__(self, other):
        return self._binary_op('__sub__', other)

    def __rsub__(self, other):
        return self._binary_op('__rsub__', other)

    def __mul__(self, other):
        return self._binary_op('__mul__', other)

    def __rmul__(self, other):
        return self._binary_op('__rmul__', other)

    def __truediv__(self, other):
        return self._binary_op('__truediv__', other)

    def __neg__(self):
        return self._method_call('__neg__')

    # ---- 方法调用 ----

    def __getattr__(self, name):
        # 允许在 Proxy 上调用方法 (如 .sin(), .cos(), .relu() 等)
        def method_proxy(*args, **kwargs):
            return self._method_call(name, *args, **kwargs)
        return method_proxy

    def _binary_op(self, method_name, other):
        """记录二元运算"""
        other_arg = other.node if isinstance(other, Proxy) else other
        node = self.tracer.graph.call_method(
            method_name,
            args=(self.node, other_arg),
        )
        return Proxy(node, self.tracer)

    def _method_call(self, method_name, *args, **kwargs):
        """记录方法调用"""
        processed_args = tuple(
            a.node if isinstance(a, Proxy) else a for a in args
        )
        processed_kwargs = {
            k: v.node if isinstance(v, Proxy) else v
            for k, v in kwargs.items()
        }
        node = self.tracer.graph.call_method(
            method_name,
            args=(self.node, *processed_args),
            kwargs=processed_kwargs if processed_kwargs else None,
        )
        return Proxy(node, self.tracer)


def create_proxy_for_function(tracer, target, args, kwargs=None):
    """
    为函数调用创建 Proxy 节点 (用于 torch.sin 等模块级函数)

    当 torch.sin(proxy) 被调用时，sin 函数检测到参数是 Proxy，
    调用此函数来记录 call_function 节点。
    """
    processed_args = tuple(
        a.node if isinstance(a, Proxy) else a for a in args
    )
    processed_kwargs = {
        k: v.node if isinstance(v, Proxy) else v
        for k, v in (kwargs or {}).items()
    }
    node = tracer.graph.call_function(
        target,
        args=processed_args,
        kwargs=processed_kwargs if processed_kwargs else None,
    )
    return Proxy(node, tracer)


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
        """
        global _current_tracer
        old_tracer = _current_tracer
        _current_tracer = self

        try:
            # 1. 为每个输入创建 placeholder + Proxy
            import inspect
            try:
                sig = inspect.signature(fn)
                param_names = list(sig.parameters.keys())
            except (ValueError, TypeError):
                param_names = [f"arg_{i}" for i in range(len(example_inputs))]

            proxies = []
            for i, inp in enumerate(example_inputs):
                name = param_names[i] if i < len(param_names) else f"arg_{i}"
                node = self.graph.placeholder(name)
                proxies.append(Proxy(node, self))

            # 2. 用 Proxy 执行函数
            output = fn(*proxies)

            # 3. 收集输出
            if isinstance(output, Proxy):
                self.graph.output(output.node)
            elif isinstance(output, (tuple, list)):
                output_nodes = tuple(
                    o.node if isinstance(o, Proxy) else o for o in output
                )
                self.graph.output(output_nodes)
            else:
                self.graph.output(output)

            return GraphModule(self.graph)

        finally:
            _current_tracer = old_tracer
