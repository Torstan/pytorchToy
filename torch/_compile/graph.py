"""
FX Graph IR -- torch.compile 的核心中间表示

对应 PyTorch: torch.fx.Graph + torch.fx.Node + torch.fx.GraphModule

Graph 由 Node 组成，每个 Node 代表一个操作：
  - placeholder: 图的输入参数
  - call_function: 调用一个函数 (如 torch.sin)
  - call_method: 调用一个方法 (如 tensor.__add__)
  - output: 图的输出
"""


class Node:
    """
    FX 图中的一个节点，对应 PyTorch torch.fx.Node

    每个 Node 记录一个操作及其输入/输出关系，
    多个 Node 通过 args 中的引用关系形成 DAG。
    """

    def __init__(self, graph, name, op, target, args=(), kwargs=None):
        self.graph = graph
        self.name = name
        self.op = op            # "placeholder" | "call_function" | "call_method" | "output"
        self.target = target    # 函数引用 or 方法名字符串
        self.args = args        # 输入 (Node 引用或常量)
        self.kwargs = kwargs or {}

    def __repr__(self):
        return f"%{self.name}"


class Graph:
    """
    FX 图，对应 PyTorch torch.fx.Graph

    维护一个有序的 Node 列表，描述计算过程。
    Graph 从 placeholder 节点(输入)开始，到 output 节点(输出)结束。
    """

    def __init__(self):
        self.nodes = []
        self._name_counter = {}

    def _unique_name(self, base):
        if base not in self._name_counter:
            self._name_counter[base] = 0
            return base
        self._name_counter[base] += 1
        return f"{base}_{self._name_counter[base]}"

    def placeholder(self, name):
        node = Node(self, self._unique_name(name), "placeholder", name)
        self.nodes.append(node)
        return node

    def call_function(self, target, args=(), kwargs=None):
        name = getattr(target, '__name__', str(target))
        node = Node(self, self._unique_name(name), "call_function", target, args, kwargs)
        self.nodes.append(node)
        return node

    def call_method(self, method_name, args=(), kwargs=None):
        node = Node(self, self._unique_name(method_name), "call_method", method_name, args, kwargs)
        self.nodes.append(node)
        return node

    def output(self, value):
        node = Node(self, "output", "output", "output", (value,))
        self.nodes.append(node)
        return node

    def python_code(self):
        """生成可读的 Python 代码表示（用于调试/日志）"""
        lines = []
        lines.append("def forward(self, {}):\n".format(
            ", ".join(n.name for n in self.nodes if n.op == "placeholder")
        ))
        for node in self.nodes:
            if node.op == "placeholder":
                continue
            elif node.op == "call_function":
                target_name = _get_target_name(node.target)
                args_str = ", ".join(_format_arg(a) for a in node.args)
                if node.kwargs:
                    kwargs_str = ", ".join(f"{k}={_format_arg(v)}" for k, v in node.kwargs.items())
                    args_str = f"{args_str}, {kwargs_str}" if args_str else kwargs_str
                lines.append(f"    {node.name} = {target_name}({args_str})\n")
            elif node.op == "call_method":
                self_arg = node.args[0] if node.args else None
                rest_args = node.args[1:] if len(node.args) > 1 else ()
                args_str = ", ".join(_format_arg(a) for a in rest_args)
                lines.append(f"    {node.name} = {_format_arg(self_arg)}.{node.target}({args_str})\n")
            elif node.op == "output":
                lines.append(f"    return {_format_arg(node.args[0])}\n")
        return "".join(lines)


class GraphModule:
    """
    可执行的图模块，对应 PyTorch torch.fx.GraphModule

    持有一个 Graph，可以解释执行。
    backend compiler 接收 GraphModule 并返回优化后的 callable。
    """

    def __init__(self, graph):
        self.graph = graph

    def forward(self, *args):
        return _interpret(self.graph, args)

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def print_readable(self):
        code = self.graph.python_code()
        print(code)
        return code


def _interpret(graph, args):
    """解释执行 Graph"""
    env = {}
    arg_idx = 0
    for node in graph.nodes:
        if node.op == "placeholder":
            env[node.name] = args[arg_idx]
            arg_idx += 1
        elif node.op == "call_function":
            fn_args = tuple(_resolve(a, env) for a in node.args)
            fn_kwargs = {k: _resolve(v, env) for k, v in node.kwargs.items()}
            env[node.name] = node.target(*fn_args, **fn_kwargs)
        elif node.op == "call_method":
            resolved_args = tuple(_resolve(a, env) for a in node.args)
            self_obj = resolved_args[0]
            method = getattr(self_obj, node.target)
            env[node.name] = method(*resolved_args[1:])
        elif node.op == "output":
            return _resolve(node.args[0], env)
    return None


def _resolve(value, env):
    """将 Node 引用解析为实际值"""
    if isinstance(value, Node):
        return env[value.name]
    if isinstance(value, (tuple, list)):
        return type(value)(_resolve(v, env) for v in value)
    return value


def _format_arg(arg):
    """格式化参数用于代码生成"""
    if isinstance(arg, Node):
        return arg.name
    return repr(arg)


def _get_target_name(target):
    """获取函数的可读名称"""
    if hasattr(target, '__module__') and hasattr(target, '__name__'):
        module = target.__module__ or ''
        return f"{module}.{target.__name__}" if module else target.__name__
    return str(target)
