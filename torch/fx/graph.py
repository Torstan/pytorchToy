"""
FX Graph IR -- torch.compile 的核心中间表示

对应 PyTorch: torch.fx.Graph + torch.fx.Node + torch.fx.GraphModule

Graph 由 Node 组成，每个 Node 代表一个操作：
  - placeholder: 图的输入参数
  - call_function: 调用一个算子 (如 "sin", "add")
  - output: 图的输出
"""

from torch._compile.ops import (
    EAGER_OP_TABLE,
    normalize_shape_args,
    register_eager_op,
    run_eager_target,
    target_name,
)

_OP_TABLE = EAGER_OP_TABLE
_normalize_shape_args = normalize_shape_args
_register_op = register_eager_op
_target_name = target_name


class Node:
    """
    FX 图中的一个节点，对应 PyTorch torch.fx.Node

    每个 Node 记录一个操作及其输入/输出关系，
    多个 Node 通过 args 中的引用关系形成 DAG。
    """

    def __init__(self, op, target, args=(), kwargs=None, name="", meta=None):
        self.op = op            # "placeholder" | "call_function" | "output"
        self.target = target    # 算子名称字符串或 callable
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.name = name
        self.meta = dict(meta or {})

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
        self._name_counters = {}

    def _fresh_name(self, base):
        """生成唯一名称"""
        index = self._name_counters.get(base, 0)
        self._name_counters[base] = index + 1
        return base if index == 0 else f"{base}_{index}"

    def placeholder(self, name):
        node = Node("placeholder", name, name=self._fresh_name(name))
        self.nodes.append(node)
        return node

    def call_function(self, target, args=(), kwargs=None):
        node = Node(
            "call_function", target,
            args=tuple(args),
            kwargs=dict(kwargs or {}),
            name=self._fresh_name(target_name(target)),
        )
        self.nodes.append(node)
        return node

    def output(self, value):
        node = Node("output", "output", args=(value,), name="output")
        self.nodes.append(node)
        return node

    def format_code(self):
        """生成可读的 Python 代码表示（用于调试/日志）"""
        placeholders = [n.target for n in self.nodes if n.op == "placeholder"]
        lines = [f"def compiled_graph({', '.join(placeholders)}):"]
        for node in self.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "call_function":
                lines.append(f"    {node.name} = {_format_call(node)}")
            if node.op == "output":
                lines.append(f"    return {_format_value(node.args[0])}")
        return "\n".join(lines)


class GraphModule:
    """
    可执行的图模块，对应 PyTorch torch.fx.GraphModule

    持有一个 Graph，可以解释执行。
    backend compiler 接收 GraphModule 并返回优化后的 callable。
    """

    def __init__(self, graph):
        self.graph = graph

    def forward(self, *args):
        return interpret(self.graph, args)

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def propagate_meta(self, example_inputs):
        from torch.fx.meta import propagate_meta

        return propagate_meta(self, example_inputs)

    def print_readable(self, print_output=True, **_unused):
        code = self.graph.format_code()
        if print_output:
            print(code)
        return code


# ---- 解释执行 ----

def interpret(graph, args):
    """解释执行 Graph"""
    env = {}
    arg_idx = 0
    for node in graph.nodes:
        if node.op == "placeholder":
            env[node.name] = args[arg_idx]
            arg_idx += 1
        elif node.op == "call_function":
            call_args = tuple(resolve(a, env) for a in node.args)
            call_kwargs = {k: resolve(v, env) for k, v in node.kwargs.items()}
            env[node.name] = run_eager_target(node.target, call_args, call_kwargs)
        elif node.op == "output":
            return resolve(node.args[0], env)
    return None


def resolve(value, env):
    """将 Node 引用解析为实际值"""
    if isinstance(value, Node):
        return env[value.name]
    if isinstance(value, (tuple, list)):
        return type(value)(resolve(v, env) for v in value)
    return value


_interpret = interpret
_resolve = resolve


# ---- 代码生成辅助 ----

# 特殊格式化规则
_FORMAT_RULES = {
    "sin": lambda node: f"torch.sin({_format_value(node.args[0])})",
    "cos": lambda node: f"torch.cos({_format_value(node.args[0])})",
    "exp": lambda node: f"torch.exp({_format_value(node.args[0])})",
    "log": lambda node: f"torch.log({_format_value(node.args[0])})",
    "add": lambda node: f"{_format_value(node.args[0])} + {_format_value(node.args[1])}",
    "sub": lambda node: f"{_format_value(node.args[0])} - {_format_value(node.args[1])}",
    "mul": lambda node: f"{_format_value(node.args[0])} * {_format_value(node.args[1])}",
    "div": lambda node: f"{_format_value(node.args[0])} / {_format_value(node.args[1])}",
    "neg": lambda node: f"-{_format_value(node.args[0])}",
    "gt": lambda node: f"{_format_value(node.args[0])}.gt({_format_value(node.args[1])})",
    "t": lambda node: f"{_format_value(node.args[0])}.t()",
    "mm": lambda node: f"{_format_value(node.args[0])}.mm({_format_value(node.args[1])})",
    "layer_norm": lambda node: (
        f"layer_norm({_format_value(node.args[0])}, {_format_value(node.args[1])}, "
        f"{_format_value(node.args[2])}, eps={_format_value(node.kwargs.get('eps', 1e-5))})"
    ),
    "call_callable": lambda node: _format_callable_call(node),
}


def _format_call(node):
    """格式化 call_function 节点为可读代码"""
    rule = _FORMAT_RULES.get(node.target)
    if rule:
        return rule(node)
    # 通用格式
    args = ", ".join(_format_value(a) for a in node.args)
    kwargs = ", ".join(f"{k}={_format_value(v)}" for k, v in node.kwargs.items())
    joined = ", ".join(part for part in (args, kwargs) if part)
    return f"{target_name(node.target)}({joined})"


def _format_callable_call(node):
    args = ", ".join(_format_value(arg) for arg in node.args[1:])
    kwargs = ", ".join(f"{key}={_format_value(value)}" for key, value in node.kwargs.items())
    joined = ", ".join(part for part in (args, kwargs) if part)
    return f"{_format_value(node.args[0])}({joined})"


def _format_value(value):
    """格式化值为可读字符串"""
    if isinstance(value, Node):
        return value.name
    if isinstance(value, tuple):
        inner = ", ".join(_format_value(item) for item in value)
        if len(value) == 1:
            inner += ","
        return f"({inner})"
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    return repr(value)
