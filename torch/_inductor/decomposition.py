"""
mini-Inductor 使用的最小 decomposition 选择与图改写。
"""

from torch._compile.graph import Graph, GraphModule, Node


def _target_name(target):
    if isinstance(target, str):
        return target
    if hasattr(target, "__name__"):
        return target.__name__
    return repr(target)


def select_decomp_table():
    import torch._prims as prims
    import torch._refs as refs

    return {
        "sin": prims.sin,
        "cos": prims.cos,
        "relu": refs.relu,
        "tanh": prims.tanh,
        "neg": prims.neg,
        "add": prims.add,
        "sub": prims.sub,
        "mul": prims.mul,
        "div": prims.div,
        "sum": prims.sum,
        "view": prims.view,
        "reshape": prims.reshape,
        "mm": prims.mm,
        "addmm": refs.addmm,
        "layer_norm": refs.layer_norm,
    }


def decompose_graph_module(graph_module, decomposition_table=None):
    decomposition_table = decomposition_table or select_decomp_table()
    graph = Graph()
    mapping = {}

    def rewrite(value):
        if isinstance(value, Node):
            return mapping[value]
        if isinstance(value, tuple):
            return tuple(rewrite(item) for item in value)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            new_node = graph.placeholder(node.target)
            mapping[node] = new_node
            continue
        if node.op == "call_function":
            target = decomposition_table.get(_target_name(node.target), node.target)
            new_node = graph.call_function(
                target,
                args=rewrite(node.args),
                kwargs=rewrite(node.kwargs),
            )
            mapping[node] = new_node
            continue
        if node.op == "output":
            graph.output(rewrite(node.args[0]))
            continue
        raise RuntimeError(f"unsupported graph node in decomposition pass: {node.op}")

    return GraphModule(graph)
