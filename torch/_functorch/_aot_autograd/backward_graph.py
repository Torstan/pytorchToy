"""
Backward graph construction for the toy AOTAutograd path.
"""

from torch._compile.ops import target_name
from torch.fx import Graph, GraphModule

def _build_backward_stub_graph():
    graph = Graph()
    grad_output = graph.placeholder("grad_output")
    graph.output(grad_output)
    return GraphModule(graph)


class _UnsupportedBackwardGraph(RuntimeError):
    pass


def _rebuild_forward_value(bw_graph, cache, value):
    from torch.fx import Node

    if isinstance(value, Node):
        cached = cache.get(value)
        if cached is not None:
            return cached
        if value.op == "placeholder":
            raise RuntimeError(f"missing backward placeholder for {value.target}")
        if value.op != "call_function":
            raise _UnsupportedBackwardGraph(
                f"cannot rebuild forward value for node op {value.op}"
            )
        rebuilt = bw_graph.call_function(
            value.target,
            args=_rebuild_forward_value(bw_graph, cache, value.args),
            kwargs=_rebuild_forward_value(bw_graph, cache, value.kwargs),
        )
        cache[value] = rebuilt
        return rebuilt
    if isinstance(value, tuple):
        return tuple(_rebuild_forward_value(bw_graph, cache, item) for item in value)
    if isinstance(value, list):
        return [_rebuild_forward_value(bw_graph, cache, item) for item in value]
    if isinstance(value, dict):
        return {key: _rebuild_forward_value(bw_graph, cache, item) for key, item in value.items()}
    return value


def _zeros_like_graph_value(bw_graph, cache, value):
    primal = _rebuild_forward_value(bw_graph, cache, value)
    return bw_graph.call_function("mul", (primal, 0.0))


def _shape_of_node(node):
    meta = getattr(node, "meta", None) or {}
    value = meta.get("val")
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(shape)


def _reduce_grad_to_shape(bw_graph, grad_value, source_shape, target_shape):
    if source_shape is None or target_shape is None or tuple(source_shape) == tuple(target_shape):
        return grad_value

    source_shape = tuple(source_shape)
    target_shape = tuple(target_shape)
    if len(target_shape) > len(source_shape):
        raise _UnsupportedBackwardGraph(
            f"cannot reduce grad from shape {source_shape} to larger target {target_shape}"
        )

    current = grad_value
    len_diff = len(source_shape) - len(target_shape)
    padded_target = (1,) * len_diff + target_shape

    for dim in range(len(source_shape) - 1, -1, -1):
        src_dim = source_shape[dim]
        tgt_dim = padded_target[dim]
        if dim < len_diff:
            current = bw_graph.call_function("sum", (current,), {"dim": dim, "keepdim": False})
            continue
        if tgt_dim == 1 and src_dim != 1:
            current = bw_graph.call_function("sum", (current,), {"dim": dim, "keepdim": True})

    if len_diff > 0 and target_shape:
        current = bw_graph.call_function("reshape", (current, *target_shape))

    return current


def _grad_to_target(bw_graph, grad_value, source_node, target_node):
    if source_node is None or target_node is None:
        return grad_value
    return _reduce_grad_to_shape(
        bw_graph,
        grad_value,
        _shape_of_node(source_node),
        _shape_of_node(target_node),
    )


def _build_backward_graph(graph_module, example_inputs):
    from torch.fx import Node
    from torch.tensor import Tensor

    output_example = graph_module(*example_inputs)
    if not hasattr(output_example, "shape"):
        raise _UnsupportedBackwardGraph("backward graph only supports Tensor outputs")

    grad_output_example = output_example * 0.0 + 1.0

    bw_graph = Graph()
    rebuilt = {}
    differentiable_placeholders = []

    nodes = list(graph_module.graph.nodes)
    if not nodes or nodes[-1].op != "output":
        raise _UnsupportedBackwardGraph("forward graph is missing output node")
    output_value = nodes[-1].args[0]
    if isinstance(output_value, (tuple, list, dict)):
        raise _UnsupportedBackwardGraph("backward graph only supports single Tensor outputs")
    if not isinstance(output_value, Node):
        raise _UnsupportedBackwardGraph("backward graph expects Tensor node output")

    def assert_no_grad_constants(value):
        if isinstance(value, Tensor) and value.requires_grad:
            raise _UnsupportedBackwardGraph(
                "backward graph for lifted parameter/buffer constants is not implemented yet"
            )
        if isinstance(value, tuple):
            for item in value:
                assert_no_grad_constants(item)
            return
        if isinstance(value, list):
            for item in value:
                assert_no_grad_constants(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                assert_no_grad_constants(item)

    for node in nodes:
        if node.op != "call_function":
            continue
        assert_no_grad_constants(node.args)
        assert_no_grad_constants(node.kwargs)

    placeholder_index = 0
    for node in nodes:
        if node.op != "placeholder":
            continue
        placeholder = bw_graph.placeholder(node.target)
        rebuilt[node] = placeholder
        example_value = example_inputs[placeholder_index]
        placeholder_index += 1
        if getattr(example_value, "requires_grad", False):
            differentiable_placeholders.append(node)

    grad_output = bw_graph.placeholder("grad_output")
    grad_map = {output_value: grad_output}

    def accumulate_grad(target, grad_value):
        if not isinstance(target, Node):
            return
        existing = grad_map.get(target)
        if existing is None:
            grad_map[target] = grad_value
            return
        grad_map[target] = bw_graph.call_function("add", (existing, grad_value))

    for node in reversed(nodes[:-1]):
        if node.op != "call_function":
            continue
        grad_value = grad_map.get(node)
        if grad_value is None:
            continue

        target = target_name(node.target)
        args = node.args

        if target == "add":
            accumulate_grad(args[0], _grad_to_target(bw_graph, grad_value, node, args[0]))
            accumulate_grad(args[1], _grad_to_target(bw_graph, grad_value, node, args[1]))
            continue

        if target == "sub":
            accumulate_grad(args[0], _grad_to_target(bw_graph, grad_value, node, args[0]))
            rhs_grad = bw_graph.call_function("neg", (grad_value,))
            accumulate_grad(args[1], _grad_to_target(bw_graph, rhs_grad, node, args[1]))
            continue

        if target == "mul":
            lhs = _rebuild_forward_value(bw_graph, rebuilt, args[0])
            rhs = _rebuild_forward_value(bw_graph, rebuilt, args[1])
            lhs_grad = bw_graph.call_function("mul", (grad_value, rhs))
            rhs_grad = bw_graph.call_function("mul", (grad_value, lhs))
            accumulate_grad(args[0], _grad_to_target(bw_graph, lhs_grad, node, args[0]))
            accumulate_grad(args[1], _grad_to_target(bw_graph, rhs_grad, node, args[1]))
            continue

        if target == "div":
            lhs = _rebuild_forward_value(bw_graph, rebuilt, args[0])
            rhs = _rebuild_forward_value(bw_graph, rebuilt, args[1])
            rhs_sq = bw_graph.call_function("mul", (rhs, rhs))
            neg_lhs = bw_graph.call_function("neg", (lhs,))
            rhs_term = bw_graph.call_function("div", (neg_lhs, rhs_sq))
            lhs_grad = bw_graph.call_function("div", (grad_value, rhs))
            rhs_grad = bw_graph.call_function("mul", (grad_value, rhs_term))
            accumulate_grad(args[0], _grad_to_target(bw_graph, lhs_grad, node, args[0]))
            accumulate_grad(args[1], _grad_to_target(bw_graph, rhs_grad, node, args[1]))
            continue

        if target == "neg":
            accumulate_grad(args[0], bw_graph.call_function("neg", (grad_value,)))
            continue

        if target == "tanh":
            primal = _rebuild_forward_value(bw_graph, rebuilt, args[0])
            tanh_primal = bw_graph.call_function("tanh", (primal,))
            tanh_sq = bw_graph.call_function("mul", (tanh_primal, tanh_primal))
            slope = bw_graph.call_function("sub", (1.0, tanh_sq))
            accumulate_grad(args[0], bw_graph.call_function("mul", (grad_value, slope)))
            continue

        if target == "relu":
            primal = _rebuild_forward_value(bw_graph, rebuilt, args[0])
            mask = bw_graph.call_function("gt", (primal, 0.0))
            accumulate_grad(args[0], bw_graph.call_function("mul", (grad_value, mask)))
            continue

        if target == "sin":
            primal = _rebuild_forward_value(bw_graph, rebuilt, args[0])
            slope = bw_graph.call_function("cos", (primal,))
            accumulate_grad(args[0], bw_graph.call_function("mul", (grad_value, slope)))
            continue

        if target == "cos":
            primal = _rebuild_forward_value(bw_graph, rebuilt, args[0])
            sine = bw_graph.call_function("sin", (primal,))
            neg_sine = bw_graph.call_function("neg", (sine,))
            accumulate_grad(args[0], bw_graph.call_function("mul", (grad_value, neg_sine)))
            continue

        if target == "mm":
            lhs = _rebuild_forward_value(bw_graph, rebuilt, args[0])
            rhs = _rebuild_forward_value(bw_graph, rebuilt, args[1])
            rhs_t = bw_graph.call_function("t", (rhs,))
            lhs_t = bw_graph.call_function("t", (lhs,))
            lhs_grad = bw_graph.call_function("mm", (grad_value, rhs_t))
            rhs_grad = bw_graph.call_function("mm", (lhs_t, grad_value))
            accumulate_grad(args[0], lhs_grad)
            accumulate_grad(args[1], rhs_grad)
            continue

        if target == "sum":
            dim = node.kwargs.get("dim", None)
            keepdim = node.kwargs.get("keepdim", False)
            if dim is not None and not keepdim:
                # grad_value has the reduced dim removed; reinsert it before expand
                unsqueezed = bw_graph.call_function("unsqueeze", (grad_value, dim))
                expanded = bw_graph.call_function(
                    "add",
                    (_zeros_like_graph_value(bw_graph, rebuilt, args[0]), unsqueezed),
                )
            else:
                expanded = bw_graph.call_function(
                    "add",
                    (_zeros_like_graph_value(bw_graph, rebuilt, args[0]), grad_value),
                )
            accumulate_grad(args[0], expanded)
            continue

        if target == "t":
            accumulate_grad(args[0], bw_graph.call_function("t", (grad_value,)))
            continue

        raise _UnsupportedBackwardGraph(f"unsupported backward target: {target}")

    outputs = []
    for placeholder in differentiable_placeholders:
        grad_value = grad_map.get(placeholder)
        if grad_value is None:
            grad_value = _zeros_like_graph_value(bw_graph, rebuilt, placeholder)
        outputs.append(grad_value)

    bw_graph.output(tuple(outputs))
    return GraphModule(bw_graph), [*example_inputs, grad_output_example], True


def _build_backward_graph_or_stub(graph_module, example_inputs):
    try:
        return _build_backward_graph(graph_module, example_inputs)
    except _UnsupportedBackwardGraph:
        output_example = graph_module(*example_inputs)
        grad_output_example = output_example * 0.0 + 1.0
        return _build_backward_stub_graph(), [grad_output_example], False



def build_backward_graph_or_stub(graph_module, example_inputs):
    return _build_backward_graph_or_stub(graph_module, example_inputs)
