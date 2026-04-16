"""
最小 AOTAutograd 兼容层。

当前实现先建立 2.x 风格的 fw/bw 编译边界：
- inference: 走 inference_compiler / fw_compiler
- training: 额外准备一个最小 backward graph 并调用 bw_compiler

这一版还没有 joint graph capture，也没有真实的 backward graph lowering。
training 正确性依赖于编译后的 forward 仍然保留 eager autograd 语义。
"""

from dataclasses import dataclass

from torch._compile.graph import Graph, GraphModule
from torch._compile.tracer import Tracer


@dataclass
class AOTCompileState:
    graph_module: object
    backward_graph_module: object
    compiled_fw: object
    compiled_bw: object
    requires_grad: bool
    backward_example_inputs: object = None
    backward_is_real: bool = False


def _value_signature(value):
    from torch.tensor import Tensor

    if isinstance(value, Tensor):
        dtype = getattr(getattr(value, "_dtype", None), "name", None)
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


def _clone_structure(value):
    from torch.tensor import Tensor

    if isinstance(value, Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_structure(item) for item in value)
    if isinstance(value, list):
        return [_clone_structure(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_structure(item) for key, item in value.items()}
    return value


def _collect_tensor_versions(value, versions):
    from torch.tensor import Tensor

    if isinstance(value, Tensor):
        versions[id(value)] = value._version
        return
    if isinstance(value, tuple):
        for item in value:
            _collect_tensor_versions(item, versions)
        return
    if isinstance(value, list):
        for item in value:
            _collect_tensor_versions(item, versions)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_tensor_versions(item, versions)


def _any_requires_grad(args, kwargs):
    from torch.tensor import Tensor

    def visit(value):
        if isinstance(value, Tensor):
            return value.requires_grad
        if isinstance(value, tuple):
            return any(visit(item) for item in value)
        if isinstance(value, list):
            return any(visit(item) for item in value)
        if isinstance(value, dict):
            return any(visit(item) for item in value.values())
        return False

    return visit(args) or visit(kwargs)


def _assert_no_input_mutation(fn, args, kwargs):
    cloned_args = tuple(_clone_structure(arg) for arg in args)
    cloned_kwargs = {key: _clone_structure(value) for key, value in kwargs.items()}
    before = {}
    _collect_tensor_versions(cloned_args, before)
    _collect_tensor_versions(cloned_kwargs, before)
    fn(*cloned_args, **cloned_kwargs)
    after = {}
    _collect_tensor_versions(cloned_args, after)
    _collect_tensor_versions(cloned_kwargs, after)
    if before != after:
        raise NotImplementedError(
            "AOTAutograd-mini does not support input mutation yet"
        )


def _build_backward_stub_graph():
    graph = Graph()
    grad_output = graph.placeholder("grad_output")
    graph.output(grad_output)
    return GraphModule(graph)


def _target_name(target):
    if isinstance(target, str):
        return target
    if hasattr(target, "__name__"):
        return target.__name__
    return repr(target)


class _UnsupportedBackwardGraph(RuntimeError):
    pass


def _rebuild_forward_value(bw_graph, cache, value):
    from torch._compile.graph import Node

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
    from torch._compile.graph import Node
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

        target_name = _target_name(node.target)
        args = node.args

        if target_name == "add":
            accumulate_grad(args[0], _grad_to_target(bw_graph, grad_value, node, args[0]))
            accumulate_grad(args[1], _grad_to_target(bw_graph, grad_value, node, args[1]))
            continue

        if target_name == "sub":
            accumulate_grad(args[0], _grad_to_target(bw_graph, grad_value, node, args[0]))
            rhs_grad = bw_graph.call_function("neg", (grad_value,))
            accumulate_grad(args[1], _grad_to_target(bw_graph, rhs_grad, node, args[1]))
            continue

        if target_name == "mul":
            lhs = _rebuild_forward_value(bw_graph, rebuilt, args[0])
            rhs = _rebuild_forward_value(bw_graph, rebuilt, args[1])
            lhs_grad = bw_graph.call_function("mul", (grad_value, rhs))
            rhs_grad = bw_graph.call_function("mul", (grad_value, lhs))
            accumulate_grad(args[0], _grad_to_target(bw_graph, lhs_grad, node, args[0]))
            accumulate_grad(args[1], _grad_to_target(bw_graph, rhs_grad, node, args[1]))
            continue

        if target_name == "div":
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

        if target_name == "neg":
            accumulate_grad(args[0], bw_graph.call_function("neg", (grad_value,)))
            continue

        if target_name == "tanh":
            primal = _rebuild_forward_value(bw_graph, rebuilt, args[0])
            tanh_primal = bw_graph.call_function("tanh", (primal,))
            tanh_sq = bw_graph.call_function("mul", (tanh_primal, tanh_primal))
            slope = bw_graph.call_function("sub", (1.0, tanh_sq))
            accumulate_grad(args[0], bw_graph.call_function("mul", (grad_value, slope)))
            continue

        if target_name == "relu":
            primal = _rebuild_forward_value(bw_graph, rebuilt, args[0])
            mask = bw_graph.call_function("gt", (primal, 0.0))
            accumulate_grad(args[0], bw_graph.call_function("mul", (grad_value, mask)))
            continue

        if target_name == "sin":
            primal = _rebuild_forward_value(bw_graph, rebuilt, args[0])
            slope = bw_graph.call_function("cos", (primal,))
            accumulate_grad(args[0], bw_graph.call_function("mul", (grad_value, slope)))
            continue

        if target_name == "cos":
            primal = _rebuild_forward_value(bw_graph, rebuilt, args[0])
            sine = bw_graph.call_function("sin", (primal,))
            neg_sine = bw_graph.call_function("neg", (sine,))
            accumulate_grad(args[0], bw_graph.call_function("mul", (grad_value, neg_sine)))
            continue

        if target_name == "mm":
            lhs = _rebuild_forward_value(bw_graph, rebuilt, args[0])
            rhs = _rebuild_forward_value(bw_graph, rebuilt, args[1])
            rhs_t = bw_graph.call_function("t", (rhs,))
            lhs_t = bw_graph.call_function("t", (lhs,))
            lhs_grad = bw_graph.call_function("mm", (grad_value, rhs_t))
            rhs_grad = bw_graph.call_function("mm", (lhs_t, grad_value))
            accumulate_grad(args[0], lhs_grad)
            accumulate_grad(args[1], rhs_grad)
            continue

        if target_name == "sum":
            expanded = bw_graph.call_function(
                "add",
                (_zeros_like_graph_value(bw_graph, rebuilt, args[0]), grad_value),
            )
            accumulate_grad(args[0], expanded)
            continue

        if target_name == "t":
            accumulate_grad(args[0], bw_graph.call_function("t", (grad_value,)))
            continue

        raise _UnsupportedBackwardGraph(f"unsupported backward target: {target_name}")

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


class AOTFunction:
    def __init__(
        self,
        fn,
        *,
        fw_compiler,
        bw_compiler=None,
        inference_compiler=None,
    ):
        self._fn = fn
        self._fw_compiler = fw_compiler
        self._bw_compiler = bw_compiler or fw_compiler
        self._inference_compiler = inference_compiler or fw_compiler
        self._cache = {}
        self._last_state = None

    def __call__(self, *args, **kwargs):
        key = _call_signature(args, kwargs)
        compiled = self._cache.get(key)
        if compiled is None:
            compiled = self._compile(args, kwargs)
            self._cache[key] = compiled
        return compiled(*args, **kwargs)

    def _compile(self, args, kwargs):
        if kwargs:
            raise NotImplementedError("AOTAutograd-mini does not support kwargs yet")

        _assert_no_input_mutation(self._fn, args, kwargs)

        tracer = Tracer()
        graph_module = tracer.trace(self._fn, args)
        requires_grad = _any_requires_grad(args, kwargs)

        if requires_grad:
            compiled_fw = self._fw_compiler(graph_module, list(args))
            backward_graph_module, backward_example_inputs, backward_is_real = _build_backward_graph_or_stub(
                graph_module,
                list(args),
            )
            compiled_bw = self._bw_compiler(
                backward_graph_module,
                list(backward_example_inputs),
            )
        else:
            compiled_fw = self._inference_compiler(graph_module, list(args))
            backward_graph_module = None
            compiled_bw = None
            backward_example_inputs = None
            backward_is_real = False

        self._last_state = AOTCompileState(
            graph_module=graph_module,
            backward_graph_module=backward_graph_module,
            compiled_fw=compiled_fw,
            compiled_bw=compiled_bw,
            requires_grad=requires_grad,
            backward_example_inputs=backward_example_inputs,
            backward_is_real=backward_is_real,
        )
        return compiled_fw


def aot_function(
    fn,
    *,
    fw_compiler,
    bw_compiler=None,
    inference_compiler=None,
):
    return AOTFunction(
        fn,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        inference_compiler=inference_compiler,
    )


def aot_module_simplified(
    fn,
    example_inputs,
    *,
    fw_compiler,
    bw_compiler=None,
    inference_compiler=None,
):
    from torch.nn.module import Module

    if not isinstance(fn, Module):
        compiled = aot_function(
            fn,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            inference_compiler=inference_compiler,
        )
        compiled(*example_inputs)
        return compiled

    parameter_names = [name for name, _value in fn.named_parameters()]
    buffer_names = [name for name, _value in _named_buffers(fn)]
    lifted_names = parameter_names + buffer_names
    num_lifted = len(lifted_names)

    def lifted_fn(*flat_args):
        lifted_values = flat_args[:num_lifted]
        user_args = flat_args[num_lifted:]
        restore = []
        try:
            for name, value in zip(parameter_names, lifted_values[:len(parameter_names)]):
                restore.append(_swap_module_tensor(fn, name, value, is_buffer=False))
            for name, value in zip(buffer_names, lifted_values[len(parameter_names):]):
                restore.append(_swap_module_tensor(fn, name, value, is_buffer=True))
            return fn(*user_args)
        finally:
            for owner, attr_name, previous, is_buffer in reversed(restore):
                table = owner._buffers if is_buffer else owner._parameters
                table[attr_name] = previous

    compiled = aot_function(
        lifted_fn,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        inference_compiler=inference_compiler,
    )
    module_wrapper = _AOTModuleWrapper(fn, compiled, parameter_names, buffer_names)
    module_wrapper(*example_inputs)
    return module_wrapper


def make_boxed_func(fn):
    return fn


def _named_buffers(module, prefix=""):
    for name, buffer in module._buffers.items():
        if buffer is not None:
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, buffer
    for name, submodule in module._modules.items():
        if submodule is not None:
            sub_prefix = f"{prefix}.{name}" if prefix else name
            yield from _named_buffers(submodule, sub_prefix)


def _resolve_module_owner(module, qualified_name):
    owner = module
    parts = qualified_name.split(".")
    for part in parts[:-1]:
        owner = owner._modules[part]
    return owner, parts[-1]


def _swap_module_tensor(module, qualified_name, value, *, is_buffer):
    owner, name = _resolve_module_owner(module, qualified_name)
    table = owner._buffers if is_buffer else owner._parameters
    previous = table[name]
    table[name] = value
    return owner, name, previous, is_buffer


class _AOTModuleWrapper:
    def __init__(self, module, compiled, parameter_names, buffer_names):
        self._module = module
        self._compiled = compiled
        self._parameter_names = list(parameter_names)
        self._buffer_names = list(buffer_names)

    @property
    def _last_state(self):
        return self._compiled._last_state

    def __call__(self, *args, **kwargs):
        lifted_args = []
        for name in self._parameter_names:
            owner, attr_name = _resolve_module_owner(self._module, name)
            lifted_args.append(owner._parameters[attr_name])
        for name in self._buffer_names:
            owner, attr_name = _resolve_module_owner(self._module, name)
            lifted_args.append(owner._buffers[attr_name])
        return self._compiled(*lifted_args, *args, **kwargs)
