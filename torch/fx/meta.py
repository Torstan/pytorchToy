"""
最小 meta propagation。

这层只负责给 Graph 节点附加 shape/dtype/requires_grad/stride 等元信息，
不参与真实执行。
"""

from torch._compile.ops import broadcast_shapes, target_name
from torch._subclasses.fake_tensor import FakeTensor
from torch.tensor import Tensor


class MetaPropagationError(RuntimeError):
    pass


def _to_fake(value):
    if isinstance(value, FakeTensor):
        return value
    if isinstance(value, Tensor):
        return FakeTensor.from_tensor(value)
    return value


def _resolve(value, env):
    from torch.fx.graph import Node

    if isinstance(value, Node):
        return env[value.name]
    if isinstance(value, tuple):
        return tuple(_resolve(item, env) for item in value)
    if isinstance(value, list):
        return [_resolve(item, env) for item in value]
    if isinstance(value, dict):
        return {key: _resolve(item, env) for key, item in value.items()}
    return value


def _unary_meta(args, kwargs):
    del kwargs
    return args[0].clone()


def _binary_meta(args, kwargs):
    del kwargs
    lhs, rhs = args
    if isinstance(lhs, FakeTensor) and not isinstance(rhs, FakeTensor):
        return lhs.clone()
    if not isinstance(lhs, FakeTensor) and isinstance(rhs, FakeTensor):
        return rhs.clone()
    if not isinstance(lhs, FakeTensor):
        raise MetaPropagationError(
            f"binary meta expects at least one FakeTensor input, got {type(lhs)} and {type(rhs)}"
        )
    shape = broadcast_shapes(lhs.shape, rhs.shape, error_type=MetaPropagationError)
    requires_grad = lhs.requires_grad or rhs.requires_grad
    return FakeTensor(shape, dtype=lhs.dtype, requires_grad=requires_grad)


def _sum_meta(args, kwargs):
    tensor = args[0]
    dim = kwargs.get("dim")
    keepdim = kwargs.get("keepdim", False)
    if dim is None:
        return FakeTensor((), dtype=tensor.dtype, requires_grad=tensor.requires_grad)
    dims = list(tensor.shape)
    if dim < 0:
        dim = len(dims) + dim
    if keepdim:
        dims[dim] = 1
    else:
        dims.pop(dim)
    return FakeTensor(tuple(dims), dtype=tensor.dtype, requires_grad=tensor.requires_grad)


def _view_meta(args, kwargs):
    del kwargs
    tensor = args[0]
    shape = args[1:]
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return FakeTensor(tuple(shape), dtype=tensor.dtype, requires_grad=tensor.requires_grad)


def _mm_meta(args, kwargs):
    del kwargs
    lhs, rhs = args
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise MetaPropagationError("mm expects 2D inputs")
    if lhs.shape[1] != rhs.shape[0]:
        raise MetaPropagationError(
            f"mm shape mismatch: {lhs.shape} x {rhs.shape}"
        )
    return FakeTensor(
        (lhs.shape[0], rhs.shape[1]),
        dtype=lhs.dtype,
        requires_grad=lhs.requires_grad or rhs.requires_grad,
    )


def _t_meta(args, kwargs):
    del kwargs
    tensor = args[0]
    if len(tensor.shape) != 2:
        raise MetaPropagationError("t expects 2D input")
    return FakeTensor(
        (tensor.shape[1], tensor.shape[0]),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


def _addmm_meta(args, kwargs):
    del kwargs
    bias, lhs, rhs = args
    mm_out = _mm_meta((lhs, rhs), {})
    if isinstance(bias, FakeTensor):
        shape = broadcast_shapes(bias.shape, mm_out.shape, error_type=MetaPropagationError)
        return FakeTensor(
            shape,
            dtype=mm_out.dtype,
            requires_grad=bias.requires_grad or mm_out.requires_grad,
        )
    return mm_out


def _shape_tuple(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        raise MetaPropagationError(f"value does not have shape metadata: {type(value)}")
    return tuple(shape)


def _layer_norm_meta(args, kwargs):
    input_value, weight, bias = args
    eps = kwargs.get("eps", 1e-5)
    del eps
    input_shape = _shape_tuple(input_value)
    weight_shape = _shape_tuple(weight)
    bias_shape = _shape_tuple(bias)
    if len(weight_shape) != 1 or len(bias_shape) != 1:
        raise MetaPropagationError("layer_norm expects 1D weight and bias")
    if not input_shape or input_shape[-1] != weight_shape[0] or weight_shape != bias_shape:
        raise MetaPropagationError(
            f"layer_norm shape mismatch: input={input_shape}, weight={weight_shape}, bias={bias_shape}"
        )
    return FakeTensor(
        input_shape,
        dtype=input_value.dtype,
        requires_grad=(
            input_value.requires_grad
            or getattr(weight, "requires_grad", False)
            or getattr(bias, "requires_grad", False)
        ),
    )


def _call_callable_meta(args, kwargs):
    try:
        return args[0](*args[1:], **kwargs)
    except Exception as exc:
        raise MetaPropagationError(
            f"failed to infer meta for callable input {args[0]!r}: {exc}"
        ) from exc


_META_RULES = {
    "sin": _unary_meta,
    "cos": _unary_meta,
    "exp": _unary_meta,
    "log": _unary_meta,
    "relu": _unary_meta,
    "tanh": _unary_meta,
    "neg": _unary_meta,
    "add": _binary_meta,
    "sub": _binary_meta,
    "mul": _binary_meta,
    "div": _binary_meta,
    "sum": _sum_meta,
    "view": _view_meta,
    "reshape": _view_meta,
    "mm": _mm_meta,
    "t": _t_meta,
    "addmm": _addmm_meta,
    "layer_norm": _layer_norm_meta,
    "call_callable": _call_callable_meta,
}


def infer_meta(target, args, kwargs):
    name = target_name(target)
    rule = _META_RULES.get(name)
    if rule is None:
        raise MetaPropagationError(f"no meta rule registered for {name}")
    return rule(args, kwargs)


def propagate_meta(graph_module, example_inputs):
    env = {}
    arg_index = 0
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            value = _to_fake(example_inputs[arg_index])
            node.meta["val"] = value
            env[node.name] = value
            arg_index += 1
            continue
        if node.op == "call_function":
            args = tuple(_resolve(arg, env) for arg in node.args)
            kwargs = {key: _resolve(value, env) for key, value in node.kwargs.items()}
            value = infer_meta(node.target, args, kwargs)
            node.meta["val"] = value
            env[node.name] = value
            continue
        if node.op == "output":
            node.meta["val"] = _resolve(node.args[0], env)
    return graph_module
