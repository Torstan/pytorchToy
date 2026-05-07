"""
Single-node lowering kernels for the toy inductor backend.
"""

from dataclasses import dataclass

from torch._compile.ops import (
    BINARY_POINTWISE_TARGETS,
    UNARY_POINTWISE_TARGETS,
    normalize_shape_args,
    run_eager_target,
    target_name,
)
from torch._inductor.lowering.pointwise import PointwiseLoweringError


@dataclass(frozen=True)
class InputRef:
    index: int


@dataclass
class SingleNodeKernel:
    target: object
    args_spec: object
    kwargs_spec: object

    def run(self, args):
        call_args = _materialize_input_specs(self.args_spec, args)
        call_kwargs = _materialize_input_specs(self.kwargs_spec, args)
        return _run_kernel_target(self.target, call_args, call_kwargs)


@dataclass
class UnaryPointwiseKernel:
    target: str
    arg_spec: object

    def run(self, args):
        value = _materialize_input_specs(self.arg_spec, args)
        if self.target == "sin":
            return value.sin()
        if self.target == "cos":
            return value.cos()
        if self.target == "relu":
            return value.relu()
        if self.target == "tanh":
            return value.tanh()
        if self.target == "neg":
            return -value
        raise RuntimeError(f"unsupported unary pointwise target: {self.target}")


@dataclass
class BinaryPointwiseKernel:
    target: str
    lhs_spec: object
    rhs_spec: object

    def run(self, args):
        lhs = _materialize_input_specs(self.lhs_spec, args)
        rhs = _materialize_input_specs(self.rhs_spec, args)
        if self.target == "add":
            return lhs + rhs
        if self.target == "sub":
            return lhs - rhs
        if self.target == "mul":
            return lhs * rhs
        if self.target == "div":
            return lhs / rhs
        raise RuntimeError(f"unsupported binary pointwise target: {self.target}")


@dataclass
class GtKernel:
    lhs_spec: object
    rhs_spec: object

    def run(self, args):
        lhs = _materialize_input_specs(self.lhs_spec, args)
        rhs = _materialize_input_specs(self.rhs_spec, args)
        return lhs.gt(rhs)


@dataclass
class MmKernel:
    lhs_spec: object
    rhs_spec: object

    def run(self, args):
        lhs = _materialize_input_specs(self.lhs_spec, args)
        rhs = _materialize_input_specs(self.rhs_spec, args)
        return lhs.mm(rhs)


@dataclass
class AddmmKernel:
    bias_spec: object
    lhs_spec: object
    rhs_spec: object

    def run(self, args):
        bias = _materialize_input_specs(self.bias_spec, args)
        lhs = _materialize_input_specs(self.lhs_spec, args)
        rhs = _materialize_input_specs(self.rhs_spec, args)
        return lhs.mm(rhs) + bias


@dataclass
class SumKernel:
    arg_spec: object
    dim: object = None
    keepdim: bool = False

    def run(self, args):
        value = _materialize_input_specs(self.arg_spec, args)
        return value.sum(dim=self.dim, keepdim=self.keepdim)


@dataclass
class TransposeKernel:
    arg_spec: object

    def run(self, args):
        value = _materialize_input_specs(self.arg_spec, args)
        return value.t()


@dataclass
class ViewKernel:
    arg_spec: object
    shape: tuple[int, ...]

    def run(self, args):
        value = _materialize_input_specs(self.arg_spec, args)
        return value.view(*self.shape)


@dataclass
class ReshapeKernel:
    arg_spec: object
    shape: tuple[int, ...]

    def run(self, args):
        value = _materialize_input_specs(self.arg_spec, args)
        return value.reshape(*self.shape)


@dataclass
class LayerNormKernel:
    input_spec: object
    weight_spec: object
    bias_spec: object
    eps: float = 1e-5

    def run(self, args):
        import torch.nn.functional as F

        input_value = _materialize_input_specs(self.input_spec, args)
        weight = _materialize_input_specs(self.weight_spec, args)
        bias = _materialize_input_specs(self.bias_spec, args)
        return F.layer_norm(input_value, weight, bias, eps=self.eps)


def _materialize_input_specs(value, args):
    if isinstance(value, InputRef):
        return args[value.index]
    if isinstance(value, tuple):
        return tuple(_materialize_input_specs(item, args) for item in value)
    if isinstance(value, list):
        return [_materialize_input_specs(item, args) for item in value]
    if isinstance(value, dict):
        return {key: _materialize_input_specs(item, args) for key, item in value.items()}
    return value


def _run_kernel_target(target, call_args, call_kwargs):
    return run_eager_target(target, call_args, call_kwargs)


def try_compile_single_op(node, env_example):
    target = target_name(node.target)
    input_nodes = []
    seen_inputs = set()
    input_indices = {}

    def add_input(value):
        if value in seen_inputs:
            return input_indices[value]
        if value.name not in env_example:
            raise PointwiseLoweringError(
                f"single-op input is missing example value: {value.name}"
            )
        seen_inputs.add(value)
        input_indices[value] = len(input_nodes)
        input_nodes.append(value)
        return input_indices[value]

    def freeze_inputs(value):
        from torch.fx import Node

        if isinstance(value, Node):
            return InputRef(add_input(value))
        if isinstance(value, tuple):
            return tuple(freeze_inputs(item) for item in value)
        if isinstance(value, list):
            return [freeze_inputs(item) for item in value]
        if isinstance(value, dict):
            return {key: freeze_inputs(item) for key, item in value.items()}
        return value

    kernel = None

    if target == "mm":
        if len(node.args) != 2 or node.kwargs:
            return None
        lhs_spec = freeze_inputs(node.args[0])
        rhs_spec = freeze_inputs(node.args[1])
        kernel = MmKernel(lhs_spec=lhs_spec, rhs_spec=rhs_spec)
    elif target == "addmm":
        if len(node.args) != 3 or node.kwargs:
            return None
        bias_spec = freeze_inputs(node.args[0])
        lhs_spec = freeze_inputs(node.args[1])
        rhs_spec = freeze_inputs(node.args[2])
        kernel = AddmmKernel(
            bias_spec=bias_spec,
            lhs_spec=lhs_spec,
            rhs_spec=rhs_spec,
        )
    elif target == "sum":
        if len(node.args) != 1:
            return None
        arg_spec = freeze_inputs(node.args[0])
        kernel = SumKernel(
            arg_spec=arg_spec,
            dim=node.kwargs.get("dim"),
            keepdim=node.kwargs.get("keepdim", False),
        )
    elif target == "t":
        if len(node.args) != 1 or node.kwargs:
            return None
        arg_spec = freeze_inputs(node.args[0])
        kernel = TransposeKernel(arg_spec=arg_spec)
    elif target == "view":
        if not node.args or node.kwargs:
            return None
        arg_spec = freeze_inputs(node.args[0])
        kernel = ViewKernel(arg_spec=arg_spec, shape=normalize_shape_args(node.args))
    elif target == "reshape":
        if not node.args or node.kwargs:
            return None
        arg_spec = freeze_inputs(node.args[0])
        kernel = ReshapeKernel(arg_spec=arg_spec, shape=normalize_shape_args(node.args))
    elif target == "layer_norm":
        if len(node.args) != 3:
            return None
        input_spec = freeze_inputs(node.args[0])
        weight_spec = freeze_inputs(node.args[1])
        bias_spec = freeze_inputs(node.args[2])
        kernel = LayerNormKernel(
            input_spec=input_spec,
            weight_spec=weight_spec,
            bias_spec=bias_spec,
            eps=node.kwargs.get("eps", 1e-5),
        )
    elif target in UNARY_POINTWISE_TARGETS:
        if len(node.args) != 1 or node.kwargs:
            return None
        kernel = UnaryPointwiseKernel(
            target=target,
            arg_spec=freeze_inputs(node.args[0]),
        )
    elif target in BINARY_POINTWISE_TARGETS:
        if len(node.args) != 2 or node.kwargs:
            return None
        kernel = BinaryPointwiseKernel(
            target=target,
            lhs_spec=freeze_inputs(node.args[0]),
            rhs_spec=freeze_inputs(node.args[1]),
        )
    elif target == "gt":
        if len(node.args) != 2 or node.kwargs:
            return None
        kernel = GtKernel(
            lhs_spec=freeze_inputs(node.args[0]),
            rhs_spec=freeze_inputs(node.args[1]),
        )

    if kernel is None:
        return None

    from torch._inductor.lowering.partition import CompiledOpStep

    return CompiledOpStep(
        target=target,
        input_nodes=input_nodes,
        output_node=node,
        compiled_kernel=kernel,
    )
