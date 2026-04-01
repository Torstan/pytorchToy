"""
Pointwise graph lowering for the toy inductor backend.

第一阶段只支持:
  - inference-only
  - 全图纯逐元素
  - 输入全部为 contiguous float32 Tensor
  - 所有 Tensor 输入 shape 完全一致
"""

from dataclasses import dataclass


_VALUE_KIND_TO_INT = {
    "input": 0,
    "temp": 1,
    "const": 2,
}

_OP_KIND_TO_INT = {
    "sin": 0,
    "cos": 1,
    "relu": 2,
    "tanh": 3,
    "neg": 4,
    "add": 5,
    "sub": 6,
    "mul": 7,
    "div": 8,
}

_UNARY_TARGETS = {"sin", "cos", "relu", "tanh", "neg"}
_BINARY_TARGETS = {"add", "sub", "mul", "div"}
_SUPPORTED_TARGETS = _UNARY_TARGETS | _BINARY_TARGETS


class PointwiseLoweringError(RuntimeError):
    """Raised when a graph cannot use the pointwise fused fast path."""


@dataclass(frozen=True)
class ValueRef:
    kind: str
    index: int

    def encode(self):
        return (_VALUE_KIND_TO_INT[self.kind], self.index)


@dataclass(frozen=True)
class Instruction:
    op: str
    dst: int
    lhs: ValueRef
    rhs: ValueRef | None = None

    def encode(self):
        lhs_kind, lhs_index = self.lhs.encode()
        if self.rhs is None:
            rhs_kind = -1
            rhs_index = -1
        else:
            rhs_kind, rhs_index = self.rhs.encode()
        return (
            _OP_KIND_TO_INT[self.op],
            self.dst,
            lhs_kind,
            lhs_index,
            rhs_kind,
            rhs_index,
        )


@dataclass
class PointwiseProgram:
    shape: tuple[int, ...]
    num_inputs: int
    num_temps: int
    consts: list[float]
    instructions: list[Instruction]
    output: ValueRef

    def compile_cpp(self):
        import _C

        output_kind, output_index = self.output.encode()
        encoded_instructions = [instr.encode() for instr in self.instructions]
        return _C.CompiledPointwiseProgram(
            list(self.shape),
            self.num_inputs,
            self.num_temps,
            list(self.consts),
            encoded_instructions,
            output_kind,
            output_index,
        )

    def can_run_with(self, args):
        from torch.tensor import Tensor, float32

        if len(args) != self.num_inputs:
            return False
        for arg in args:
            if not isinstance(arg, Tensor):
                return False
            if arg.requires_grad:
                return False
            if not arg.is_contiguous():
                return False
            if tuple(arg.shape) != self.shape:
                return False
            if getattr(getattr(arg, "_dtype", None), "name", None) != float32.name:
                return False
        return True


def lower_pointwise_graph(graph_module, example_inputs):
    from torch._compile.graph import Node
    from torch.tensor import Tensor, float32

    if not example_inputs:
        raise PointwiseLoweringError("pointwise fast path requires at least one tensor input")

    tensor_inputs = []
    for inp in example_inputs:
        if not isinstance(inp, Tensor):
            raise PointwiseLoweringError("pointwise fast path only supports Tensor inputs")
        if inp.requires_grad:
            raise PointwiseLoweringError("requires_grad=True falls back to eager graph execution")
        if not inp.is_contiguous():
            raise PointwiseLoweringError("non-contiguous inputs fall back to eager graph execution")
        if getattr(getattr(inp, "_dtype", None), "name", None) != float32.name:
            raise PointwiseLoweringError("pointwise fast path only supports float32 tensors")
        tensor_inputs.append(inp)

    shape = tuple(tensor_inputs[0].shape)
    for inp in tensor_inputs[1:]:
        if tuple(inp.shape) != shape:
            raise PointwiseLoweringError("pointwise fast path requires exact shape matches")

    graph = graph_module.graph
    env = {}
    consts = []
    const_to_index = {}
    instructions = []
    num_inputs = 0
    num_temps = 0
    output_ref = None

    def add_const(value):
        if not isinstance(value, (int, float)):
            raise PointwiseLoweringError(f"unsupported constant operand type: {type(value)}")
        key = float(value)
        if key not in const_to_index:
            const_to_index[key] = len(consts)
            consts.append(key)
        return ValueRef("const", const_to_index[key])

    def resolve_ref(value):
        if isinstance(value, Node):
            if value not in env:
                raise PointwiseLoweringError(f"pointwise operand is not available in env: {value!r}")
            return env[value]
        if isinstance(value, (int, float)):
            return add_const(value)
        raise PointwiseLoweringError(f"unsupported pointwise operand: {value!r}")

    for node in graph.nodes:
        if node.op == "placeholder":
            env[node] = ValueRef("input", num_inputs)
            num_inputs += 1
            continue

        if node.op == "call_function":
            if node.target not in _SUPPORTED_TARGETS:
                raise PointwiseLoweringError(f"unsupported pointwise target: {node.target}")
            if node.kwargs:
                raise PointwiseLoweringError(f"kwargs are not supported in pointwise fast path: {node.target}")

            if node.target in _UNARY_TARGETS:
                if len(node.args) != 1:
                    raise PointwiseLoweringError(f"invalid unary node arity for {node.target}")
                lhs = resolve_ref(node.args[0])
                rhs = None
            else:
                if len(node.args) != 2:
                    raise PointwiseLoweringError(f"invalid binary node arity for {node.target}")
                lhs = resolve_ref(node.args[0])
                rhs = resolve_ref(node.args[1])

            env[node] = ValueRef("temp", num_temps)
            instructions.append(Instruction(node.target, num_temps, lhs, rhs))
            num_temps += 1
            continue

        if node.op == "output":
            if len(node.args) != 1:
                raise PointwiseLoweringError("pointwise fast path expects a single output value")
            output_value = node.args[0]
            if isinstance(output_value, (tuple, list)):
                raise PointwiseLoweringError("pointwise fast path does not support tuple/list outputs")
            output_ref = resolve_ref(output_value)
            continue

        raise PointwiseLoweringError(f"unsupported graph node op: {node.op}")

    if num_inputs != len(example_inputs):
        raise PointwiseLoweringError("placeholder count does not match runtime inputs")
    if output_ref is None:
        raise PointwiseLoweringError("graph is missing an output node")
    if output_ref.kind != "temp":
        raise PointwiseLoweringError("pointwise fast path requires output to come from a computed node")
    if not instructions:
        raise PointwiseLoweringError("pointwise fast path requires at least one pointwise op")

    return PointwiseProgram(
        shape=shape,
        num_inputs=num_inputs,
        num_temps=num_temps,
        consts=consts,
        instructions=instructions,
        output=output_ref,
    )
