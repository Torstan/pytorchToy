"""
Pointwise graph lowering for the toy inductor backend.

第一阶段只支持:
  - inference-only
  - 全图纯逐元素
  - 输入全部为 contiguous float32 Tensor
  - 所有 Tensor 输入 shape 完全一致
"""

import ctypes
from dataclasses import dataclass
import hashlib
import os
import shutil
import subprocess
import tempfile

from torch._compile.ops import (
    BINARY_POINTWISE_TARGETS,
    POINTWISE_TARGETS,
    UNARY_POINTWISE_TARGETS,
    broadcast_shapes,
    target_name,
)


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

_NATIVE_KERNEL_CACHE = {}


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
    allow_requires_grad: bool = False

    def compile_interpreter(self):
        import _C

        output_kind, output_index = self.output.encode()
        encoded_instructions = [instr.encode() for instr in self.instructions]
        return CppPointwiseKernel(
            compiled_program=_C.CompiledPointwiseProgram(
                list(self.shape),
                self.num_inputs,
                self.num_temps,
                list(self.consts),
                encoded_instructions,
                output_kind,
                output_index,
            ),
            shape=self.shape,
            allow_requires_grad=self.allow_requires_grad,
        )

    def compile(self):
        try:
            return self.compile_native()
        except PointwiseLoweringError:
            return self.compile_interpreter()

    def compile_native(self):
        if os.name == "nt":
            raise PointwiseLoweringError("native pointwise JIT is only enabled on POSIX")

        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            raise PointwiseLoweringError("no C++ compiler available for native pointwise JIT")

        symbol = f"pointwise_{hashlib.sha256(self.render_signature().encode('utf-8')).hexdigest()[:16]}"
        source = self.render_native_source(symbol)
        key = hashlib.sha256(source.encode("utf-8")).hexdigest()
        cached = _NATIVE_KERNEL_CACHE.get(key)
        if cached is not None:
            library, kernel_fn = cached
            return NativePointwiseKernel(
                shape=self.shape,
                num_inputs=self.num_inputs,
                symbol=symbol,
                library=library,
                kernel_fn=kernel_fn,
                allow_requires_grad=self.allow_requires_grad,
            )

        cache_dir = os.path.join(tempfile.gettempdir(), "pytorchtoy_pointwise")
        os.makedirs(cache_dir, exist_ok=True)
        file_stem = f"pointwise_{key[:16]}"
        cpp_path = os.path.join(cache_dir, f"{file_stem}.cpp")
        so_path = os.path.join(cache_dir, f"{file_stem}.so")

        if not os.path.exists(so_path):
            with open(cpp_path, "w", encoding="utf-8") as source_file:
                source_file.write(source)
            cmd = [
                compiler,
                "-shared",
                "-fPIC",
                "-O3",
                "-std=c++17",
                "-march=native",
                "-ffast-math",
                cpp_path,
                "-o",
                so_path,
            ]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                detail = getattr(exc, "stderr", "") or str(exc)
                raise PointwiseLoweringError(
                    f"native pointwise JIT compile failed: {detail.strip()}"
                ) from exc

        try:
            library = ctypes.CDLL(so_path)
            kernel_fn = getattr(library, symbol)
        except (AttributeError, OSError) as exc:
            raise PointwiseLoweringError(
                f"native pointwise JIT load failed: {exc}"
            ) from exc

        kernel_fn.argtypes = [ctypes.c_void_p] * (self.num_inputs + 1)
        kernel_fn.restype = None

        _NATIVE_KERNEL_CACHE[key] = (library, kernel_fn)
        return NativePointwiseKernel(
            shape=self.shape,
            num_inputs=self.num_inputs,
            symbol=symbol,
            library=library,
            kernel_fn=kernel_fn,
            allow_requires_grad=self.allow_requires_grad,
        )

    def render_native_source(self, symbol_name):
        args = [
            f"const float* __restrict__ in{i}"
            for i in range(self.num_inputs)
        ]
        args.append("float* __restrict__ out")

        lines = [
            "#include <cmath>",
            "#include <limits>",
            "",
            f'extern "C" __attribute__((visibility("default")))',
            f"void {symbol_name}({', '.join(args)}) {{",
            "    #if defined(__GNUC__)",
            "    #pragma GCC ivdep",
            "    #endif",
            f"    for (int i = 0; i < {self.numel}; ++i) {{",
        ]

        for instr in self.instructions:
            lines.append(f"        float t{instr.dst} = {self._render_expr(instr)};")

        lines.append(f"        out[i] = {self._render_value_ref(self.output)};")
        lines.append("    }")
        lines.append("}")
        return "\n".join(lines) + "\n"

    @property
    def numel(self):
        total = 1
        for size in self.shape:
            total *= size
        return total

    def render_signature(self):
        encoded_instructions = [instr.encode() for instr in self.instructions]
        return repr(
            (
                self.shape,
                self.num_inputs,
                self.num_temps,
                tuple(self.consts),
                tuple(encoded_instructions),
                self.output.encode(),
            )
        )

    def _render_expr(self, instr):
        lhs = self._render_value_ref(instr.lhs)
        if instr.op == "sin":
            return f"std::sin({lhs})"
        if instr.op == "cos":
            return f"std::cos({lhs})"
        if instr.op == "relu":
            return f"(({lhs}) > 0.0f ? ({lhs}) : 0.0f)"
        if instr.op == "tanh":
            return f"std::tanh({lhs})"
        if instr.op == "neg":
            return f"-({lhs})"

        rhs = self._render_value_ref(instr.rhs)
        if instr.op == "add":
            return f"({lhs}) + ({rhs})"
        if instr.op == "sub":
            return f"({lhs}) - ({rhs})"
        if instr.op == "mul":
            return f"({lhs}) * ({rhs})"
        if instr.op == "div":
            return f"({lhs}) / ({rhs})"
        raise PointwiseLoweringError(f"unsupported codegen op: {instr.op}")

    def _render_value_ref(self, value_ref):
        if value_ref.kind == "input":
            return f"in{value_ref.index}[i]"
        if value_ref.kind == "temp":
            return f"t{value_ref.index}"
        if value_ref.kind == "const":
            return _format_cpp_float(self.consts[value_ref.index])
        raise PointwiseLoweringError(f"unsupported value ref kind: {value_ref.kind}")


@dataclass
class NativePointwiseKernel:
    shape: tuple[int, ...]
    num_inputs: int
    symbol: str
    library: object
    kernel_fn: object
    allow_requires_grad: bool = False

    def run(self, args):
        import _C
        from torch.tensor import Tensor

        if len(args) != self.num_inputs:
            raise RuntimeError(f"{self.symbol}: input count mismatch")

        prepared_args = _prepare_pointwise_runtime_args(
            args,
            self.shape,
            allow_requires_grad=self.allow_requires_grad,
            kernel_name=self.symbol,
        )
        output = Tensor(_C.empty(list(self.shape)))
        call_args = [ctypes.c_void_p(arg._c.data_ptr_address()) for arg in prepared_args]
        call_args.append(ctypes.c_void_p(output._c.data_ptr_address()))
        self.kernel_fn(*call_args)
        return output


@dataclass
class CppPointwiseKernel:
    compiled_program: object
    shape: tuple[int, ...]
    allow_requires_grad: bool = False

    def run(self, args):
        from torch.tensor import Tensor

        prepared_args = _prepare_pointwise_runtime_args(
            args,
            self.shape,
            allow_requires_grad=self.allow_requires_grad,
            kernel_name="CompiledPointwiseProgram",
        )
        return Tensor(self.compiled_program.run([arg._c for arg in prepared_args]))


def _format_cpp_float(value):
    text = repr(float(value))
    if text == "nan":
        return 'std::numeric_limits<float>::quiet_NaN()'
    if text == "inf":
        return 'std::numeric_limits<float>::infinity()'
    if text == "-inf":
        return '-std::numeric_limits<float>::infinity()'
    return f"{text}f"


def _is_broadcastable_to(input_shape, output_shape):
    input_shape = list(input_shape)
    output_shape = list(output_shape)
    while output_shape:
        out_dim = output_shape.pop()
        in_dim = input_shape.pop() if input_shape else 1
        if in_dim not in (1, out_dim):
            return False
    return not input_shape


def _prepare_pointwise_runtime_args(args, output_shape, *, allow_requires_grad, kernel_name):
    prepared_args = []
    output_shape = tuple(output_shape)
    for arg in args:
        if allow_requires_grad and arg.requires_grad:
            arg = arg.detach()
        elif arg.requires_grad:
            raise RuntimeError(f"{kernel_name}: requires_grad input is not supported")

        if tuple(arg.shape) != output_shape:
            if not _is_broadcastable_to(tuple(arg.shape), output_shape):
                raise RuntimeError(
                    f"{kernel_name}: input shape {tuple(arg.shape)} cannot broadcast to {output_shape}"
                )
            if len(arg.shape) < len(output_shape):
                arg = arg.view((1,) * (len(output_shape) - len(arg.shape)) + tuple(arg.shape))
            arg = arg.expand(output_shape).contiguous()
        elif not arg.is_contiguous():
            arg = arg.contiguous()

        prepared_args.append(arg)
    return prepared_args


def _broadcast_shapes(lhs_shape, rhs_shape):
    return broadcast_shapes(
        lhs_shape,
        rhs_shape,
        error_type=PointwiseLoweringError,
    )


def lower_pointwise_graph(graph_module, example_inputs, *, allow_requires_grad=False):
    from torch.fx import Node
    from torch.tensor import Tensor, float32

    if not example_inputs:
        raise PointwiseLoweringError("pointwise fast path requires at least one tensor input")

    tensor_inputs = []
    for inp in example_inputs:
        if not isinstance(inp, Tensor):
            raise PointwiseLoweringError("pointwise fast path only supports Tensor inputs")
        if inp.requires_grad and not allow_requires_grad:
            raise PointwiseLoweringError("requires_grad=True falls back to eager graph execution")
        if not inp.is_contiguous():
            raise PointwiseLoweringError("non-contiguous inputs fall back to eager graph execution")
        if getattr(getattr(inp, "_dtype", None), "name", None) != float32.name:
            raise PointwiseLoweringError("pointwise fast path only supports float32 tensors")
        tensor_inputs.append(inp)

    shape = tuple(tensor_inputs[0].shape)
    for inp in tensor_inputs[1:]:
        shape = _broadcast_shapes(shape, tuple(inp.shape))

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
            target = target_name(node.target)
            if target not in POINTWISE_TARGETS:
                raise PointwiseLoweringError(f"unsupported pointwise target: {node.target}")
            if node.kwargs:
                raise PointwiseLoweringError(f"kwargs are not supported in pointwise fast path: {node.target}")

            if target in UNARY_POINTWISE_TARGETS:
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
            instructions.append(Instruction(target, num_temps, lhs, rhs))
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
        allow_requires_grad=allow_requires_grad,
    )
