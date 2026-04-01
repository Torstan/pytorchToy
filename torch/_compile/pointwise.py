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

    def compile_interpreter(self):
        import _C

        output_kind, output_index = self.output.encode()
        encoded_instructions = [instr.encode() for instr in self.instructions]
        return CppPointwiseKernel(_C.CompiledPointwiseProgram(
            list(self.shape),
            self.num_inputs,
            self.num_temps,
            list(self.consts),
            encoded_instructions,
            output_kind,
            output_index,
        ))

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
            return cached

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

        kernel = NativePointwiseKernel(
            shape=self.shape,
            num_inputs=self.num_inputs,
            symbol=symbol,
            library=library,
            kernel_fn=kernel_fn,
        )
        _NATIVE_KERNEL_CACHE[key] = kernel
        return kernel

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

    def run(self, args):
        import _C
        from torch.tensor import Tensor

        if len(args) != self.num_inputs:
            raise RuntimeError(f"{self.symbol}: input count mismatch")

        output = Tensor(_C.empty(list(self.shape)))
        call_args = [ctypes.c_void_p(arg._c.data_ptr_address()) for arg in args]
        call_args.append(ctypes.c_void_p(output._c.data_ptr_address()))
        self.kernel_fn(*call_args)
        return output


@dataclass
class CppPointwiseKernel:
    compiled_program: object

    def run(self, args):
        from torch.tensor import Tensor

        return Tensor(self.compiled_program.run([arg._c for arg in args]))


def _format_cpp_float(value):
    text = repr(float(value))
    if text == "nan":
        return 'std::numeric_limits<float>::quiet_NaN()'
    if text == "inf":
        return 'std::numeric_limits<float>::infinity()'
    if text == "-inf":
        return '-std::numeric_limits<float>::infinity()'
    return f"{text}f"


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


@dataclass
class CompiledRegion:
    start_index: int
    end_index: int
    input_nodes: list
    output_node: object
    compiled_kernel: object

    def run(self, env):
        args = [env[node.name] for node in self.input_nodes]
        env[self.output_node.name] = self.compiled_kernel.run(args)


@dataclass
class CompiledGraph:
    placeholders: list
    steps: list
    output_value: object

    def run(self, args):
        from torch._compile.graph import _resolve

        if len(args) != len(self.placeholders):
            raise RuntimeError(
                f"compiled graph expected {len(self.placeholders)} inputs, got {len(args)}"
            )

        env = {}
        for node, value in zip(self.placeholders, args):
            env[node.name] = value

        for step in self.steps:
            if isinstance(step, CompiledRegion):
                step.run(env)
            else:
                env[step.name] = _run_call_function_node(step, env)

        return _resolve(self.output_value, env)


def compile_graph_module(graph_module, example_inputs):
    try:
        return lower_pointwise_graph(graph_module, example_inputs).compile()
    except PointwiseLoweringError:
        compiled_graph = _compile_partitioned_graph(graph_module, example_inputs)
        if compiled_graph is None:
            raise
        return compiled_graph


def _compile_partitioned_graph(graph_module, example_inputs):
    graph = graph_module.graph
    placeholders = []
    steps = []
    users = _build_users(graph.nodes)
    env_example = {}
    compiled_any = False
    output_value = None
    arg_idx = 0
    idx = 0

    while idx < len(graph.nodes):
        node = graph.nodes[idx]

        if node.op == "placeholder":
            if arg_idx >= len(example_inputs):
                return None
            placeholders.append(node)
            env_example[node.name] = example_inputs[arg_idx]
            arg_idx += 1
            idx += 1
            continue

        if node.op == "call_function":
            region = _try_compile_region(graph.nodes, idx, users, env_example)
            if region is not None:
                compiled_any = True
                region.run(env_example)
                steps.append(region)
                idx = region.end_index + 1
                continue

            env_example[node.name] = _run_call_function_node(node, env_example)
            steps.append(node)
            idx += 1
            continue

        if node.op == "output":
            if len(node.args) != 1:
                return None
            output_value = node.args[0]
            idx += 1
            continue

        return None

    if not compiled_any or output_value is None:
        return None
    return CompiledGraph(placeholders, steps, output_value)


def _try_compile_region(nodes, start_index, users, env_example):
    start_node = nodes[start_index]
    if start_node.op != "call_function" or start_node.target not in _SUPPORTED_TARGETS:
        return None

    max_end_index = start_index
    while max_end_index + 1 < len(nodes):
        next_node = nodes[max_end_index + 1]
        if next_node.op != "call_function" or next_node.target not in _SUPPORTED_TARGETS:
            break
        max_end_index += 1

    for end_index in range(max_end_index, start_index - 1, -1):
        region_nodes = nodes[start_index:end_index + 1]
        if not _region_has_single_output(region_nodes, users):
            continue
        try:
            region_graph_module, region_inputs, input_nodes = _build_region_graph_module(
                region_nodes, env_example
            )
            compiled_kernel = lower_pointwise_graph(
                region_graph_module, region_inputs
            ).compile()
        except PointwiseLoweringError:
            continue

        return CompiledRegion(
            start_index=start_index,
            end_index=end_index,
            input_nodes=input_nodes,
            output_node=region_nodes[-1],
            compiled_kernel=compiled_kernel,
        )

    return None


def _build_region_graph_module(region_nodes, env_example):
    from torch._compile.graph import Graph, GraphModule, Node

    graph = Graph()
    mapping = {}
    input_nodes = []
    seen_inputs = set()

    def register_input(node):
        if node not in seen_inputs:
            seen_inputs.add(node)
            input_nodes.append(node)
            mapping[node] = graph.placeholder(node.name)

    def rewrite(value):
        if isinstance(value, Node):
            if value in mapping:
                return mapping[value]
            if value.name not in env_example:
                raise PointwiseLoweringError(
                    f"region input is missing example value: {value.name}"
                )
            register_input(value)
            return mapping[value]
        if isinstance(value, tuple):
            return tuple(rewrite(item) for item in value)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    for node in region_nodes:
        new_node = graph.call_function(
            node.target,
            args=rewrite(node.args),
            kwargs=rewrite(node.kwargs),
        )
        mapping[node] = new_node

    graph.output(mapping[region_nodes[-1]])
    region_inputs = [env_example[node.name] for node in input_nodes]
    return GraphModule(graph), region_inputs, input_nodes


def _region_has_single_output(region_nodes, users):
    region_set = set(region_nodes)
    for node in region_nodes[:-1]:
        if any(user not in region_set for user in users.get(node, ())):
            return False
    return True


def _build_users(nodes):
    from torch._compile.graph import Node

    users = {}

    def visit_arg(owner, value):
        if isinstance(value, Node):
            users.setdefault(value, []).append(owner)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit_arg(owner, item)
        elif isinstance(value, dict):
            for item in value.values():
                visit_arg(owner, item)

    for node in nodes:
        for arg in node.args:
            visit_arg(node, arg)
        for value in node.kwargs.values():
            visit_arg(node, value)
    return users


def _run_call_function_node(node, env):
    from torch._compile.graph import _OP_TABLE, _resolve

    call_args = tuple(_resolve(arg, env) for arg in node.args)
    call_kwargs = {key: _resolve(value, env) for key, value in node.kwargs.items()}
    op_fn = _OP_TABLE.get(node.target)
    if op_fn is None:
        raise RuntimeError(f"unsupported compiled target: {node.target}")
    return op_fn(call_args, call_kwargs)
