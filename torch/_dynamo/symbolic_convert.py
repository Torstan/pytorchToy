import builtins
import dis
import inspect
import types
from dataclasses import dataclass

from torch._compile.graph import GraphModule
from torch._compile.tracer import UnsupportedTraceError
from torch.tensor import Tensor

from .variables import (
    ConstantVariable,
    PythonObjectVariable,
    TensorMethodVariable,
    TensorVariable,
    TorchModuleVariable,
    TorchOperatorVariable,
    UserFunctionVariable,
)


_BINARY_OPS = {
    "+": ("add", lambda lhs, rhs: lhs + rhs),
    "-": ("sub", lambda lhs, rhs: lhs - rhs),
    "*": ("mul", lambda lhs, rhs: lhs * rhs),
    "/": ("div", lambda lhs, rhs: lhs / rhs),
}

_COMPARE_OPS = {
    ">": lambda lhs, rhs: lhs > rhs,
    "<": lambda lhs, rhs: lhs < rhs,
    ">=": lambda lhs, rhs: lhs >= rhs,
    "<=": lambda lhs, rhs: lhs <= rhs,
    "==": lambda lhs, rhs: lhs == rhs,
    "!=": lambda lhs, rhs: lhs != rhs,
    "is": lambda lhs, rhs: lhs is rhs,
    "is not": lambda lhs, rhs: lhs is not rhs,
}

_TORCH_OPERATOR_NAMES = {
    "sin",
    "cos",
    "exp",
    "log",
    "relu",
    "tanh",
    "sum",
    "view",
    "reshape",
    "mm",
    "addmm",
}

_TENSOR_METHOD_NAMES = {
    "sin",
    "cos",
    "exp",
    "log",
    "relu",
    "tanh",
    "sum",
    "view",
    "reshape",
    "mm",
    "t",
    "gt",
}


@dataclass
class _InlineSpec:
    fn: object
    code: object
    globals_dict: dict
    closure_values: dict
    hidden_locals: dict
    parameter_names: tuple[str, ...]


def _callable_signature(fn):
    forward = getattr(fn, "forward", None)
    if callable(forward) and hasattr(fn, "__dict__") and "_modules" in fn.__dict__:
        try:
            return inspect.signature(forward)
        except (TypeError, ValueError):
            return None
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def _closure_values(fn):
    code = getattr(fn, "__code__", None)
    closure = getattr(fn, "__closure__", None)
    if code is None or not closure:
        return {}
    values = {}
    for name, cell in zip(code.co_freevars, closure):
        try:
            values[name] = cell.cell_contents
        except ValueError:
            continue
    return values


def _build_inline_spec(fn):
    method_fn = getattr(fn, "__func__", None)
    self_obj = getattr(fn, "__self__", None)
    if method_fn is not None and self_obj is not None:
        signature = _callable_signature(fn)
        if signature is None:
            raise UnsupportedTraceError("unable to inspect bound method signature for bytecode capture")
        return _InlineSpec(
            fn=fn,
            code=method_fn.__code__,
            globals_dict=getattr(method_fn, "__globals__", {}),
            closure_values=_closure_values(method_fn),
            hidden_locals={method_fn.__code__.co_varnames[0]: self_obj},
            parameter_names=tuple(signature.parameters.keys()),
        )

    if inspect.isfunction(fn):
        signature = _callable_signature(fn)
        if signature is None:
            raise UnsupportedTraceError("unable to inspect function signature for bytecode capture")
        return _InlineSpec(
            fn=fn,
            code=fn.__code__,
            globals_dict=getattr(fn, "__globals__", {}),
            closure_values=_closure_values(fn),
            hidden_locals={},
            parameter_names=tuple(signature.parameters.keys()),
        )

    forward = getattr(type(fn), "forward", None)
    bound_forward = getattr(fn, "forward", None)
    if callable(forward) and callable(bound_forward):
        signature = _callable_signature(fn)
        if signature is None:
            raise UnsupportedTraceError("unable to inspect module forward signature for bytecode capture")
        return _InlineSpec(
            fn=fn,
            code=forward.__code__,
            globals_dict=getattr(forward, "__globals__", {}),
            closure_values=_closure_values(forward),
            hidden_locals={forward.__code__.co_varnames[0]: fn},
            parameter_names=tuple(signature.parameters.keys()),
        )

    call = getattr(type(fn), "__call__", None)
    if call is not None and getattr(call, "__code__", None) is not None:
        signature = _callable_signature(fn)
        if signature is None:
            raise UnsupportedTraceError("unable to inspect callable object signature for bytecode capture")
        return _InlineSpec(
            fn=fn,
            code=call.__code__,
            globals_dict=getattr(call, "__globals__", {}),
            closure_values=_closure_values(call),
            hidden_locals={call.__code__.co_varnames[0]: fn},
            parameter_names=tuple(signature.parameters.keys()),
        )

    raise UnsupportedTraceError(f"bytecode capture does not support callable: {type(fn)}")


def _wrap_python_value(value):
    if isinstance(value, Tensor):
        return ConstantVariable(value)
    if isinstance(value, types.ModuleType) and getattr(value, "__name__", None) == "torch":
        return TorchModuleVariable(value)
    if inspect.isfunction(value):
        return UserFunctionVariable(value)
    if getattr(value, "__func__", None) is not None and getattr(value, "__self__", None) is not None:
        return UserFunctionVariable(value)
    if isinstance(value, (int, float, str, bool, type(None), tuple, list, dict)):
        return ConstantVariable(value)
    return PythonObjectVariable(value)


def _is_inlineable_python_callable(fn):
    if getattr(fn, "__func__", None) is not None and getattr(fn, "__self__", None) is not None:
        return True
    if inspect.isfunction(fn):
        return True

    forward = getattr(type(fn), "forward", None)
    bound_forward = getattr(fn, "forward", None)
    if callable(forward) and callable(bound_forward):
        return getattr(forward, "__code__", None) is not None

    call = getattr(type(fn), "__call__", None)
    return call is not None and getattr(call, "__code__", None) is not None


def _materialize_variable(value):
    if isinstance(value, TensorVariable):
        return value.node
    if isinstance(value, ConstantVariable):
        return value.value
    if isinstance(value, PythonObjectVariable):
        return value.value
    raise UnsupportedTraceError(f"cannot materialize variable type: {type(value)}")


def _contains_tensor_variable(value):
    if isinstance(value, TensorVariable):
        return True
    if isinstance(value, (ConstantVariable, PythonObjectVariable)):
        return False
    if isinstance(value, tuple):
        return any(_contains_tensor_variable(item) for item in value)
    if isinstance(value, list):
        return any(_contains_tensor_variable(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_tensor_variable(item) for item in value.values())
    return False


def _is_branch_stable_python_value(value):
    if isinstance(value, (int, float, str, bool, type(None))):
        return True
    if isinstance(value, tuple):
        return all(_is_branch_stable_python_value(item) for item in value)
    return False


class InstructionTranslator:
    def __init__(self, graph, spec, locals_env, *, inline_depth=0, max_inline_depth=4):
        self.graph = graph
        self.spec = spec
        self.locals_env = dict(locals_env)
        self.stack = []
        self.inline_depth = inline_depth
        self.max_inline_depth = max_inline_depth

    def run(self):
        instructions = list(dis.get_instructions(self.spec.code))
        offset_to_index = {
            instruction.offset: index
            for index, instruction in enumerate(instructions)
        }
        index = 0
        while index < len(instructions):
            instruction = instructions[index]
            action, payload = self._dispatch(instruction, offset_to_index)
            if action == "next":
                index += 1
                continue
            if action == "jump":
                index = payload
                continue
            if action == "return":
                return payload
            raise RuntimeError(f"unsupported bytecode action: {action}")
        raise UnsupportedTraceError("bytecode capture finished without RETURN_VALUE")

    def _dispatch(self, instruction, offset_to_index):
        opname = instruction.opname
        if opname in ("RESUME", "CACHE", "NOP", "PUSH_NULL"):
            return "next", None
        if opname == "LOAD_FAST":
            self._push(self.locals_env[instruction.argval])
            return "next", None
        if opname == "STORE_FAST":
            self.locals_env[instruction.argval] = self._pop()
            return "next", None
        if opname == "LOAD_CONST":
            self._push(ConstantVariable(instruction.argval))
            return "next", None
        if opname == "LOAD_GLOBAL":
            self._push(self._load_global(instruction.argval))
            return "next", None
        if opname == "LOAD_DEREF":
            value = self.spec.closure_values.get(instruction.argval)
            self._push(_wrap_python_value(value))
            return "next", None
        if opname == "LOAD_ATTR":
            owner = self._pop()
            self._push(self._load_attr(owner, instruction.argval))
            return "next", None
        if opname == "CALL":
            arg_count = instruction.arg or 0
            call_args = [self._pop() for _ in range(arg_count)]
            call_args.reverse()
            callable_var = self._pop()
            self._push(self._call_callable(callable_var, call_args))
            return "next", None
        if opname == "BINARY_OP":
            rhs = self._pop()
            lhs = self._pop()
            self._push(self._binary_op(lhs, rhs, instruction.argrepr))
            return "next", None
        if opname == "COMPARE_OP":
            rhs = self._pop()
            lhs = self._pop()
            self._push(self._compare_op(lhs, rhs, instruction.argrepr))
            return "next", None
        if opname == "POP_JUMP_IF_FALSE":
            condition = self._pop()
            if not self._truth_value(condition):
                return "jump", self._jump_index(instruction.argval, offset_to_index)
            return "next", None
        if opname == "POP_JUMP_IF_TRUE":
            condition = self._pop()
            if self._truth_value(condition):
                return "jump", self._jump_index(instruction.argval, offset_to_index)
            return "next", None
        if opname in ("JUMP_FORWARD", "JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"):
            return "jump", self._jump_index(instruction.argval, offset_to_index)
        if opname == "RETURN_VALUE":
            return "return", self._pop()
        raise UnsupportedTraceError(f"bytecode capture does not support opcode: {opname}")

    def _push(self, value):
        self.stack.append(value)

    def _pop(self):
        if not self.stack:
            raise UnsupportedTraceError("bytecode stack underflow during capture")
        return self.stack.pop()

    def _load_global(self, name):
        if name in self.spec.globals_dict:
            return _wrap_python_value(self.spec.globals_dict[name])
        if name in self.spec.closure_values:
            return _wrap_python_value(self.spec.closure_values[name])
        if hasattr(builtins, name):
            return _wrap_python_value(getattr(builtins, name))
        raise UnsupportedTraceError(f"unsupported global lookup during bytecode capture: {name}")

    def _load_attr(self, owner, name):
        if isinstance(owner, TensorVariable):
            if name not in _TENSOR_METHOD_NAMES:
                raise UnsupportedTraceError(f"unsupported Tensor attribute during bytecode capture: {name}")
            return TensorMethodVariable(owner, name)
        if isinstance(owner, TorchModuleVariable):
            if name not in _TORCH_OPERATOR_NAMES:
                raise UnsupportedTraceError(f"unsupported torch attribute during bytecode capture: {name}")
            return TorchOperatorVariable(name, getattr(owner.module, name))
        if isinstance(owner, ConstantVariable):
            return _wrap_python_value(getattr(owner.value, name))
        if isinstance(owner, PythonObjectVariable):
            return _wrap_python_value(getattr(owner.value, name))
        raise UnsupportedTraceError(f"unsupported LOAD_ATTR owner during bytecode capture: {type(owner)}")

    def _call_callable(self, callable_var, args):
        if isinstance(callable_var, TensorMethodVariable):
            return self._emit_graph_call(callable_var.name, [callable_var.base, *args])
        if isinstance(callable_var, TorchOperatorVariable):
            return self._emit_graph_call(callable_var.name, args)
        if isinstance(callable_var, UserFunctionVariable):
            return self._inline_user_function(callable_var.fn, args)
        if isinstance(callable_var, ConstantVariable) and callable(callable_var.value):
            if any(_contains_tensor_variable(arg) for arg in args) and _is_inlineable_python_callable(callable_var.value):
                return self._inline_user_function(callable_var.value, args)
            return self._constant_python_call(callable_var.value, args)
        if isinstance(callable_var, PythonObjectVariable) and callable(callable_var.value):
            if any(_contains_tensor_variable(arg) for arg in args) and _is_inlineable_python_callable(callable_var.value):
                return self._inline_user_function(callable_var.value, args)
            return self._constant_python_call(callable_var.value, args)
        raise UnsupportedTraceError(f"unsupported callable during bytecode capture: {type(callable_var)}")

    def _constant_python_call(self, fn, args):
        if any(_contains_tensor_variable(arg) for arg in args):
            raise UnsupportedTraceError(f"unsupported Python callable with symbolic args: {fn}")
        eager_args = [_materialize_variable(arg) for arg in args]
        return _wrap_python_value(fn(*eager_args))

    def _inline_user_function(self, fn, args):
        if self.inline_depth >= self.max_inline_depth:
            raise UnsupportedTraceError("bytecode inline depth exceeded")
        inline_spec = _build_inline_spec(fn)
        if len(args) != len(inline_spec.parameter_names):
            raise UnsupportedTraceError(
                f"bytecode inline arity mismatch for {fn}: expected {len(inline_spec.parameter_names)}, got {len(args)}"
            )
        inline_locals = {}
        for name, value in inline_spec.hidden_locals.items():
            inline_locals[name] = _wrap_python_value(value)
        for name, value in zip(inline_spec.parameter_names, args):
            inline_locals[name] = value
        return InstructionTranslator(
            self.graph,
            inline_spec,
            inline_locals,
            inline_depth=self.inline_depth + 1,
            max_inline_depth=self.max_inline_depth,
        ).run()

    def _binary_op(self, lhs, rhs, symbol):
        op_info = _BINARY_OPS.get(symbol)
        if op_info is None:
            raise UnsupportedTraceError(f"unsupported BINARY_OP during bytecode capture: {symbol}")
        target, eager_op = op_info
        if not _contains_tensor_variable(lhs) and not _contains_tensor_variable(rhs):
            return ConstantVariable(
                eager_op(_materialize_variable(lhs), _materialize_variable(rhs))
            )
        return self._emit_graph_call(target, [lhs, rhs])

    def _compare_op(self, lhs, rhs, symbol):
        eager_op = _COMPARE_OPS.get(symbol)
        if eager_op is None:
            raise UnsupportedTraceError(f"unsupported COMPARE_OP during bytecode capture: {symbol}")
        if _contains_tensor_variable(lhs) or _contains_tensor_variable(rhs):
            if symbol == ">":
                return self._emit_graph_call("gt", [lhs, rhs])
            raise UnsupportedTraceError(
                f"tensor COMPARE_OP during bytecode capture only supports '>', got: {symbol}"
            )
        eager_lhs = _materialize_variable(lhs)
        eager_rhs = _materialize_variable(rhs)
        if not _is_branch_stable_python_value(eager_lhs) or not _is_branch_stable_python_value(eager_rhs):
            raise UnsupportedTraceError(
                "mutable Python comparisons are not safe to specialize during bytecode capture"
            )
        return ConstantVariable(eager_op(eager_lhs, eager_rhs))

    def _truth_value(self, value):
        if _contains_tensor_variable(value):
            raise UnsupportedTraceError("data-dependent branches are not supported during bytecode capture")
        eager_value = _materialize_variable(value)
        if not _is_branch_stable_python_value(eager_value):
            raise UnsupportedTraceError(
                "mutable Python truthiness is not safe to specialize during bytecode capture"
            )
        return bool(eager_value)

    def _jump_index(self, target_offset, offset_to_index):
        target_index = offset_to_index.get(target_offset)
        if target_index is None:
            raise UnsupportedTraceError(
                f"bytecode jump target does not resolve to an instruction offset: {target_offset}"
            )
        return target_index

    def _emit_graph_call(self, target, args):
        if not any(_contains_tensor_variable(arg) for arg in args):
            return ConstantVariable(
                self._eager_call_target(
                    target,
                    [_materialize_variable(arg) for arg in args],
                )
            )
        node = self.graph.call_function(
            target,
            args=tuple(_materialize_variable(arg) for arg in args),
        )
        return TensorVariable(node)

    def _eager_call_target(self, target, args):
        if target == "sin":
            return args[0].sin()
        if target == "cos":
            return args[0].cos()
        if target == "exp":
            return args[0].exp()
        if target == "log":
            return args[0].log()
        if target == "relu":
            return args[0].relu()
        if target == "tanh":
            return args[0].tanh()
        if target == "sum":
            return args[0].sum()
        if target == "view":
            return args[0].view(*args[1:])
        if target == "reshape":
            return args[0].reshape(*args[1:])
        if target == "mm":
            return args[0].mm(args[1])
        if target == "t":
            return args[0].t()
        if target == "gt":
            return args[0].gt(args[1])
        if target == "addmm":
            return args[1].mm(args[2]) + args[0]
        if target == "add":
            return args[0] + args[1]
        if target == "sub":
            return args[0] - args[1]
        if target == "mul":
            return args[0] * args[1]
        if target == "div":
            return args[0] / args[1]
        raise UnsupportedTraceError(f"unsupported eager bytecode target: {target}")


def convert_callable_to_graph(fn, example_inputs, *, graph_input_positions=None):
    from torch._compile.graph import Graph

    spec = _build_inline_spec(fn)
    if len(example_inputs) != len(spec.parameter_names):
        raise UnsupportedTraceError(
            f"bytecode capture arity mismatch: expected {len(spec.parameter_names)}, got {len(example_inputs)}"
        )
    if graph_input_positions is None:
        graph_input_positions = tuple(range(len(example_inputs)))

    graph = Graph()
    locals_env = {}
    for name, value in spec.hidden_locals.items():
        locals_env[name] = _wrap_python_value(value)
    graph_example_inputs = []
    graph_input_positions = tuple(graph_input_positions)
    graph_input_position_set = set(graph_input_positions)
    for index, (name, value) in enumerate(zip(spec.parameter_names, example_inputs)):
        if index in graph_input_position_set:
            node = graph.placeholder(name)
            graph_example_inputs.append(value)
            if isinstance(value, Tensor):
                locals_env[name] = TensorVariable(node)
                continue
        locals_env[name] = _wrap_python_value(value)

    output = InstructionTranslator(graph, spec, locals_env).run()
    graph.output(_materialize_variable(output))
    graph_module = GraphModule(graph)
    graph_module.propagate_meta(graph_example_inputs)
    return graph_module
