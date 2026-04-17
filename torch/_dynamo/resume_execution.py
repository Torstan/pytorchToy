"""
最小 graph break / resume 执行计划。

这一版不做 bytecode continuation，只做函数体顶层语句级分段：
- 直线 `Assign` / `Return` 语句尝试单独编译
- `If` 等 graph break 语句保留 eager 执行
- eager 语句产出的局部变量会继续喂给后续 compiled region
"""

import ast
import inspect
import textwrap
from dataclasses import dataclass

from torch._compile.tracer import Tracer, UnsupportedTraceError

_NO_RETURN = object()


def _is_runtime_input(value):
    from torch.tensor import Tensor

    return isinstance(value, (Tensor, int, float, bool, str, type(None)))


def _assigned_names(stmt):
    names = []

    def add_name(name):
        if name not in names:
            names.append(name)

    def visit_target(target):
        if isinstance(target, ast.Name):
            add_name(target.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                visit_target(item)

    def visit_stmt(node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                visit_target(target)
            return
        if isinstance(node, ast.AugAssign):
            visit_target(node.target)
            return
        if isinstance(node, ast.AnnAssign):
            visit_target(node.target)
            return
        if isinstance(node, ast.If):
            for child in node.body:
                visit_stmt(child)
            for child in node.orelse:
                visit_stmt(child)
            return
        if isinstance(node, (ast.For, ast.While)):
            for child in node.body:
                visit_stmt(child)
            for child in node.orelse:
                visit_stmt(child)

    visit_stmt(stmt)
    return tuple(names)


def _assigned_names_in_block(stmts):
    names = []
    for stmt in stmts:
        for name in _assigned_names(stmt):
            if name not in names:
                names.append(name)
    return tuple(names)


def _contains_return(stmt):
    found = False

    class Visitor(ast.NodeVisitor):
        def visit_Return(self, node):
            del node
            nonlocal found
            found = True

    Visitor().visit(stmt)
    return found


def _stmt_guaranteed_return(stmt):
    if isinstance(stmt, ast.Return):
        return True
    if isinstance(stmt, ast.If):
        if not stmt.body or not stmt.orelse:
            return False
        return _block_guaranteed_return(stmt.body) and _block_guaranteed_return(stmt.orelse)
    return False


def _block_guaranteed_return(stmts):
    for stmt in stmts:
        if _stmt_guaranteed_return(stmt):
            return True
    return False


def _build_return_source(output_names):
    if not output_names:
        return "return None"
    if len(output_names) == 1:
        return f"return {output_names[0]}"
    return "return " + ", ".join(output_names)


def _build_output_expr(output_names):
    if not output_names:
        return "None"
    if len(output_names) == 1:
        return output_names[0]
    return "(" + ", ".join(output_names) + ")"


def _safe_maybe_return_outputs(stmt):
    if not isinstance(stmt, ast.If):
        return None

    body_returns = _block_guaranteed_return(stmt.body)
    orelse_returns = _block_guaranteed_return(stmt.orelse)
    if body_returns == orelse_returns:
        return None

    non_return_branch = stmt.orelse if body_returns else stmt.body
    if any(_contains_return(child) for child in non_return_branch):
        return None
    return _assigned_names_in_block(non_return_branch)


def _make_step_function(spec, stmts, env, local_names, *, output_names, returns, maybe_returns=False):
    param_names = tuple(
        name for name in local_names
        if name in env and _is_runtime_input(env[name])
    )
    captured_names = [
        name for name in local_names
        if name in env and name not in param_names
    ]

    globals_dict = dict(spec.globals_dict)
    globals_dict.update(spec.closure_values)
    for name in captured_names:
        globals_dict[name] = env[name]

    if maybe_returns:
        body_lines = [
            "    def __resume_inner():",
        ]
        body_lines.extend(
            textwrap.indent(ast.unparse(stmt), "        ")
            for stmt in stmts
        )
        body_lines.append(
            f"        return (__resume_no_return__, {_build_output_expr(output_names)})"
        )
        body_lines.append("    __resume_result = __resume_inner()")
        body_lines.append("    if __resume_result[0] is __resume_no_return__:")
        body_lines.append("        return False, __resume_result[1]")
        body_lines.append("    return True, __resume_result")
    else:
        body_lines = [
            textwrap.indent(ast.unparse(stmt), "    ")
            for stmt in stmts
        ]
        if not returns:
            body_lines.append(f"    {_build_return_source(output_names)}")

    params = ", ".join(param_names)
    source = f"def __resume_step({params}):\n" + "\n".join(body_lines) + "\n"

    globals_dict["__resume_no_return__"] = _NO_RETURN
    namespace = {}
    exec(compile(source, spec.filename, "exec"), globals_dict, namespace)
    return namespace["__resume_step"], param_names


def _normalize_step_outputs(result, output_names):
    if not output_names:
        return ()
    if len(output_names) == 1:
        return (result,)
    if not isinstance(result, (tuple, list)):
        raise RuntimeError("resume step with multiple outputs must return tuple/list")
    if len(result) != len(output_names):
        raise RuntimeError("resume step output arity mismatch")
    return tuple(result)


@dataclass
class CompiledStep:
    fn: object
    simulate_fn: object
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    returns: bool

    def _execute(self, env, fn):
        args = [env[name] for name in self.input_names]
        result = fn(*args)
        if self.returns:
            return True, result
        for name, value in zip(self.output_names, _normalize_step_outputs(result, self.output_names)):
            env[name] = value
        return False, None

    def run(self, env):
        return self._execute(env, self.fn)

    def simulate(self, env):
        return self._execute(env, self.simulate_fn)


@dataclass
class EagerStep:
    fn: object
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    returns: bool
    maybe_returns: bool = False
    guaranteed_return: bool = False

    def run(self, env):
        args = [env[name] for name in self.input_names]
        result = self.fn(*args)
        if self.maybe_returns:
            returned, value = result
            if returned:
                return True, value
            for name, output in zip(
                self.output_names,
                _normalize_step_outputs(value, self.output_names),
            ):
                env[name] = output
            return False, None
        if self.returns:
            return True, result
        for name, value in zip(self.output_names, _normalize_step_outputs(result, self.output_names)):
            env[name] = value
        return False, None

    def simulate(self, env):
        return self.run(env)


@dataclass
class _CallableSpec:
    fn: object
    function_def: object
    signature: object
    globals_dict: dict
    closure_values: dict
    filename: str
    self_name: str | None = None

    def bind_arguments(self, args, kwargs):
        bound = self.signature.bind(*args, **kwargs)
        env = {}
        if self.self_name is not None:
            env[self.self_name] = self.fn
        env.update(bound.arguments)
        return env


class ResumeExecutionPlan:
    def __init__(self, spec, steps):
        self._spec = spec
        self._steps = steps

    def __call__(self, *args, **kwargs):
        env = self._spec.bind_arguments(args, kwargs)
        for step in self._steps:
            returned, result = step.run(env)
            if returned:
                return result
        raise RuntimeError("resume execution plan finished without return")


def _extract_function_def(fn):
    source = textwrap.dedent(inspect.getsource(fn))
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise RuntimeError("unable to find function definition for resume execution")


def _make_callable_spec(fn):
    if inspect.isfunction(fn):
        function_def = _extract_function_def(fn)
        return _CallableSpec(
            fn=fn,
            function_def=function_def,
            signature=inspect.signature(fn),
            globals_dict=getattr(fn, "__globals__", {}),
            closure_values={
                name: cell.cell_contents
                for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ())
            },
            filename=inspect.getsourcefile(fn) or "<resume_execution>",
        )

    forward = getattr(type(fn), "forward", None)
    bound_forward = getattr(fn, "forward", None)
    if callable(forward) and callable(bound_forward):
        function_def = _extract_function_def(forward)
        args = list(function_def.args.args)
        self_name = args[0].arg if args else None
        return _CallableSpec(
            fn=fn,
            function_def=function_def,
            signature=inspect.signature(bound_forward),
            globals_dict=getattr(forward, "__globals__", {}),
            closure_values={},
            filename=inspect.getsourcefile(forward) or "<resume_execution>",
            self_name=self_name,
        )

    return None


def _build_eager_step(spec, stmt, env, local_names):
    output_names = _assigned_names_in_block([stmt])
    returns = isinstance(stmt, ast.Return)
    maybe_returns = not returns and _contains_return(stmt)
    if maybe_returns:
        safe_outputs = _safe_maybe_return_outputs(stmt)
        if safe_outputs is None and output_names:
            return None
        if safe_outputs is not None:
            output_names = safe_outputs
    eager_fn, input_names = _make_step_function(
        spec,
        [stmt],
        env,
        local_names,
        output_names=output_names,
        returns=returns,
        maybe_returns=maybe_returns,
    )
    return EagerStep(
        fn=eager_fn,
        input_names=input_names,
        output_names=output_names,
        returns=returns,
        maybe_returns=maybe_returns,
        guaranteed_return=returns or _stmt_guaranteed_return(stmt),
    )


def _is_linear_compilable_stmt(stmt):
    if isinstance(stmt, ast.Assign):
        return len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
    return isinstance(stmt, ast.Return)


def _build_compiled_step(spec, stmts, env, local_names, backend):
    if not stmts or not all(_is_linear_compilable_stmt(stmt) for stmt in stmts):
        return None

    returns = isinstance(stmts[-1], ast.Return)
    if any(isinstance(stmt, ast.Return) for stmt in stmts[:-1]):
        return None
    output_names = () if returns else _assigned_names_in_block(stmts)

    step_fn, input_names = _make_step_function(
        spec,
        stmts,
        env,
        local_names,
        output_names=output_names,
        returns=returns,
    )
    if not input_names:
        return None

    example_inputs = [env[name] for name in input_names]
    tracer = Tracer()
    graph_module = tracer.trace(step_fn, example_inputs)
    compiled_fn = backend(graph_module, list(example_inputs))
    return CompiledStep(
        fn=compiled_fn,
        simulate_fn=step_fn,
        input_names=input_names,
        output_names=output_names,
        returns=returns,
    )


def _linear_region_end(body, start_index):
    index = start_index
    while index < len(body) and _is_linear_compilable_stmt(body[index]):
        if isinstance(body[index], ast.Return):
            return index + 1
        index += 1
    return index


def _build_compiled_linear_region(spec, body, start_index, env, local_names, backend):
    region_end = _linear_region_end(body, start_index)
    for end_index in range(region_end, start_index, -1):
        stmts = body[start_index:end_index]
        try:
            step = _build_compiled_step(spec, stmts, env, local_names, backend)
        except UnsupportedTraceError:
            step = None
        if step is not None:
            return step, end_index
    return None, start_index


def build_resume_plan(fn, backend, args, kwargs):
    try:
        spec = _make_callable_spec(fn)
    except (OSError, TypeError, RuntimeError, SyntaxError):
        return None

    if spec is None:
        return None

    try:
        env = spec.bind_arguments(args, kwargs)
    except TypeError:
        return None

    local_names = list(env.keys())
    steps = []
    body = spec.function_def.body
    index = 0

    while index < len(body):
        stmt = body[index]

        if _is_linear_compilable_stmt(stmt):
            step, end_index = _build_compiled_linear_region(
                spec,
                body,
                index,
                env,
                local_names,
                backend,
            )
            if step is not None:
                returned, result = step.simulate(env)
                steps.append(step)
                for name in step.output_names:
                    if name not in local_names:
                        local_names.append(name)
                if returned:
                    return ResumeExecutionPlan(spec, steps)
                index = end_index
                continue

        step = _build_eager_step(spec, stmt, env, local_names)
        if step is None:
            return None
        returned, result = step.simulate(env)
        steps.append(step)

        for name in step.output_names:
            if name not in local_names:
                local_names.append(name)

        if returned and step.guaranteed_return:
            return ResumeExecutionPlan(spec, steps)
        index += 1

    return None
