"""
最小显式 guard manager。

这一层把之前隐式的“签名 tuple 比较”收敛为：
- `Guard`: 单条 guard
- `GuardManager`: 一组 guard 的匹配与描述

目标不是复刻上游所有 guard 细节，而是把 cache / recompile / logging
建立在显式 guard 对象之上。
"""

from dataclasses import dataclass, field

from torch.tensor import Tensor


_SKIP = object()


def _callable_python_impl(fn):
    method_fn = getattr(fn, "__func__", None)
    if getattr(method_fn, "__code__", None) is not None:
        return method_fn

    if getattr(fn, "__code__", None) is not None:
        return fn

    forward = getattr(fn, "forward", None)
    if callable(forward):
        forward_fn = getattr(forward, "__func__", forward)
        if getattr(forward_fn, "__code__", None) is not None:
            return forward_fn

    call_fn = getattr(type(fn), "__call__", None)
    if call_fn is None:
        return None
    call_fn = getattr(call_fn, "__func__", call_fn)
    if getattr(call_fn, "__code__", None) is not None:
        return call_fn
    return None


def _callable_globals(fn):
    callable_fn = _callable_python_impl(fn)
    if callable_fn is None:
        return None
    return getattr(callable_fn, "__globals__", None)


def input_value_signature(value):
    if isinstance(value, Tensor):
        dtype = getattr(getattr(value, "_dtype", None), "name", repr(getattr(value, "_dtype", None)))
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
        return ("tuple", tuple(input_value_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(input_value_signature(item) for item in value))
    if isinstance(value, dict):
        items = tuple((key, input_value_signature(item)) for key, item in sorted(value.items()))
        return ("dict", items)
    return (type(value).__name__, id(value))


def _captured_value_signature(value):
    if isinstance(value, Tensor):
        return ("tensor_id", id(value))
    if isinstance(value, (int, float, str, bool, type(None))):
        return (type(value).__name__, value)
    if isinstance(value, tuple):
        items = []
        for item in value:
            signature = _captured_value_signature(item)
            if signature is _SKIP:
                return _SKIP
            items.append(signature)
        return ("tuple", tuple(items))
    if isinstance(value, list):
        items = []
        for item in value:
            signature = _captured_value_signature(item)
            if signature is _SKIP:
                return _SKIP
            items.append(signature)
        return ("list", tuple(items))
    if isinstance(value, dict):
        items = []
        for key, item in sorted(value.items()):
            key_signature = _captured_value_signature(key)
            value_signature = _captured_value_signature(item)
            if key_signature is _SKIP or value_signature is _SKIP:
                return _SKIP
            items.append((key_signature, value_signature))
        return ("dict", tuple(items))
    return (type(value).__name__, id(value))


def _closure_values(fn):
    callable_fn = _callable_python_impl(fn)
    code = getattr(callable_fn, "__code__", None)
    closure = getattr(callable_fn, "__closure__", None)
    if code is None or not closure:
        return {}

    values = {}
    for name, cell in zip(code.co_freevars, closure):
        try:
            values[name] = cell.cell_contents
        except ValueError:
            continue
    return values


def _resolve_module_path(fn, path):
    module = fn
    for name in path:
        module = getattr(module, name)
    return module


def _format_signature(signature, kind):
    if kind == "input":
        tag = signature[0]
        if tag == "tensor":
            _, shape, dtype, requires_grad, contiguous = signature
            return (
                f"tensor shape={shape} dtype={dtype} "
                f"requires_grad={requires_grad} contiguous={contiguous}"
            )
        if tag in ("int", "float", "str", "bool", "NoneType"):
            return f"{tag} value={signature[1]!r}"
        if tag == "tuple":
            return f"tuple len={len(signature[1])}"
        if tag == "list":
            return f"list len={len(signature[1])}"
        if tag == "dict":
            keys = [key for key, _value in signature[1]]
            return f"dict keys={keys!r}"
        return f"{tag} id={signature[1]}"

    tag = signature[0]
    if tag == "tensor_id":
        return f"tensor_id={signature[1]}"
    if tag in ("int", "float", "str", "bool", "NoneType"):
        return f"{tag} value={signature[1]!r}"
    if tag == "tuple":
        return f"tuple len={len(signature[1])}"
    if tag == "list":
        return f"list len={len(signature[1])}"
    if tag == "dict":
        return f"dict len={len(signature[1])}"
    return f"{tag} id={signature[1]}"


@dataclass(frozen=True)
class Source:
    kind: str
    key: object | None = None
    path: tuple[str, ...] = ()

    def get(self, args, kwargs, fn):
        if self.kind == "arg":
            return args[self.key]
        if self.kind == "kwarg":
            return kwargs[self.key]
        if self.kind == "global":
            return _callable_globals(fn)[self.key]
        if self.kind == "closure":
            return _closure_values(fn)[self.key]
        if self.kind == "callable_attr":
            return getattr(fn, self.key)
        if self.kind == "module_self":
            return _resolve_module_path(fn, self.path)
        module = _resolve_module_path(fn, self.path)
        if self.kind == "module_attr":
            return getattr(module, self.key)
        if self.kind == "module_param":
            return module._parameters[self.key]
        if self.kind == "module_buffer":
            return module._buffers[self.key]
        raise RuntimeError(f"unsupported guard source kind: {self.kind}")

    def label(self):
        if self.kind == "arg":
            return f"arg{self.key}"
        if self.kind == "kwarg":
            return f"kwarg[{self.key!r}]"
        if self.kind == "global":
            return f"global[{self.key}]"
        if self.kind == "closure":
            return f"closure[{self.key}]"
        if self.kind == "callable_attr":
            return f"callable_attr[{self.key}]"

        path = ".".join(self.path)
        if self.kind == "module_self":
            return f"module_id[{path or 'self'}]"
        prefix = f"{path}." if path else ""
        if self.kind == "module_attr":
            return f"module_attr[{prefix}{self.key}]"
        if self.kind == "module_param":
            return f"module_param[{prefix}{self.key}]"
        if self.kind == "module_buffer":
            return f"module_buffer[{prefix}{self.key}]"
        raise RuntimeError(f"unsupported guard source kind: {self.kind}")


@dataclass(frozen=True)
class Guard:
    source: Source
    expected: object
    kind: str

    def matches(self, args, kwargs, fn):
        try:
            value = self.source.get(args, kwargs, fn)
        except Exception:
            return False
        if self.kind == "input":
            return input_value_signature(value) == self.expected
        if self.kind == "captured":
            return _captured_value_signature(value) == self.expected
        raise RuntimeError(f"unsupported guard kind: {self.kind}")

    def describe(self):
        return f"{self.source.label()}: {_format_signature(self.expected, self.kind)}"


@dataclass
class GuardManager:
    guards: list[Guard] = field(default_factory=list)

    def matches(self, args, kwargs, fn):
        return all(guard.matches(args, kwargs, fn) for guard in self.guards)

    def describe(self):
        return [guard.describe() for guard in self.guards]


def _build_input_guards(args, kwargs):
    guards = []
    for index, value in enumerate(args):
        guards.append(
            Guard(
                source=Source("arg", key=index),
                expected=input_value_signature(value),
                kind="input",
            )
        )
    for key, value in sorted(kwargs.items()):
        guards.append(
            Guard(
                source=Source("kwarg", key=key),
                expected=input_value_signature(value),
                kind="input",
            )
        )
    return guards


def _build_global_guards(fn):
    callable_fn = _callable_python_impl(fn)
    code = getattr(callable_fn, "__code__", None)
    globals_dict = _callable_globals(fn)
    if code is None or globals_dict is None:
        return []

    guards = []
    for name in sorted(code.co_names):
        if name not in globals_dict:
            continue
        signature = _captured_value_signature(globals_dict[name])
        if signature is _SKIP:
            continue
        guards.append(
            Guard(
                source=Source("global", key=name),
                expected=signature,
                kind="captured",
            )
        )
    return guards


def _build_closure_guards(fn):
    guards = []
    for name, value in sorted(_closure_values(fn).items()):
        signature = _captured_value_signature(value)
        if signature is _SKIP:
            continue
        guards.append(
            Guard(
                source=Source("closure", key=name),
                expected=signature,
                kind="captured",
            )
        )
    return guards


def _append_module_guards(guards, module, path=(), seen=None):
    seen = seen or set()
    module_id = id(module)

    guards.append(
        Guard(
            source=Source("module_self", path=path),
            expected=(type(module).__name__, module_id),
            kind="captured",
        )
    )

    if module_id in seen:
        return
    seen.add(module_id)

    for name, value in sorted(module.__dict__.items()):
        if name in ("_parameters", "_buffers", "_modules"):
            continue
        signature = _captured_value_signature(value)
        if signature is _SKIP:
            continue
        guards.append(
            Guard(
                source=Source("module_attr", key=name, path=path),
                expected=signature,
                kind="captured",
            )
        )

    for name, value in sorted(module.__dict__.get("_parameters", {}).items()):
        if value is None:
            continue
        guards.append(
            Guard(
                source=Source("module_param", key=name, path=path),
                expected=("tensor_id", id(value)),
                kind="captured",
            )
        )

    for name, value in sorted(module.__dict__.get("_buffers", {}).items()):
        if value is None:
            continue
        guards.append(
            Guard(
                source=Source("module_buffer", key=name, path=path),
                expected=("tensor_id", id(value)),
                kind="captured",
            )
        )

    for name, value in sorted(module.__dict__.get("_modules", {}).items()):
        if value is None:
            continue
        _append_module_guards(guards, value, path=path + (name,), seen=seen)


def _build_module_guards(fn):
    if not (hasattr(fn, "__dict__") and "_modules" in fn.__dict__):
        return []
    guards = []
    _append_module_guards(guards, fn)
    return guards


def _build_callable_object_guards(fn):
    if getattr(fn, "__code__", None) is not None:
        return []
    if getattr(fn, "__func__", None) is not None:
        return []
    if not hasattr(fn, "__dict__"):
        return []
    if "_modules" in fn.__dict__:
        return []
    if _callable_python_impl(fn) is None:
        return []

    guards = []
    for name, value in sorted(fn.__dict__.items()):
        signature = _captured_value_signature(value)
        if signature is _SKIP:
            continue
        guards.append(
            Guard(
                source=Source("callable_attr", key=name),
                expected=signature,
                kind="captured",
            )
        )
    return guards


def build_guard_manager(fn, args, kwargs):
    guards = []
    guards.extend(_build_input_guards(args, kwargs))
    guards.extend(_build_global_guards(fn))
    guards.extend(_build_closure_guards(fn))
    guards.extend(_build_module_guards(fn))
    guards.extend(_build_callable_object_guards(fn))
    return GuardManager(guards)


def callable_guard_signature(fn):
    manager = build_guard_manager(fn, (), {})
    return tuple((guard.source.label(), guard.expected, guard.kind) for guard in manager.guards if guard.kind == "captured")


def describe_callable_guards(fn):
    manager = build_guard_manager(fn, (), {})
    return [guard.describe() for guard in manager.guards if guard.kind == "captured"]
