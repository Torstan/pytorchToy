"""
最小 guard 快照工具。

这一层不实现完整的 Guard 对象求值，只负责把当前 callable
依赖的外部状态编码成可比较的签名，用于触发重编译。
"""

from torch.tensor import Tensor


_SKIP = object()


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
    return _SKIP


def _module_state_signature(module, seen=None):
    seen = seen or set()
    module_id = id(module)
    if module_id in seen:
        return ("module_ref", module_id)
    seen.add(module_id)

    attrs = []
    for name, value in sorted(module.__dict__.items()):
        if name in ("_parameters", "_buffers", "_modules"):
            continue
        signature = _captured_value_signature(value)
        if signature is not _SKIP:
            attrs.append((name, signature))

    params = []
    for name, value in sorted(module.__dict__.get("_parameters", {}).items()):
        if value is not None:
            params.append((name, ("tensor_id", id(value))))

    buffers = []
    for name, value in sorted(module.__dict__.get("_buffers", {}).items()):
        if value is not None:
            buffers.append((name, ("tensor_id", id(value))))

    submodules = []
    for name, value in sorted(module.__dict__.get("_modules", {}).items()):
        if value is not None:
            submodules.append((name, _module_state_signature(value, seen)))

    return (
        "module_state",
        tuple(attrs),
        tuple(params),
        tuple(buffers),
        tuple(submodules),
    )


def _global_state_signature(fn):
    code = getattr(fn, "__code__", None)
    globals_dict = getattr(fn, "__globals__", None)
    if code is None or globals_dict is None:
        return ()

    items = []
    for name in sorted(code.co_names):
        if name not in globals_dict:
            continue
        signature = _captured_value_signature(globals_dict[name])
        if signature is not _SKIP:
            items.append((name, signature))
    return tuple(items)


def _closure_state_signature(fn):
    code = getattr(fn, "__code__", None)
    closure = getattr(fn, "__closure__", None)
    if code is None or not closure:
        return ()

    items = []
    for name, cell in zip(code.co_freevars, closure):
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        signature = _captured_value_signature(value)
        if signature is not _SKIP:
            items.append((name, signature))
    return tuple(items)


def callable_guard_signature(fn):
    return (
        _global_state_signature(fn),
        _closure_state_signature(fn),
        _module_state_signature(fn) if hasattr(fn, "__dict__") and "_modules" in fn.__dict__ else None,
    )


def describe_callable_guards(fn):
    global_state, closure_state, module_state = callable_guard_signature(fn)
    lines = []

    for name, signature in global_state:
        lines.append(f"global[{name}]: {signature!r}")

    for name, signature in closure_state:
        lines.append(f"closure[{name}]: {signature!r}")

    if module_state is not None:
        lines.append(f"module_state: {module_state!r}")

    return lines
