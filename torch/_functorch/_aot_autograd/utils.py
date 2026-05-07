"""
AOTAutograd utility helpers.
"""

from dataclasses import dataclass

@dataclass
class AOTCompileState:
    graph_module: object
    backward_graph_module: object
    compiled_fw: object
    compiled_bw: object
    requires_grad: bool
    backward_example_inputs: object = None
    backward_is_real: bool = False


def value_signature(value):
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
        return ("tuple", tuple(value_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(value_signature(item) for item in value))
    if isinstance(value, dict):
        items = tuple((key, value_signature(item)) for key, item in sorted(value.items()))
        return ("dict", items)
    return (type(value).__name__, id(value))


def call_signature(args, kwargs):
    return (
        tuple(value_signature(arg) for arg in args),
        tuple((key, value_signature(value)) for key, value in sorted(kwargs.items())),
    )


def clone_structure(value):
    from torch.tensor import Tensor

    if isinstance(value, Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_structure(item) for item in value)
    if isinstance(value, list):
        return [clone_structure(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_structure(item) for key, item in value.items()}
    return value


def collect_tensor_versions(value, versions):
    from torch.tensor import Tensor

    if isinstance(value, Tensor):
        versions[id(value)] = value._version
        return
    if isinstance(value, tuple):
        for item in value:
            collect_tensor_versions(item, versions)
        return
    if isinstance(value, list):
        for item in value:
            collect_tensor_versions(item, versions)
        return
    if isinstance(value, dict):
        for item in value.values():
            collect_tensor_versions(item, versions)


def any_requires_grad(args, kwargs):
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


def differentiable_input_indices(args):
    from torch.tensor import Tensor

    indices = []
    for index, value in enumerate(args):
        if isinstance(value, Tensor) and value.requires_grad:
            indices.append(index)
    return tuple(indices)


def assert_no_input_mutation(fn, args, kwargs):
    cloned_args = tuple(clone_structure(arg) for arg in args)
    cloned_kwargs = {key: clone_structure(value) for key, value in kwargs.items()}
    before = {}
    collect_tensor_versions(cloned_args, before)
    collect_tensor_versions(cloned_kwargs, before)
    fn(*cloned_args, **cloned_kwargs)
    after = {}
    collect_tensor_versions(cloned_args, after)
    collect_tensor_versions(cloned_kwargs, after)
    if before != after:
        raise NotImplementedError(
            "AOTAutograd-mini does not support input mutation yet"
        )

_value_signature = value_signature
_call_signature = call_signature
_clone_structure = clone_structure
_collect_tensor_versions = collect_tensor_versions
_any_requires_grad = any_requires_grad
_differentiable_input_indices = differentiable_input_indices
_assert_no_input_mutation = assert_no_input_mutation
