"""
Module lifting helpers for AOTAutograd.
"""

from .api import aot_function

def aot_module_simplified(
    fn,
    example_inputs,
    *,
    fw_compiler,
    bw_compiler=None,
    inference_compiler=None,
):
    from torch.nn.module import Module

    if not isinstance(fn, Module):
        compiled = aot_function(
            fn,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            inference_compiler=inference_compiler,
        )
        compiled(*example_inputs)
        return compiled

    parameter_names = [name for name, _value in fn.named_parameters()]
    buffer_names = [name for name, _value in _named_buffers(fn)]
    lifted_names = parameter_names + buffer_names
    num_lifted = len(lifted_names)

    def lifted_fn(*flat_args):
        lifted_values = flat_args[:num_lifted]
        user_args = flat_args[num_lifted:]
        restore = []
        try:
            for name, value in zip(parameter_names, lifted_values[:len(parameter_names)]):
                restore.append(_swap_module_tensor(fn, name, value, is_buffer=False))
            for name, value in zip(buffer_names, lifted_values[len(parameter_names):]):
                restore.append(_swap_module_tensor(fn, name, value, is_buffer=True))
            return fn(*user_args)
        finally:
            for owner, attr_name, previous, is_buffer in reversed(restore):
                table = owner._buffers if is_buffer else owner._parameters
                table[attr_name] = previous

    compiled = aot_function(
        lifted_fn,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        inference_compiler=inference_compiler,
    )
    module_wrapper = _AOTModuleWrapper(fn, compiled, parameter_names, buffer_names)
    module_wrapper(*example_inputs)
    return module_wrapper




def _named_buffers(module, prefix=""):
    for name, buffer in module._buffers.items():
        if buffer is not None:
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, buffer
    for name, submodule in module._modules.items():
        if submodule is not None:
            sub_prefix = f"{prefix}.{name}" if prefix else name
            yield from _named_buffers(submodule, sub_prefix)


def _resolve_module_owner(module, qualified_name):
    owner = module
    parts = qualified_name.split(".")
    for part in parts[:-1]:
        owner = owner._modules[part]
    return owner, parts[-1]


def _swap_module_tensor(module, qualified_name, value, *, is_buffer):
    owner, name = _resolve_module_owner(module, qualified_name)
    table = owner._buffers if is_buffer else owner._parameters
    previous = table[name]
    table[name] = value
    return owner, name, previous, is_buffer


class _AOTModuleWrapper:
    def __init__(self, module, compiled, parameter_names, buffer_names):
        self._module = module
        self._compiled = compiled
        self._parameter_names = list(parameter_names)
        self._buffer_names = list(buffer_names)

    @property
    def _last_state(self):
        return self._compiled._last_state

    def __call__(self, *args, **kwargs):
        lifted_args = []
        for name in self._parameter_names:
            owner, attr_name = _resolve_module_owner(self._module, name)
            lifted_args.append(owner._parameters[attr_name])
        for name in self._buffer_names:
            owner, attr_name = _resolve_module_owner(self._module, name)
            lifted_args.append(owner._buffers[attr_name])
        return self._compiled(*lifted_args, *args, **kwargs)
