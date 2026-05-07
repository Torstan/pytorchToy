"""
Runtime attachment for compiled AOTAutograd backward functions.
"""

def _unwrap_backward_result(value):
    from torch.tensor import Tensor

    if isinstance(value, Tensor):
        return (value._c,)
    if isinstance(value, tuple):
        return tuple(item._c if isinstance(item, Tensor) else item for item in value)
    if isinstance(value, list):
        return tuple(item._c if isinstance(item, Tensor) else item for item in value)
    return (value,)


class _CompiledBackwardRuntime:
    def __init__(self, compiled_bw, forward_inputs):
        self._compiled_bw = compiled_bw
        self._forward_inputs = tuple(forward_inputs)

    def _do_backward(self, grad_outputs_list):
        import _C
        import torch
        from torch.tensor import Tensor

        grad_outputs = []
        for grad in grad_outputs_list:
            if isinstance(grad, _C.Tensor):
                grad_outputs.append(Tensor(grad))
            else:
                grad_outputs.append(grad)

        with torch.no_grad():
            result = self._compiled_bw(*self._forward_inputs, *grad_outputs)
        return _unwrap_backward_result(result)


def _attach_compiled_backward(output, args, compiled_bw, differentiable_input_indices):
    import _C
    from torch.tensor import Tensor

    if not differentiable_input_indices:
        return output
    if not isinstance(output, Tensor):
        raise NotImplementedError(
            "AOTAutograd-mini runtime compiled backward only supports Tensor outputs"
        )

    py_fn = _C.PyFunction(_CompiledBackwardRuntime(compiled_bw, args))
    py_fn.num_inputs = len(differentiable_input_indices)
    py_fn.requires_grad = True

    for index in differentiable_input_indices:
        inp = args[index]
        if inp._grad_fn is not None:
            py_fn.add_previous_function(inp._grad_fn, inp._output_index)
        else:
            py_fn.add_leaf_tensor(inp._c)

    output._set_creator(py_fn)
    return output



def attach_compiled_backward(output, args, compiled_bw, differentiable_input_indices):
    return _attach_compiled_backward(
        output,
        args,
        compiled_bw,
        differentiable_input_indices,
    )
