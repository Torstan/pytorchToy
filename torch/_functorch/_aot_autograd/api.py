"""
Public AOTAutograd function API implementation.
"""

from torch.fx import Tracer

from .backward_graph import build_backward_graph_or_stub
from .runtime import attach_compiled_backward
from .utils import (
    AOTCompileState,
    any_requires_grad,
    assert_no_input_mutation,
    call_signature,
    differentiable_input_indices,
)

class AOTFunction:
    def __init__(
        self,
        fn,
        *,
        fw_compiler,
        bw_compiler=None,
        inference_compiler=None,
    ):
        self._fn = fn
        self._fw_compiler = fw_compiler
        self._bw_compiler = bw_compiler or fw_compiler
        self._inference_compiler = inference_compiler or fw_compiler
        self._cache = {}
        self._last_state = None

    def __call__(self, *args, **kwargs):
        key = call_signature(args, kwargs)
        compiled = self._cache.get(key)
        if compiled is None:
            compiled = self._compile(args, kwargs)
            self._cache[key] = compiled
        return compiled(*args, **kwargs)

    def _compile(self, args, kwargs):
        if kwargs:
            raise NotImplementedError("AOTAutograd-mini does not support kwargs yet")

        assert_no_input_mutation(self._fn, args, kwargs)

        tracer = Tracer()
        graph_module = tracer.trace(self._fn, args)
        requires_grad = any_requires_grad(args, kwargs)
        diff_input_indices = ()

        if requires_grad:
            diff_input_indices = differentiable_input_indices(args)
            backward_graph_module, backward_example_inputs, backward_is_real = build_backward_graph_or_stub(
                graph_module,
                list(args),
            )
            if backward_is_real:
                compiled_fw = self._fw_compiler(graph_module, list(args))
                compiled_bw = self._bw_compiler(
                    backward_graph_module,
                    list(backward_example_inputs),
                )
            else:
                compiled_fw = self._fn
                compiled_bw = None
        else:
            compiled_fw = self._inference_compiler(graph_module, list(args))
            backward_graph_module = None
            compiled_bw = None
            backward_example_inputs = None
            backward_is_real = False

        self._last_state = AOTCompileState(
            graph_module=graph_module,
            backward_graph_module=backward_graph_module,
            compiled_fw=compiled_fw,
            compiled_bw=compiled_bw,
            requires_grad=requires_grad,
            backward_example_inputs=backward_example_inputs,
            backward_is_real=backward_is_real,
        )

        if requires_grad and backward_is_real:
            def compiled_with_runtime_backward(*runtime_args, **runtime_kwargs):
                if runtime_kwargs:
                    raise NotImplementedError("AOTAutograd-mini does not support kwargs yet")
                output = compiled_fw(*runtime_args)
                return attach_compiled_backward(
                    output,
                    runtime_args,
                    compiled_bw,
                    diff_input_indices,
                )

            return compiled_with_runtime_backward

        return compiled_fw


def aot_function(
    fn,
    *,
    fw_compiler,
    bw_compiler=None,
    inference_compiler=None,
):
    return AOTFunction(
        fn,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        inference_compiler=inference_compiler,
    )


def make_boxed_func(fn):
    return fn
