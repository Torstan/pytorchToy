"""
最小 torch._inductor.compile_fx 总控。
"""

from torch._compile.pointwise import PointwiseLoweringError, compile_graph_module
from torch._functorch.aot_autograd import aot_module_simplified

from .decomposition import decompose_graph_module, select_decomp_table


def _eager_graph_compiler(graph_module, example_inputs):
    del example_inputs

    def compiled(*args):
        return graph_module(*args)

    return compiled


def _inference_graph_compiler(graph_module, example_inputs):
    try:
        compiled_program = compile_graph_module(graph_module, example_inputs)
    except PointwiseLoweringError:
        return _eager_graph_compiler(graph_module, example_inputs)

    def compiled(*args):
        return compiled_program.run(args)

    return compiled


def _needs_autograd(example_inputs):
    from torch.tensor import Tensor

    for value in example_inputs:
        if isinstance(value, Tensor) and value.requires_grad:
            return True
    return False


def compile_fx(graph_module, example_inputs):
    decomposed = decompose_graph_module(
        graph_module,
        decomposition_table=select_decomp_table(),
    )
    decomposed.propagate_meta(example_inputs)

    if _needs_autograd(example_inputs):
        return aot_module_simplified(
            decomposed,
            example_inputs,
            fw_compiler=_eager_graph_compiler,
            bw_compiler=_eager_graph_compiler,
            inference_compiler=_inference_graph_compiler,
        )

    return _inference_graph_compiler(decomposed, example_inputs)
