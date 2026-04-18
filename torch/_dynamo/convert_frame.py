"""
最小 convert_frame 入口。

这一层先不实现真正的 frame-eval hook，
只提供“按 code object 做最小 bytecode symbolic capture”的桥接。
"""

from torch._compile.tracer import UnsupportedTraceError

from .symbolic_convert import convert_callable_to_graph


def convert_frame(fn, example_inputs, *, graph_input_positions=None):
    if not example_inputs:
        raise UnsupportedTraceError("bytecode capture requires at least one runtime input")
    return convert_callable_to_graph(
        fn,
        example_inputs,
        graph_input_positions=graph_input_positions,
    )
