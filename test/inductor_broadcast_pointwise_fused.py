import torch

from torch._compile.pointwise import CppPointwiseKernel, NativePointwiseKernel, compile_graph_module
from torch.fx import Tracer


def fn(x, b):
    return torch.tanh(torch.relu(x + b))


x = torch.randn(4, 5)
b = torch.randn(5)

graph_module = Tracer().trace(fn, (x, b))
compiled = compile_graph_module(graph_module, [x, b])

assert isinstance(compiled, (NativePointwiseKernel, CppPointwiseKernel)), type(compiled)
torch.testing.assert_close(compiled.run((x, b)), fn(x, b))

public = torch.compile(fn, backend="inductor", fullgraph=True)
torch.testing.assert_close(public(x, b), fn(x, b))

print("inductor_broadcast_pointwise_fused: ok")
