import torch
import torch.nn.functional as F

from torch._compile.tracer import Tracer


def fn(x, weight, bias):
    return F.linear(x, weight, bias)


x = torch.randn(4, 3)
weight = torch.randn(5, 3)
bias = torch.randn(5)

graph_module = Tracer().trace(fn, (x, weight, bias))
targets = [node.target for node in graph_module.graph.nodes if node.op == "call_function"]

assert targets == ["t", "mm", "add"], targets
torch.testing.assert_close(graph_module(x, weight, bias), fn(x, weight, bias))

print("prims_linear_lowering_input: ok")
