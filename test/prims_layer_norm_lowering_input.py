import torch
import torch.nn.functional as F

from torch.fx import Tracer
from torch._inductor.decomposition import decompose_graph_module, select_decomp_table


def fn(x, weight, bias):
    return F.layer_norm(x, weight, bias)


x = torch.randn(4, 5)
weight = torch.ones(5)
bias = torch.zeros(5)

graph_module = Tracer().trace(fn, (x, weight, bias))
targets = [node.target for node in graph_module.graph.nodes if node.op == "call_function"]
assert targets == ["layer_norm"], targets
torch.testing.assert_close(graph_module(x, weight, bias), fn(x, weight, bias))

decomposed = decompose_graph_module(graph_module, select_decomp_table())
call_targets = [node.target for node in decomposed.graph.nodes if node.op == "call_function"]
assert len(call_targets) == 1
assert callable(call_targets[0])
assert call_targets[0].__module__ == "torch._refs", call_targets[0]
torch.testing.assert_close(decomposed(x, weight, bias), fn(x, weight, bias))

print("prims_layer_norm_lowering_input: ok")
