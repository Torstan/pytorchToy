import torch

from torch._compile.tracer import Tracer
from torch._inductor.decomposition import decompose_graph_module, select_decomp_table


def fn(x, y):
    return torch.relu(torch.tanh(x + y))


x = torch.randn(4, 4)
y = torch.randn(4, 4)

gm = Tracer().trace(fn, (x, y))
decomposed = decompose_graph_module(gm, select_decomp_table())

allowed_modules = {"torch._prims", "torch._refs"}
for node in decomposed.graph.nodes:
    if node.op != "call_function":
        continue
    assert callable(node.target)
    assert node.target.__module__ in allowed_modules, node.target

torch.testing.assert_close(decomposed(x, y), fn(x, y))

print("prims_backend_contract: ok")
