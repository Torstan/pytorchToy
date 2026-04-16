import torch

from torch._compile.tracer import Tracer
from torch._inductor.decomposition import decompose_graph_module, select_decomp_table


def fn(x, y):
    return torch.relu(x + y)


x = torch.randn(4, 4)
y = torch.randn(4, 4)

gm = Tracer().trace(fn, (x, y))
decomposed = decompose_graph_module(gm, select_decomp_table())

targets = [node.target for node in decomposed.graph.nodes if node.op == "call_function"]
assert len(targets) == 2
assert all(callable(target) for target in targets)
assert targets[0].__module__ == "torch._prims"
assert targets[1].__module__ == "torch._refs"

torch.testing.assert_close(decomposed(x, y), fn(x, y))

print("prims_decomposition_basic: ok")
