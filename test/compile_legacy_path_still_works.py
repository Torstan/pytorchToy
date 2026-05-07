import torch

from torch._inductor.lowering.partition import compile_graph_module
from torch.fx import Tracer


def legacy_demo(x, y):
    return torch.relu(torch.sin(x) + torch.cos(y))


x = torch.randn(8, 8)
y = torch.randn(8, 8)
graph_module = Tracer().trace(legacy_demo, (x, y))
compiled = compile_graph_module(graph_module, [x, y])

ref = legacy_demo(x, y)
out = compiled.run((x, y))
torch.testing.assert_close(out, ref)

print("compile_legacy_path_still_works: ok")
