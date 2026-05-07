import torch

from torch._compile.pointwise import CompiledGraph, CompiledRegion, compile_graph_module
from torch.fx import Tracer


def mixed_region_demo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    c = torch.relu(a + b)
    d = c.sum()
    return (d + 1.0).tanh()


x = torch.randn(32, 32)
y = torch.randn(32, 32)

graph_module = Tracer().trace(mixed_region_demo, (x, y))
compiled_graph = compile_graph_module(graph_module, [x, y])

assert isinstance(compiled_graph, CompiledGraph)
region_count = sum(isinstance(step, CompiledRegion) for step in compiled_graph.steps)
assert region_count == 2, f"expected 2 compiled regions, got {region_count}"
assert any(
    getattr(step, "target", None) == "sum" for step in compiled_graph.steps
), "expected the unsupported sum node to stay in the mixed executor"

ref = mixed_region_demo(x, y)
out = compiled_graph.run((x, y))
torch.testing.assert_close(out, ref)

compiled_demo = torch.compile(mixed_region_demo, backend="inductor", fullgraph=True)
public_out = compiled_demo(x, y)
torch.testing.assert_close(public_out, ref)

print("compile_partitioned_graph: ok")
