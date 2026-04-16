import torch
from torch.fx import Graph, GraphModule, propagate_meta


graph = Graph()
x = graph.placeholder("x")
y = graph.placeholder("y")
sin = graph.call_function("sin", (x,))
add = graph.call_function("add", (sin, y))
summed = graph.call_function("sum", (add,), {"dim": 1, "keepdim": True})
graph.output(summed)

gm = GraphModule(graph)
example_x = torch.randn(2, 3)
example_y = torch.randn(2, 3)
propagate_meta(gm, (example_x, example_y))

assert x.meta["val"].shape == (2, 3)
assert sin.meta["val"].shape == (2, 3)
assert add.meta["val"].shape == (2, 3)
assert summed.meta["val"].shape == (2, 1)
assert summed.meta["val"].dtype == torch.float32

print("fx_graph_meta_prop: ok")
