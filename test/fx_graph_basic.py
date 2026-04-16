from torch.fx import Graph, GraphModule
import torch


graph = Graph()
x = graph.placeholder("x")
y = graph.placeholder("y")
sin = graph.call_function("sin", (x,))
cos = graph.call_function("cos", (y,))
out = graph.call_function("add", (sin, cos))
graph.output(out)

module = GraphModule(graph)
result = module(torch.randn(4, 4), torch.randn(4, 4))

assert isinstance(result, torch.Tensor)
assert result.shape == (4, 4)
assert "torch.sin" in module.print_readable(print_output=False)

print("fx_graph_basic: ok")
