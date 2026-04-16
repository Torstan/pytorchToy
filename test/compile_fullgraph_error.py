import torch

from torch._compile.tracer import UnsupportedTraceError


def demo(x):
    if x.item() > 0:
        return torch.sin(x)
    return torch.cos(x)


compiled = torch.compile(demo, backend="eager", fullgraph=True)

try:
    compiled(torch.tensor(1.0))
except UnsupportedTraceError:
    print("compile_fullgraph_error: ok")
else:
    raise AssertionError("expected fullgraph=True to raise on graph break")
