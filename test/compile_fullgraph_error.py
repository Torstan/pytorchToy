import torch
from torch.fx import UnsupportedTraceError


def demo(x):
    if x.sum().item() > 0:
        return x + 1
    return x - 1


compiled = torch.compile(demo, backend="eager", fullgraph=True)
x = torch.ones([2, 2])
try:
    compiled(x)
    raise AssertionError("expected UnsupportedTraceError")
except UnsupportedTraceError:
    pass
