import torch

from torch.fx import Tracer
from torch._inductor import compile_fx


def fn(x, y):
    return torch.relu(torch.sin(x) + torch.cos(y))


x = torch.randn(4, 4)
y = torch.randn(4, 4)

gm = Tracer().trace(fn, (x, y))
compiled = compile_fx(gm, [x, y])

torch.testing.assert_close(compiled(x, y), fn(x, y))

print("inductor_compile_fx_basic: ok")
