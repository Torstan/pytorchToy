import torch

from torch._functorch.aot_autograd import aot_function

from test_utils import make_compiler


seen = {"fw": 0, "bw": 0, "inf": 0}


def fn(x, y):
    return torch.relu(x + y)


compiled = aot_function(
    fn,
    fw_compiler=make_compiler("fw", seen),
    bw_compiler=make_compiler("bw", seen),
    inference_compiler=make_compiler("inf", seen),
)

x = torch.randn(4, 4)
y = torch.randn(4, 4)

ref = fn(x, y)
out = compiled(x, y)
torch.testing.assert_close(out, ref)

out_again = compiled(x, y)
torch.testing.assert_close(out_again, ref)

assert seen == {"fw": 0, "bw": 0, "inf": 1}, seen

print("aot_autograd_inference_basic: ok")
