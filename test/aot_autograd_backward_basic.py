import torch

from torch._functorch.aot_autograd import aot_function

from test_utils import make_compiler


seen = {"fw": 0, "bw": 0}


def fn(x, y):
    return (torch.tanh(x) * y).sum()


compiled = aot_function(
    fn,
    fw_compiler=make_compiler("fw", seen),
    bw_compiler=make_compiler("bw", seen),
)

x = torch.tensor([1.0, -2.0])
y = torch.tensor([3.0, 4.0])
x.requires_grad = True
y.requires_grad = True

x_ref = torch.tensor([1.0, -2.0])
y_ref = torch.tensor([3.0, 4.0])
x_ref.requires_grad = True
y_ref.requires_grad = True

ref = fn(x_ref, y_ref)
ref.backward()

out = compiled(x, y)
out.backward()

torch.testing.assert_close(out, ref)
torch.testing.assert_close(x.grad, x_ref.grad)
torch.testing.assert_close(y.grad, y_ref.grad)

assert seen == {"fw": 1, "bw": 1}, seen

print("aot_autograd_backward_basic: ok")
