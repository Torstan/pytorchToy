import torch


def fn(x, w, b):
    return torch.relu(x.mm(w) + b)


compiled = torch.compile(fn, backend="inductor")

x = torch.randn(4, 3)
w = torch.randn(3, 5)
b = torch.randn(5)

ref = fn(x, w, b)
out = compiled(x, w, b)

torch.testing.assert_close(out, ref)

print("inductor_linear_relu: ok")
