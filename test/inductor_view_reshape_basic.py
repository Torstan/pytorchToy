import torch


def fn(x):
    reshaped = x.reshape(2, 6)
    return torch.tanh(reshaped).sum()


compiled = torch.compile(fn, backend="inductor")

x = torch.randn(3, 4)

ref = fn(x)
out = compiled(x)

torch.testing.assert_close(out, ref)

print("inductor_view_reshape_basic: ok")
