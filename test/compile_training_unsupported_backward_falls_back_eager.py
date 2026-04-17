import torch


def fn(x):
    return torch.tanh(x).reshape(2, 2)


compiled = torch.compile(fn, backend="inductor")

x = torch.randn(4)
x.requires_grad = True

x_ref = x.clone()
x_ref.requires_grad = True

out = compiled(x)
ref = fn(x_ref)

assert out.requires_grad, "expected unsupported backward path to preserve eager autograd"

out.sum().backward()
ref.sum().backward()

torch.testing.assert_close(out, ref)
torch.testing.assert_close(x.grad, x_ref.grad)

print("compile_training_unsupported_backward_falls_back_eager: ok")
