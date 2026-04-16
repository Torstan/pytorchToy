import torch


def fn(x, y):
    return (torch.tanh(x) * y).sum()


compiled = torch.compile(fn, backend="inductor")

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

print("compile_training_aot_inductor_basic: ok")
