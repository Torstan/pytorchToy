import torch


def fn(x, w, b):
    return torch.relu(x.mm(w) + b).sum()


compiled = torch.compile(fn, backend="inductor")

x = torch.randn(4, 3)
w = torch.randn(3, 5)
b = torch.randn(5)
x.requires_grad = True
w.requires_grad = True
b.requires_grad = True

x_ref = x.clone()
w_ref = w.clone()
b_ref = b.clone()
x_ref.requires_grad = True
w_ref.requires_grad = True
b_ref.requires_grad = True

ref = fn(x_ref, w_ref, b_ref)
ref.backward()

out = compiled(x, w, b)
out.backward()

torch.testing.assert_close(out, ref)
torch.testing.assert_close(x.grad, x_ref.grad)
torch.testing.assert_close(w.grad, w_ref.grad)
torch.testing.assert_close(b.grad, b_ref.grad)

print("compile_training_linear_aot_inductor: ok")
