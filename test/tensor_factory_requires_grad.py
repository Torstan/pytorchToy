import torch


x = torch.randn(10, requires_grad=True)
y = torch.zeros(3, requires_grad=True)
z = torch.ones(2, requires_grad=True)
w = torch.tensor([1.0, 2.0], requires_grad=True)

assert x.requires_grad is True
assert y.requires_grad is True
assert z.requires_grad is True
assert w.requires_grad is True

print("tensor_factory_requires_grad: ok")
