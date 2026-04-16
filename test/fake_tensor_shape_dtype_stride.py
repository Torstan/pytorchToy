import torch
from torch._subclasses.fake_tensor import FakeTensor


fake = FakeTensor((2, 3, 4), dtype=torch.float32, requires_grad=True)
view = fake.view(6, 4)
reduced = fake.sum(dim=1, keepdim=True)

assert fake.stride() == (12, 4, 1)
assert view.shape == (6, 4)
assert reduced.shape == (2, 1, 4)
assert reduced.requires_grad is True

print("fake_tensor_shape_dtype_stride: ok")
