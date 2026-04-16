import torch
from torch._subclasses.fake_tensor import FakeTensor


x = torch.randn(2, 3)
fake = FakeTensor.from_tensor(x)

assert fake.shape == (2, 3)
assert fake.dtype == torch.float32
assert fake.device == "cpu"
assert fake.is_contiguous()
assert fake.stride() == x.stride()

print("fake_tensor_meta_basic: ok")
