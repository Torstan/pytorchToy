#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import _C
import _nn_C

from torch.tensor import Tensor


def assert_close(name, a, b, atol=1e-5):
    if not np.allclose(a, b, atol=atol):
        print(f"{name}: mismatch")
        print(a)
        print(b)
        raise SystemExit(1)
    print(f"{name}: ok")


linear = nn.Linear(4, 3)
x = torch.randn(2, 4)

y = linear(x)
cache_key_before = linear._weight_t_cache_key
if linear._weight_t_cache is None:
    print("cache was not populated")
    raise SystemExit(1)

ref = Tensor(_nn_C.linear_forward(x._c, linear.weight._c,
                                  linear.bias._c if linear.bias is not None else _C.empty([1]),
                                  linear.bias is not None))
assert_close("forward", y.numpy(), ref.numpy())

opt = optim.Adam(linear.parameters(), lr=0.01)
opt.zero_grad()
y.sum().backward()
opt.step()

y2 = linear(x)
cache_key_after = linear._weight_t_cache_key
if cache_key_after == cache_key_before:
    print("cache key did not refresh after optimizer step")
    raise SystemExit(1)

ref2 = Tensor(_nn_C.linear_forward(x._c, linear.weight._c,
                                   linear.bias._c if linear.bias is not None else _C.empty([1]),
                                   linear.bias is not None))
assert_close("forward_after_step", y2.numpy(), ref2.numpy())

print("linear cache test passed")
