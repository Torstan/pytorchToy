#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn


def check_grad(name, param):
    grad = param.grad
    if grad is None:
        print(f"{name}: grad is None")
        return False
    grad_sum = float(np.abs(grad.numpy()).sum())
    print(f"{name}: |grad| sum = {grad_sum:.6f}")
    return grad_sum > 0.0


rnn = nn.RNN(3, 4, batch_first=True)
linear = nn.Linear(4, 2)
x = torch.randn(2, 5, 3)

out, _ = rnn(x)
loss = linear(out).sum()
loss.backward()

checks = []
for name, param in rnn.named_parameters():
    checks.append(check_grad(f"rnn.{name}", param))
for name, param in linear.named_parameters():
    checks.append(check_grad(f"linear.{name}", param))

if not all(checks):
    raise SystemExit(1)

print("mixed autograd test passed")
