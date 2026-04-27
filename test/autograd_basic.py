#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable, Function

a = Variable(torch.FloatTensor([2.0, 3.0]), requires_grad=True)
b = Variable(torch.FloatTensor([4.0, 5.0]), requires_grad=True)
p = a
c = p * b        # c = [8.0, 15.0]
y = c.sum()      # y = 23.0
y.backward()
print(f'a.grad = {a.grad}')
print(f'b.grad = {b.grad}')
print(f'c = {c}')
print(f'y = {y}')

x = Variable(torch.FloatTensor([1.0, 2.0]), requires_grad=True)
z = Variable(torch.FloatTensor([3.0, 4.0]), requires_grad=True)
s = (x + z).sum()
s.backward()
assert x.grad.data[0] == 1.0 and x.grad.data[1] == 1.0
assert z.grad.data[0] == 1.0 and z.grad.data[1] == 1.0

mask = torch.tensor([1.0, 3.0]).gt(torch.tensor([2.0, 2.0]))
assert mask[0] == 0.0 and mask[1] == 1.0
