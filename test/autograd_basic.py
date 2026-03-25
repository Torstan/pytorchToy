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
