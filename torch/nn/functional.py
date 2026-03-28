"""
nn.functional — 函数式接口
所有 autograd 在 C++ 侧完成，Python 零开销
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import _C

from torch.tensor import Tensor


def linear(input, weight, bias=None, packed_weight=None):
    return Tensor(_C.autograd_linear(input._c, weight._c,
                  bias._c if bias is not None else None,
                  packed_weight._c if packed_weight is not None else None))


def softmax(input, dim=-1):
    return Tensor(_C.autograd_softmax(input._c, dim))


def log_softmax(input, dim=-1):
    return Tensor(_C.autograd_log_softmax(input._c, dim))


def nll_loss(log_probs, target):
    return Tensor(_C.autograd_nll_loss(log_probs._c, target._c))


def cross_entropy(input, target):
    return Tensor(_C.autograd_cross_entropy(input._c, target._c))


def relu(input):
    return input.relu()


def tanh(input):
    return input.tanh()
