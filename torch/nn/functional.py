"""
nn.functional — 函数式接口
提供不含可训练参数的 nn 操作
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import _nn_C

from torch.tensor import Tensor


def softmax(input, dim=-1):
    return Tensor(_nn_C.softmax(input._c, dim))


def log_softmax(input, dim=-1):
    return Tensor(_nn_C.log_softmax(input._c, dim))


def relu(input):
    return Tensor(_nn_C.elementwise_relu(input._c))


def tanh(input):
    return Tensor(_nn_C.elementwise_tanh(input._c))
