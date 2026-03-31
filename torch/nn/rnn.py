"""
nn.RNN — 循环神经网络层
h_t = tanh(x_t @ W_ih^T + h_{t-1} @ W_hh^T + b_ih + b_hh)

Python 侧只做参数组织和模块封装，建图与 backward 统一走 C++ autograd。
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import _C
import _nn_C

from torch.tensor import Tensor
from torch.nn.module import Module
from torch.nn.parameter import Parameter


class RNN(Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # 参数初始化 (uniform(-k, k), k = 1/sqrt(hidden_size))
        k = 1.0 / math.sqrt(hidden_size)

        w_ih = _C.empty([hidden_size, input_size])
        _nn_C.fill_uniform(w_ih, -k, k)
        self.weight_ih_l0 = Parameter(Tensor(w_ih))

        w_hh = _C.empty([hidden_size, hidden_size])
        _nn_C.fill_uniform(w_hh, -k, k)
        self.weight_hh_l0 = Parameter(Tensor(w_hh))

        b_ih = _C.empty([hidden_size])
        _nn_C.fill_uniform(b_ih, -k, k)
        self.bias_ih_l0 = Parameter(Tensor(b_ih))

        b_hh = _C.empty([hidden_size])
        _nn_C.fill_uniform(b_hh, -k, k)
        self.bias_hh_l0 = Parameter(Tensor(b_hh))

    def forward(self, input, hidden=None):
        has_hidden = hidden is not None
        hidden_c = hidden._c if has_hidden else _C.empty([1])

        output_c, h_n_c = _C.autograd_rnn(
            input._c, hidden_c, has_hidden,
            self.weight_ih_l0._c, self.weight_hh_l0._c,
            self.bias_ih_l0._c, self.bias_hh_l0._c,
            self.batch_first)

        return Tensor(output_c), Tensor(h_n_c)

    def __repr__(self):
        return (f"RNN(input_size={self.input_size}, "
                f"hidden_size={self.hidden_size}, "
                f"batch_first={self.batch_first})")
