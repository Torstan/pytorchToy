"""
nn.Linear — 全连接层
y = x @ W^T + b
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
from torch.autograd_engine import record


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 参数初始化 (Kaiming uniform)
        k = 1.0 / math.sqrt(in_features)
        w = _C.empty([out_features, in_features])
        _nn_C.fill_uniform(w, -k, k)
        self.weight = Parameter(Tensor(w))

        if bias:
            b = _C.empty([out_features])
            _nn_C.fill_uniform(b, -k, k)
            self.bias = Parameter(Tensor(b))
        else:
            self.bias = None

    def forward(self, input):
        has_bias = self.bias is not None
        bias_c = self.bias._c if has_bias else _C.empty([1])
        output_c = _nn_C.linear_forward(input._c, self.weight._c, bias_c, has_bias)
        output = Tensor(output_c)

        # 记录到 tape
        saved_input = input
        saved_weight = self.weight
        saved_bias = self.bias

        def backward_fn(grad_outputs):
            grad_out = grad_outputs[0]
            gi_c, gw_c, gb_c = _nn_C.linear_backward(
                grad_out._c, saved_input._c, saved_weight._c)
            grads = [Tensor(gi_c), Tensor(gw_c)]
            if saved_bias is not None:
                grads.append(Tensor(gb_c))
            return grads

        inputs = [input, self.weight]
        if has_bias:
            inputs.append(self.bias)
        record([output], inputs, backward_fn)

        return output

    def __repr__(self):
        return (f"Linear(in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None})")
