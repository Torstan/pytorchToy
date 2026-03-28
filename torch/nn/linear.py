"""
nn.Linear — 全连接层
y = x @ W^T + b

backward 由 F.linear 算子级 autograd 完成（融合 C++ forward/backward）
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


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._weight_t_cache = None
        self._weight_t_cache_key = None

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

    def _get_packed_weight(self):
        cache_key = (self.weight.data_ptr(), self.weight._version)
        if self._weight_t_cache is None or self._weight_t_cache_key != cache_key:
            packed = _nn_C.transpose_last2(self.weight._c)
            if not packed.is_contiguous():
                packed = packed.contiguous()
            self._weight_t_cache = Tensor(packed)
            self._weight_t_cache_key = cache_key
        return self._weight_t_cache

    def forward(self, input):
        import torch.nn.functional as F
        return F.linear(input, self.weight, self.bias,
                        packed_weight=self._get_packed_weight())

    def __repr__(self):
        return (f"Linear(in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None})")
