"""
nn.Parameter — 标记为可训练参数的 Tensor
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.tensor import Tensor


class Parameter(Tensor):
    """
    Parameter 是 Tensor 的子类，自动标记 requires_grad=True。
    当赋值给 Module 的属性时，会被自动注册到 _parameters 字典中。
    """

    def __init__(self, data, requires_grad=True):
        if isinstance(data, Tensor):
            super().__init__(data._c, requires_grad=requires_grad)
        else:
            super().__init__(data, requires_grad=requires_grad)
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Parameter({self._c.__repr__()})"
