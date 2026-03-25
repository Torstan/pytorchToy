"""
Adam 优化器
实现 Adam (Adaptive Moment Estimation) 算法
C++ 内核加速
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import _C
import _nn_C

from torch.tensor import Tensor


class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0

        # 初始化一阶矩和二阶矩
        self.m = []  # 一阶矩 (mean)
        self.v = []  # 二阶矩 (variance)
        for p in self.params:
            self.m.append(_C.empty(p.size()))
            self.v.append(_C.empty(p.size()))

    def zero_grad(self):
        """清零所有参数的梯度"""
        for p in self.params:
            p._c.zero_grad()

    def step(self):
        """执行一步 Adam 更新（C++ 内核）"""
        self.step_count += 1
        t = self.step_count

        for i, p in enumerate(self.params):
            _nn_C.adam_step(p._c, self.m[i], self.v[i],
                           self.lr, self.beta1, self.beta2, self.eps, t)
