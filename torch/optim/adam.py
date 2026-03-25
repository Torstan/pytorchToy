"""
Adam 优化器
实现 Adam (Adaptive Moment Estimation) 算法
"""

import sys
import os
import math
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
            self.m.append(Tensor(_C.empty(p.size())))
            self.v.append(Tensor(_C.empty(p.size())))

    def zero_grad(self):
        """清零所有参数的梯度"""
        for p in self.params:
            p.grad = None

    def step(self):
        """执行一步 Adam 更新"""
        self.step_count += 1
        t = self.step_count

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad
            m = self.m[i]
            v = self.v[i]

            n = p.numel()
            for j in range(n):
                g = grad._c.flat_get(j)

                # 更新一阶矩: m = beta1 * m + (1 - beta1) * g
                m_val = self.beta1 * m._c.flat_get(j) + (1 - self.beta1) * g
                m._c.flat_set(j, m_val)

                # 更新二阶矩: v = beta2 * v + (1 - beta2) * g^2
                v_val = self.beta2 * v._c.flat_get(j) + (1 - self.beta2) * g * g
                v._c.flat_set(j, v_val)

                # 偏差修正
                m_hat = m_val / (1 - self.beta1 ** t)
                v_hat = v_val / (1 - self.beta2 ** t)

                # 更新参数: p = p - lr * m_hat / (sqrt(v_hat) + eps)
                p._c.flat_set(j,
                    p._c.flat_get(j) - self.lr * m_hat / (math.sqrt(v_hat) + self.eps))
