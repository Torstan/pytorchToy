"""
nn.CrossEntropyLoss — 交叉熵损失函数
loss = -log(softmax(input))[target]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import _C
import _nn_C

from torch.tensor import Tensor
from torch.nn.module import Module
from torch.autograd_engine import record


class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        """
        input: [N, C] logits
        target: [N] integer class indices (可以是 float 存储的整数)
        """
        loss_c, softmax_c = _nn_C.cross_entropy_forward(input._c, target._c)
        loss = Tensor(loss_c)

        # 记录到 tape
        saved_softmax = softmax_c
        saved_target = target

        def backward_fn(grad_outputs):
            grad_out = grad_outputs[0]
            gi_c = _nn_C.cross_entropy_backward(grad_out._c, saved_softmax, saved_target._c)
            return [Tensor(gi_c)]

        # input 是 loss 的唯一需要梯度的输入（target 不需要）
        record([loss], [input], backward_fn)

        return loss

    def __repr__(self):
        return "CrossEntropyLoss()"
